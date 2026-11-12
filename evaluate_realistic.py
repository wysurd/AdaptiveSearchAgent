import csv
import random
import statistics
from functools import lru_cache
from pathlib import Path

import pygame

from belief import Belief
from pathfinding import astar

#map settings
TILE_SIZE = 40

game_map = [
    "111111111111111111111111",
    "100000010000000010000001",
    "100000010000000010000001",
    "100000010000000010000001",
    "100000000000000000000001",
    "100000010000000010000001",
    "100000010000000010000001",
    "111011111110111111110111",
    "100000000000000000000001",
    "111110111111110111011111",
    "100000010000000010000001",
    "100000010000000010000001",
    "100000000000000000000001",
    "100000010000000010000001",
    "100000010000000010000001",
    "111111111111111111111111"
]

#experiment settings
SEED = 2026
SCENARIO_COUNT = 40
DT = 0.05
SEARCH_BUDGET = 12.0
PREDICTION_INTERVAL = 0.25
ENEMY_SPEED = 240
PLAYER_SPEED = 220
VISION_RANGE = 240
COMMITMENT_SECONDS = 1.0

MODES = [
    "baseline",
    "probabilistic",
    "hybrid_no_commitment",
    "hybrid"
]

output_folder = Path(__file__).resolve().parent

#build the map
walls = []
floor_tiles = []

for row_index, row in enumerate(game_map):
    for column_index, tile in enumerate(row):
        if tile == "1":
            walls.append(
                pygame.Rect(
                    column_index * TILE_SIZE,
                    row_index * TILE_SIZE,
                    TILE_SIZE,
                    TILE_SIZE
                )
            )
        else:
            floor_tiles.append((column_index, row_index))


def tile_center(tile):
    #convert a tile to its pixel center
    return pygame.Vector2(
        tile[0] * TILE_SIZE + TILE_SIZE / 2,
        tile[1] * TILE_SIZE + TILE_SIZE / 2
    )


def position_tile(position):
    #convert pixels to grid coordinates
    return (
        int(position.x // TILE_SIZE),
        int(position.y // TILE_SIZE)
    )


def can_see(start, end):
    #check continuous positions
    if start.distance_squared_to(end) > VISION_RANGE ** 2:
        return False

    line_start = (round(start.x), round(start.y))
    line_end = (round(end.x), round(end.y))

    return not any(
        wall.clipline(line_start, line_end)
        for wall in walls
    )


@lru_cache(maxsize=None)
def visible_from_center(tile):
    #cache tile-center negative evidence
    position = tile_center(tile)

    return tuple(
        target for target in floor_tiles
        if can_see(position, tile_center(target))
    )


def follow_path(position, path, speed, elapsed):
    #move continuously through tile centers
    remaining = speed * elapsed

    while path and remaining > 0:
        target = tile_center(path[0])
        offset = target - position
        distance = offset.length()

        if distance <= remaining:
            position = target
            remaining -= distance
            path.pop(0)
        else:
            position += offset.normalize() * remaining
            remaining = 0

    return position


def generate_scenarios():
    #generate scenarios without using strategy outcomes
    generator = random.Random(SEED)
    scenarios = []
    used = set()

    for attempt in range(30000):
        if len(scenarios) == SCENARIO_COUNT:
            break

        enemy_start = generator.choice(floor_tiles)
        enemy_position = tile_center(enemy_start)

        visible_starts = [
            tile for tile in floor_tiles
            if 2 <= (
                abs(tile[0] - enemy_start[0])
                + abs(tile[1] - enemy_start[1])
            ) <= 5
            and can_see(enemy_position, tile_center(tile))
        ]

        if not visible_starts:
            continue

        player_start = generator.choice(visible_starts)
        destination = generator.choice(floor_tiles)
        route = astar(game_map, player_start, destination)

        if route is None:
            continue

        hidden_index = None

        for index in range(1, len(route)):
            if not can_see(enemy_position, tile_center(route[index])):
                hidden_index = index
                break

        if hidden_index is None:
            continue

        #allow a longer escape before search begins
        extra_steps = generator.randint(3, 6)
        search_start_index = hidden_index + extra_steps

        if search_start_index >= len(route):
            continue

        hidden_prefix = route[
            hidden_index:search_start_index + 1
        ]

        if any(
            can_see(enemy_position, tile_center(tile))
            for tile in hidden_prefix
        ):
            continue

        #check intermediate positions during the escape
        last_seen = route[hidden_index - 1]
        escaped_route = route[
            hidden_index - 1:search_start_index + 1
        ]

        valid_escape = True

        #the first segment crosses out of sight
        for start, end in zip(
            escaped_route[1:],
            escaped_route[2:]
        ):
            start_position = tile_center(start)
            offset = tile_center(end) - start_position

            for fraction in (0.25, 0.5, 0.75):
                sample = start_position + offset * fraction

                if can_see(enemy_position, sample):
                    valid_escape = False
                    break

            if not valid_escape:
                break

        if not valid_escape:
            continue

        hidden_start = route[search_start_index]
        snapshot = (enemy_start, last_seen, hidden_start)

        if snapshot in used:
            continue

        behavior = (
            "hide" if len(scenarios) % 2 == 0 else "move"
        )

        player_route = []

        if behavior == "move":
            player_route = route[search_start_index + 1:]

            extra_goal = generator.choice(floor_tiles)
            extra_route = astar(
                game_map,
                route[-1],
                extra_goal
            )

            if extra_route is not None:
                player_route.extend(extra_route[1:])

        #approximate age from center-to-center travel
        unseen_seconds = (
            (search_start_index - hidden_index + 1)
            * TILE_SIZE
            / PLAYER_SPEED
        )

        scenarios.append({
            "name": f"scenario_{len(scenarios) + 1:02d}",
            "behavior": behavior,
            "enemy_start": enemy_start,
            "last_seen": last_seen,
            "hidden_start": hidden_start,
            "unseen_seconds": unseen_seconds,
            "player_route": player_route
        })

        used.add(snapshot)

    if len(scenarios) != SCENARIO_COUNT:
        raise RuntimeError(
            f"generated only {len(scenarios)} scenarios"
        )

    return scenarios


def run_trial(scenario, mode):
    #initialize identical search snapshots
    enemy_position = tile_center(scenario["enemy_start"])
    player_position = tile_center(scenario["hidden_start"])

    player_path = list(scenario["player_route"])
    enemy_path = []
    enemy_goal = None
    last_seen = scenario["last_seen"]

    belief = Belief(game_map)
    belief.observe_player(last_seen)

    #advance belief through the unseen interval
    age = scenario["unseen_seconds"]
    prediction_steps = int(age / PREDICTION_INTERVAL)

    for _ in range(prediction_steps):
        belief.predict()

    prediction_timer = age % PREDICTION_INTERVAL
    checked_last_seen = False
    committed_goal = None
    commitment_timer = 0
    distance_traveled = 0

    steps = round(SEARCH_BUDGET / DT)

    for step in range(steps):
        elapsed = step * DT

        #check actual continuous visibility
        if can_see(enemy_position, player_position):
            return {
                "reacquired": 1,
                "reacquisition_seconds": round(elapsed, 2),
                "enemy_pixels_traveled": round(distance_traveled, 2)
            }

        prediction_timer += DT
        commitment_timer = max(0, commitment_timer - DT)

        if mode != "baseline":
            while prediction_timer >= PREDICTION_INTERVAL:
                belief.predict()
                prediction_timer -= PREDICTION_INTERVAL

        #make decisions at tile centers
        if not enemy_path:
            enemy_tile = position_tile(enemy_position)

            if mode != "baseline":
                belief.observe_empty(
                    visible_from_center(enemy_tile)
                )

                most_likely = belief.most_likely(enemy_tile)

            if mode == "baseline":
                enemy_goal = last_seen

            elif mode == "probabilistic":
                enemy_goal = most_likely

            else:
                if enemy_tile == last_seen:
                    checked_last_seen = True

                if not checked_last_seen:
                    enemy_goal = last_seen

                elif mode == "hybrid_no_commitment":
                    enemy_goal = most_likely

                else:
                    if (
                        committed_goal is None
                        or enemy_tile == committed_goal
                        or commitment_timer <= 0
                    ):
                        committed_goal = most_likely
                        commitment_timer = COMMITMENT_SECONDS

                    enemy_goal = committed_goal

            path = astar(
                game_map,
                enemy_tile,
                enemy_goal
            )

            #commit only to the next movement segment
            if path is not None and len(path) > 1:
                enemy_path = [path[1]]

        #advance both characters by the same timestep
        previous_enemy = enemy_position.copy()

        enemy_position = follow_path(
            enemy_position,
            enemy_path,
            ENEMY_SPEED,
            DT
        )

        player_position = follow_path(
            player_position,
            player_path,
            PLAYER_SPEED,
            DT
        )

        distance_traveled += enemy_position.distance_to(
            previous_enemy
        )

        if can_see(enemy_position, player_position):
            return {
                "reacquired": 1,
                "reacquisition_seconds": round(elapsed + DT, 2),
                "enemy_pixels_traveled": round(distance_traveled, 2)
            }

    return {
        "reacquired": 0,
        "reacquisition_seconds": "",
        "enemy_pixels_traveled": round(distance_traveled, 2)
    }


def save_csv(filename, rows):
    #save without replacing earlier experiment files
    with (output_folder / filename).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def paired_outcome(first, second):
    #compare success first and time second
    if first["reacquired"] != second["reacquired"]:
        return (
            "win" if first["reacquired"] else "loss"
        )

    if not first["reacquired"]:
        return "tie"

    difference = (
        first["reacquisition_seconds"]
        - second["reacquisition_seconds"]
    )

    if difference < -DT / 2:
        return "win"

    if difference > DT / 2:
        return "loss"

    return "tie"


def main():
    scenarios = generate_scenarios()
    results = []
    lookup = {}

    print(
        f"seed: {SEED} | scenarios: {SCENARIO_COUNT} | "
        f"enemy: {ENEMY_SPEED} | player: {PLAYER_SPEED}"
    )

    for index, scenario in enumerate(scenarios, start=1):
        for mode in MODES:
            outcome = run_trial(scenario, mode)

            row = {
                "experiment_seed": SEED,
                "scenario": scenario["name"],
                "behavior": scenario["behavior"],
                "enemy_start": str(scenario["enemy_start"]),
                "last_seen": str(scenario["last_seen"]),
                "hidden_start": str(scenario["hidden_start"]),
                "unseen_seconds": round(
                    scenario["unseen_seconds"], 3
                ),
                "mode": mode,
                **outcome
            }

            results.append(row)
            lookup[(scenario["name"], mode)] = row

        if index % 10 == 0:
            print(f"completed {index}/{SCENARIO_COUNT} scenarios")

    summary = []

    for behavior in ("all", "hide", "move"):
        print(f"\n{behavior} scenarios")

        for mode in MODES:
            selected = [
                row for row in results
                if row["mode"] == mode
                and (
                    behavior == "all"
                    or row["behavior"] == behavior
                )
            ]

            successful_times = [
                row["reacquisition_seconds"]
                for row in selected
                if row["reacquired"]
            ]

            successes = len(successful_times)
            median = (
                statistics.median(successful_times)
                if successful_times else ""
            )

            capped_mean = statistics.mean(
                row["reacquisition_seconds"]
                if row["reacquired"] else SEARCH_BUDGET
                for row in selected
            )

            summary.append({
                "behavior": behavior,
                "mode": mode,
                "scenarios": len(selected),
                "reacquired": successes,
                "reacquisition_rate": successes / len(selected),
                "median_seconds_successes_only": median,
                "mean_capped_search_seconds": capped_mean
            })

            median_label = (
                f"{median:.2f}s" if successful_times else "n/a"
            )

            print(
                f"{mode}: {successes}/{len(selected)} | "
                f"median successful time: {median_label} | "
                f"mean capped time: {capped_mean:.2f}s"
            )

    paired_rows = []

    print("\nhybrid paired comparisons")

    for opponent in MODES:
        if opponent == "hybrid":
            continue

        counts = {"win": 0, "loss": 0, "tie": 0}

        for scenario in scenarios:
            name = scenario["name"]

            outcome = paired_outcome(
                lookup[(name, "hybrid")],
                lookup[(name, opponent)]
            )

            counts[outcome] += 1

        paired_rows.append({
            "strategy": "hybrid",
            "opponent": opponent,
            **counts
        })

        print(
            f"against {opponent}: "
            f"{counts['win']} wins, "
            f"{counts['loss']} losses, "
            f"{counts['tie']} ties"
        )

    save_csv("evaluation_realistic_results.csv", results)
    save_csv("evaluation_realistic_summary.csv", summary)
    save_csv("evaluation_realistic_pairs.csv", paired_rows)

    print("\nsaved realistic results, summary, and pairs")


if __name__ == "__main__":
    main()