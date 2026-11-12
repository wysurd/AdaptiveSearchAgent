import csv
import random
import statistics
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
EXPERIMENT_SEED = 2026
SCENARIO_COUNT = 40
STEP_SECONDS = 0.25
SEARCH_STEPS = 48
VISION_RANGE = 240
COMMITMENT_STEPS = 4
PLAYER_HOLD_STEPS = 2

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
    column, row = tile

    return (
        column * TILE_SIZE + TILE_SIZE / 2,
        row * TILE_SIZE + TILE_SIZE / 2
    )


def can_see(enemy_tile, target_tile):
    #check distance and blocking walls
    start = tile_center(enemy_tile)
    end = tile_center(target_tile)

    distance_squared = (
        (end[0] - start[0]) ** 2
        + (end[1] - start[1]) ** 2
    )

    if distance_squared > VISION_RANGE ** 2:
        return False

    return not any(
        wall.clipline(start, end) for wall in walls
    )


def generate_scenarios():
    #generate repeatable loss-of-sight snapshots
    generator = random.Random(EXPERIMENT_SEED)
    scenarios = []
    used_snapshots = set()

    for attempt in range(20000):
        if len(scenarios) >= SCENARIO_COUNT:
            break

        enemy_start = generator.choice(floor_tiles)

        #choose a nearby visible player location
        visible_starts = [
            tile for tile in floor_tiles
            if 2 <= (
                abs(tile[0] - enemy_start[0])
                + abs(tile[1] - enemy_start[1])
            ) <= 5
            and can_see(enemy_start, tile)
        ]

        if not visible_starts:
            continue

        player_start = generator.choice(visible_starts)
        destination = generator.choice(floor_tiles)

        route = astar(game_map, player_start, destination)

        if route is None or len(route) < 4:
            continue

        #find the first tile beyond enemy vision
        hidden_index = None

        for index in range(1, len(route)):
            if not can_see(enemy_start, route[index]):
                hidden_index = index
                break

        if hidden_index is None:
            continue

        last_seen = route[hidden_index - 1]
        hidden_start = route[hidden_index]

        #avoid starting directly next to the enemy
        separation = (
            abs(hidden_start[0] - enemy_start[0])
            + abs(hidden_start[1] - enemy_start[1])
        )

        if separation < 3:
            continue

        snapshot = (enemy_start, last_seen, hidden_start)

        if snapshot in used_snapshots:
            continue

        #continue along the escape route
        escape_route = route[hidden_index:]

        #some players hide while others keep moving
        if len(scenarios) % 2 == 0:
            behavior = "hide"
            escape_route = [hidden_start]

        else:
            behavior = "move"

            extra_goal = generator.choice(floor_tiles)
            extra_route = astar(
                game_map,
                escape_route[-1],
                extra_goal
            )

            if extra_route is not None:
                escape_route.extend(extra_route[1:])

        positions = []

        for tile in escape_route:
            positions.extend([tile] * PLAYER_HOLD_STEPS)

        scenarios.append({
            "name": f"scenario_{len(scenarios) + 1:02d}",
            "behavior": behavior,
            "enemy_start": enemy_start,
            "last_seen": last_seen,
            "hidden_start": hidden_start,
            "positions": positions
        })

        used_snapshots.add(snapshot)

    if len(scenarios) != SCENARIO_COUNT:
        raise RuntimeError(
            f"generated only {len(scenarios)} scenarios"
        )

    return scenarios


def summarize_trial(
    mode,
    reacquired,
    seconds,
    distance,
    samples,
    top_tile_hits
):
    #use blanks when a metric does not apply
    return {
        "mode": mode,
        "reacquired": int(reacquired),
        "reacquisition_seconds": seconds,
        "enemy_tiles_traveled": distance,
        "mean_true_tile_probability": (
            sum(samples) / len(samples) if samples else ""
        ),
        "top_tile_accuracy": (
            top_tile_hits / len(samples) if samples else ""
        )
    }


def run_trial(scenario, mode):
    #give each strategy identical initial information
    enemy_tile = scenario["enemy_start"]
    last_seen = scenario["last_seen"]
    positions = scenario["positions"]

    belief = Belief(game_map)
    belief.observe_player(last_seen)

    distance = 0
    samples = []
    top_tile_hits = 0

    checked_last_seen = False
    committed_goal = None
    commitment_left = 0

    for step in range(SEARCH_STEPS):
        player_tile = positions[min(step, len(positions) - 1)]

        #check visibility before movement
        if can_see(enemy_tile, player_tile):
            return summarize_trial(
                mode,
                True,
                step * STEP_SECONDS,
                distance,
                samples,
                top_tile_hits
            )

        if mode != "baseline":
            #predict without using the hidden player tile
            belief.predict()

            visible_tiles = [
                tile for tile in floor_tiles
                if can_see(enemy_tile, tile)
            ]

            belief.observe_empty(visible_tiles)
            most_likely = belief.most_likely(enemy_tile)

            #use the true tile only to score the belief
            samples.append(belief.probabilities[player_tile])

            if most_likely == player_tile:
                top_tile_hits += 1

        #select a destination
        if mode == "baseline":
            goal = last_seen

        elif mode == "probabilistic":
            goal = most_likely

        else:
            if enemy_tile == last_seen:
                checked_last_seen = True

            if not checked_last_seen:
                goal = last_seen

            elif mode == "hybrid_no_commitment":
                goal = most_likely

            else:
                if (
                    committed_goal is None
                    or enemy_tile == committed_goal
                    or commitment_left <= 0
                ):
                    committed_goal = most_likely
                    commitment_left = COMMITMENT_STEPS

                goal = committed_goal
                commitment_left -= 1

        #move one tile along an a-star route
        path = astar(game_map, enemy_tile, goal)

        if path is not None and len(path) > 1:
            enemy_tile = path[1]
            distance += 1

        #check visibility after movement
        if can_see(enemy_tile, player_tile):
            return summarize_trial(
                mode,
                True,
                (step + 1) * STEP_SECONDS,
                distance,
                samples,
                top_tile_hits
            )

    return summarize_trial(
        mode,
        False,
        "",
        distance,
        samples,
        top_tile_hits
    )


def save_csv(filename, rows):
    #save one experiment table
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


def main():
    scenarios = generate_scenarios()
    results = []

    print(
        f"seed: {EXPERIMENT_SEED} | "
        f"scenarios: {len(scenarios)} | "
        f"budget: {SEARCH_STEPS * STEP_SECONDS:.0f}s"
    )

    for scenario in scenarios:
        for mode in MODES:
            result = run_trial(scenario, mode)

            results.append({
                "experiment_seed": EXPERIMENT_SEED,
                "scenario": scenario["name"],
                "behavior": scenario["behavior"],
                "enemy_start": str(scenario["enemy_start"]),
                "last_seen": str(scenario["last_seen"]),
                "hidden_start": str(scenario["hidden_start"]),
                **result
            })

    #summarize all scenarios and each behavior
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

            successes = [
                row for row in selected
                if row["reacquired"]
            ]

            times = [
                row["reacquisition_seconds"]
                for row in successes
            ]

            median_time = statistics.median(times) if times else ""
            success_rate = len(successes) / len(selected)

            #count failures as using the full search budget
            capped_times = [
                row["reacquisition_seconds"]
                if row["reacquired"]
                else SEARCH_STEPS * STEP_SECONDS
                for row in selected
            ]

            mean_capped_time = statistics.mean(capped_times)

            summary.append({
                "behavior": behavior,
                "mode": mode,
                "scenarios": len(selected),
                "reacquired": len(successes),
                "reacquisition_rate": success_rate,
                "median_seconds_successes_only": median_time,
                "mean_capped_search_seconds": mean_capped_time,
                "mean_enemy_tiles_traveled": statistics.mean(
                    row["enemy_tiles_traveled"]
                    for row in selected
                )
            })

            median_label = (
                f"{median_time:.2f}s" if times else "n/a"
            )

            print(
                f"{mode}: {len(successes)}/{len(selected)} "
                f"({success_rate:.1%}) | "
                f"median successful time: {median_label} | "
                f"mean capped time: {mean_capped_time:.2f}s"
            )

    #preserve earlier experiment files
    save_csv("evaluation_expanded_results.csv", results)
    save_csv("evaluation_expanded_summary.csv", summary)

    print("\nsaved evaluation_expanded_results.csv")
    print("saved evaluation_expanded_summary.csv")


if __name__ == "__main__":
    main()