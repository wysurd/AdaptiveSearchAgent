import csv

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

#evaluation settings
STEP_SECONDS = 0.25
SEARCH_STEPS = 48
VISION_RANGE = 240
COMMITMENT_STEPS = 4

ENEMY_START = (3, 4)
LAST_SEEN = (6, 4)

#build wall rectangles
walls = []

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


def make_player_positions(waypoints):
    #build a valid scripted route
    route = [waypoints[0]]

    for start, goal in zip(waypoints, waypoints[1:]):
        path = astar(game_map, start, goal)

        if path is None:
            raise ValueError(
                f"no player route between {start} and {goal}"
            )

        route.extend(path[1:])

    #hold each tile for two simulation steps
    positions = []

    for tile in route:
        positions.extend([tile, tile])

    return positions


def make_result(
    mode,
    reacquired,
    reacquisition_seconds,
    distance_traveled,
    belief_samples,
    top_tile_hits
):
    #summarize one trial
    if belief_samples:
        mean_probability = (
            sum(belief_samples) / len(belief_samples)
        )

        top_tile_rate = top_tile_hits / len(belief_samples)

    else:
        mean_probability = ""
        top_tile_rate = ""

    return {
        "mode": mode,
        "reacquired": int(reacquired),
        "reacquisition_seconds": reacquisition_seconds,
        "enemy_tiles_traveled": distance_traveled,
        "mean_true_tile_probability": mean_probability,
        "top_tile_accuracy": top_tile_rate
    }


def run_trial(mode, player_positions):
    #start with identical information
    enemy_tile = ENEMY_START
    belief = Belief(game_map)
    belief.observe_player(LAST_SEEN)

    distance_traveled = 0
    belief_samples = []
    top_tile_hits = 0

    checked_last_seen = False
    committed_goal = None
    commitment_left = 0

    for step in range(SEARCH_STEPS):
        player_tile = player_positions[
            min(step, len(player_positions) - 1)
        ]

        #check visibility before movement
        if can_see(enemy_tile, player_tile):
            return make_result(
                mode,
                True,
                step * STEP_SECONDS,
                distance_traveled,
                belief_samples,
                top_tile_hits
            )

        if mode != "baseline":
            #predict and apply negative evidence
            belief.predict()

            visible_tiles = [
                tile for tile in belief.tiles
                if can_see(enemy_tile, tile)
            ]

            belief.observe_empty(visible_tiles)
            most_likely_tile = belief.most_likely(enemy_tile)

            #score the belief without guiding the enemy
            belief_samples.append(
                belief.probabilities[player_tile]
            )

            if most_likely_tile == player_tile:
                top_tile_hits += 1

        #choose the search destination
        if mode == "baseline":
            goal = LAST_SEEN

        elif mode == "probabilistic":
            goal = most_likely_tile

        else:
            #check the last sighting first
            if enemy_tile == LAST_SEEN:
                checked_last_seen = True

            if not checked_last_seen:
                goal = LAST_SEEN

            else:
                #hold a goal for up to one second
                if (
                    committed_goal is None
                    or enemy_tile == committed_goal
                    or commitment_left <= 0
                ):
                    committed_goal = most_likely_tile
                    commitment_left = COMMITMENT_STEPS

                goal = committed_goal
                commitment_left -= 1

        #move one tile along the selected route
        path = astar(game_map, enemy_tile, goal)

        if path is not None and len(path) > 1:
            enemy_tile = path[1]
            distance_traveled += 1

        #check visibility after movement
        if can_see(enemy_tile, player_tile):
            return make_result(
                mode,
                True,
                (step + 1) * STEP_SECONDS,
                distance_traveled,
                belief_samples,
                top_tile_hits
            )

    return make_result(
        mode,
        False,
        "",
        distance_traveled,
        belief_samples,
        top_tile_hits
    )


def main():
    #use the same scenarios as the first experiment
    scenarios = {
        "nearby_hide": [
            (8, 3)
        ],
        "cross_upper_rooms": [
            (8, 3),
            (12, 2),
            (20, 2)
        ],
        "move_to_lower_rooms": [
            (8, 3),
            (11, 8),
            (13, 12)
        ],
        "return_toward_last_seen": [
            (8, 3),
            (10, 4),
            (6, 4)
        ]
    }

    modes = ("baseline", "probabilistic", "hybrid")
    results = []

    for name, waypoints in scenarios.items():
        player_positions = make_player_positions(waypoints)

        if can_see(ENEMY_START, player_positions[0]):
            raise ValueError(
                f"{name} starts with the player visible"
            )

        for mode in modes:
            result = run_trial(mode, player_positions)
            result = {"scenario": name, **result}
            results.append(result)

            seconds = result["reacquisition_seconds"]

            if result["reacquired"]:
                outcome = f"reacquired in {seconds:.2f} seconds"
            else:
                outcome = "not reacquired within 12 seconds"

            print(
                f"{name} | {mode} | {outcome} | "
                f"tiles traveled: "
                f"{result['enemy_tiles_traveled']}"
            )

    #preserve the original experiment file
    filename = "evaluation_hybrid_results.csv"

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nreacquisition summary")

    for mode in modes:
        mode_results = [
            result for result in results
            if result["mode"] == mode
        ]

        successes = sum(
            result["reacquired"] for result in mode_results
        )

        print(
            f"{mode}: {successes}/{len(mode_results)} scenarios"
        )

    print(f"\nsaved {filename}")


if __name__ == "__main__":
    main()