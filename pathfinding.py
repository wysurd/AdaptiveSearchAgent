import heapq


def heuristic(current, goal):
    #manhattan distance
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


def get_neighbors(position, game_map):
    column, row = position

    candidates = [
        (column + 1, row),
        (column - 1, row),
        (column, row + 1),
        (column, row - 1)
    ]

    neighbors = []

    for next_column, next_row in candidates:
        if 0 <= next_row < len(game_map):
            if 0 <= next_column < len(game_map[next_row]):
                if game_map[next_row][next_column] == "0":
                    neighbors.append((next_column, next_row))

    return neighbors


def astar(game_map, start, goal):
    #check start and goal
    for column, row in (start, goal):
        if not 0 <= row < len(game_map):
            return None

        if not 0 <= column < len(game_map[row]):
            return None

        if game_map[row][column] != "0":
            return None

    #prepare search
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)

        #skip outdated entries
        if current_cost != g_score.get(current):
            continue

        #rebuild the completed path
        if current == goal:
            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            path.reverse()
            return path

        #check neighboring tiles
        for neighbor in get_neighbors(current, game_map):
            next_cost = current_cost + 1

            if next_cost < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = next_cost

                priority = next_cost + heuristic(neighbor, goal)

                heapq.heappush(
                    open_set,
                    (priority, next_cost, neighbor)
                )

    return None