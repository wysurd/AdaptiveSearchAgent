import math

from pathfinding import get_neighbors


class Belief:
    def __init__(self, game_map):
        self.game_map = game_map

        self.tiles = [
            (column, row)
            for row, line in enumerate(game_map)
            for column, value in enumerate(line)
            if value == "0"
        ]

        self.neighbors = {
            tile: get_neighbors(tile, game_map)
            for tile in self.tiles
        }

        self.reset()

    def reset(self):
        #start with equal probabilities
        probability = 1 / len(self.tiles)

        self.probabilities = {
            tile: probability for tile in self.tiles
        }

    def normalize(self):
        #keep total probability equal to one
        total = sum(self.probabilities.values())

        if total > 0:
            for tile in self.tiles:
                self.probabilities[tile] /= total
        else:
            self.reset()

    def observe_player(self, tile):
        #record a direct sighting
        self.probabilities = {
            position: 0.0 for position in self.tiles
        }

        self.probabilities[tile] = 1.0

    def predict(self):
        #apply one markov transition
        predicted = {
            tile: 0.0 for tile in self.tiles
        }

        for tile, probability in self.probabilities.items():
            neighbors = self.neighbors[tile]

            if not neighbors:
                predicted[tile] += probability
                continue

            predicted[tile] += probability * 0.2
            moving_probability = probability * 0.8 / len(neighbors)

            for neighbor in neighbors:
                predicted[neighbor] += moving_probability

        self.probabilities = predicted

    def observe_empty(self, visible_tiles):
        #remove visibly empty locations
        visible_tiles = set(visible_tiles)

        for tile in visible_tiles:
            self.probabilities[tile] = 0.0

        total = sum(self.probabilities.values())

        if total > 0:
            self.normalize()
            return

        #recover across unseen locations
        candidates = [
            tile for tile in self.tiles
            if tile not in visible_tiles
        ]

        if not candidates:
            self.reset()
            return

        candidate_set = set(candidates)
        probability = 1 / len(candidates)

        self.probabilities = {
            tile: probability if tile in candidate_set else 0.0
            for tile in self.tiles
        }

    def observe_sound(self, heard_tile):
        #combine prior belief with uncertain sound evidence
        uniform = 1 / len(self.tiles)
        sigma = 2.0

        for tile in self.tiles:
            distance_squared = (
                (tile[0] - heard_tile[0]) ** 2
                + (tile[1] - heard_tile[1]) ** 2
            )

            likelihood = 0.01 + math.exp(
                -distance_squared / (2 * sigma ** 2)
            )

            #allow recovery from an incorrect prior
            prior = (
                self.probabilities[tile] * 0.95
                + uniform * 0.05
            )

            self.probabilities[tile] = prior * likelihood

        self.normalize()

    def most_likely(self, enemy_tile):
        #prefer nearby tiles when probabilities tie
        return max(
            self.tiles,
            key=lambda tile: (
                self.probabilities[tile],
                -(
                    abs(tile[0] - enemy_tile[0])
                    + abs(tile[1] - enemy_tile[1])
                )
            )
        )