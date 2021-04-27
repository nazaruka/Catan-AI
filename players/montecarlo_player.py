import copy
from collections import Counter
from math import ceil
from typing import Dict, Callable, List

from algorithms.abstract_state import AbstractState, AbstractMove
from algorithms.mcts import mcts
from game.catan_state import CatanState
from game.development_cards import DevelopmentCard
from game.pieces import Road, Colony
from game.resource import Resource, ResourceAmounts
from players.abstract_player import AbstractPlayer
from players.random_player import RandomPlayer
from players.filters import create_bad_robber_placement_and_monte_carlo_filter
from train_and_test.logger import logger


class MonteCarloPlayer(AbstractPlayer):
    default_weights = {Colony.City: 10000, Colony.Settlement: 10000, Road.Paved: 10,
                       DevelopmentCard.VictoryPoint: 100, DevelopmentCard.Knight: 100}

    def weighted_probabilities_heuristic(self, s: CatanState):
        if self._players_and_factors is None:
            self._players_and_factors = [(self, len(s.players) - 1)] + [(p, -1) for p in s.players if p is not self]

        score = 0
        # noinspection PyTypeChecker
        for player, factor in self._players_and_factors:
            for location in s.board.get_locations_colonised_by_player(player):
                weight = self.weights[s.board.get_colony_type_at_location(location)]
                for dice_value in s.board.get_surrounding_dice_values(location):
                    score += s.probabilities_by_dice_values[dice_value] * weight * factor

            for road in s.board.get_roads_paved_by_player(player):
                weight = self.weights[Road.Paved]
                for dice_value in s.board.get_adjacent_to_path_dice_values(road):
                    score += s.probabilities_by_dice_values[dice_value] * weight * factor

            for development_card in {DevelopmentCard.VictoryPoint, DevelopmentCard.Knight}:
                weight = self.weights[development_card]
                score += self.get_unexposed_development_cards()[development_card] * weight * factor
        return score

    def __init__(self, seed=None, timeout_seconds=None, heuristic=None, weights=default_weights, branching_factor=3459, filter_moves=lambda x, y: x):
        super().__init__(seed, timeout_seconds)
        self.weights = weights
        self._players_and_factors = None

        if heuristic is None:
            heuristic = self.weighted_probabilities_heuristic

        self.montecarlo = mcts(is_maximizing_player=lambda p: p is self, 
            evaluate_heuristic_value=heuristic,
            filter_moves=filter_moves)

    def choose_move(self, state: CatanState):
        best_move = self.montecarlo.get_best_move(state)
        if best_move is not None:
            return best_move
        else:
            logger.warning('returning a random move')
            return RandomPlayer.choose_move(self, state)

    def choose_resources_to_drop(self) -> Dict[Resource, int]:
        if sum(self.resources.values()) < 8:
            return {}
        resources_count = sum(self.resources.values())
        resources_to_drop_count = ceil(resources_count / 2)
        if self.can_settle_city() and resources_count >= sum(ResourceAmounts.city.values()) * 2:
            self.remove_resources_and_piece_for_city()
            resources_to_drop = copy.deepcopy(self.resources)
            self.add_resources_and_piece_for_city()

        elif self.can_settle_settlement() and resources_count >= sum(ResourceAmounts.settlement.values()) * 2:
            self.remove_resources_and_piece_for_settlement()
            resources_to_drop = copy.deepcopy(self.resources)
            self.add_resources_and_piece_for_settlement()

        elif (self.has_resources_for_development_card() and
              resources_count >= sum(ResourceAmounts.development_card.values()) * 2):
            self.remove_resources_for_development_card()
            resources_to_drop = copy.deepcopy(self.resources)
            self.add_resources_for_development_card()

        elif self.can_pave_road() and resources_count >= sum(ResourceAmounts.road.values()) * 2:
            self.remove_resources_and_piece_for_road()
            resources_to_drop = copy.deepcopy(self.resources)
            self.add_resources_and_piece_for_road()

        else:
            return RandomPlayer.choose_resources_to_drop(self)

        resources_to_drop = [resource for resource, count in resources_to_drop.items() for _ in range(count)]
        return Counter(self._random_choice(resources_to_drop, resources_to_drop_count, replace=False))

    def set_heuristic(self, evaluate_heuristic_value: Callable[[AbstractState], float]):
        """
        set heuristic evaluation of a state in a game
        :param evaluate_heuristic_value: a callable that given state returns a float. higher means "better" state
        """
        self.montecarlo.evaluate_heuristic_value = evaluate_heuristic_value

    def set_filter(self, filter_moves: Callable[[List[AbstractMove]], List[AbstractMove]]):
        """
        set the filtering of moves in each step
        :param filter_moves: a callable that given list of moves, returns a list of moves that will be further developed
        """
        self.montecarlo.filter_moves = filter_moves