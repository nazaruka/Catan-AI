from players.montecarlo_player import MonteCarloPlayer
from players.filters import create_bad_robber_placement_and_monte_carlo_filter

class MonteCarloWithFilterPlayer(MonteCarloPlayer):
    def __init__(self, seed=None, timeout_seconds=None, branching_factor=3459):
        super().__init__(seed=seed,
                         timeout_seconds=timeout_seconds,
                         filter_moves=create_bad_robber_placement_and_monte_carlo_filter(seed, self, branching_factor))
