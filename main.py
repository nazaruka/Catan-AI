import os
import time
import matplotlib.pyplot as plt
import numpy as np

from game.catan_state import CatanState
from players.montecarlo_with_filter_player import MonteCarloWithFilterPlayer
from players.random_player import RandomPlayer
from train_and_test.logger import logger, fileLogger

p0_scores = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
p1_scores = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
winners = []

def scores_changed(state, previous_scores, scores):
    for player in state.players:
        if previous_scores[player] != scores[player]:
            return True
    return False

def clean_previous_images():
    for file_name in os.listdir(path='.'):
        if file_name.split(sep='_')[0] == 'turn':
            os.remove(file_name)

def execute_game(plot_map=True):
    seed = None
    p0 = MonteCarloWithFilterPlayer(seed)
    p1 = RandomPlayer(seed)
    players = [p0, p1]

    state = CatanState(players, seed)

    turn_count = 0
    score_by_player = state.get_scores_by_player()
    if plot_map:
        state.board.plot_map('turn_{}_scores_{}.png'
                             .format(turn_count, ''.join('{}_'.format(v) for v in score_by_player.values())))

    while not state.is_final():
        # noinspection PyProtectedMember
        logger.info('----------------------p{}\'s turn----------------------'.format(state._current_player_index))

        turn_count += 1
        robber_placement = state.board.get_robber_land()

        move = state.get_current_player().choose_move(state)
        assert not scores_changed(state, score_by_player, state.get_scores_by_player())
        state.make_move(move)
        state.make_random_move()

        score_by_player = state.get_scores_by_player()

        move_data = {k: v for k, v in move.__dict__.items() if (v and k != 'resources_updates') and not
                     (k == 'robber_placement_land' and v == robber_placement) and not
                     (isinstance(v, dict) and sum(v.values()) == 0)}
        logger.info('| {}| turn: {:3} | move:{} |'.format(''.join('{} '.format(v) for v in score_by_player.values()),
                                                          turn_count, move_data))
        if plot_map and (turn_count == 4 or turn_count % 50 == 0):
            image_name = 'turn_4_scores_{}.png'.format(
                turn_count, ''.join('{}_'.format(v) for v in score_by_player.values()))
            state.board.plot_map(image_name, state.current_dice_number)

    if plot_map:
        state.board.plot_map('turn_{}_scores_{}.png'
                             .format(turn_count, ''.join('{}_'.format(v) for v in score_by_player.values())))
    
    players_scores_by_names = {(k, v.__class__): score_by_player[v]
                               for k, v in locals().items() if v in players
                               }
    fileLogger.info('\n' + '\n'.join(' {:150} : {} '.format(str(name), score)
                                     for name, score in players_scores_by_names.items()) +
                    '\n turns it took: {}\n'.format(turn_count) + ('-' * 156))

    p0_type = type(p0).__name__
    p_others_type = type(p1).__name__

def get_scores():
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
 
    p0 = list(p0_scores.values())
    p1 = list(p1_scores.values())
 
    br1 = np.arange(len(p0))
    br2 = [x + barWidth for x in br1]
 
    plt.bar(br1, p0, color ='r', width = barWidth,
        edgecolor ='grey', label ='IT')
    plt.bar(br2, p1, color ='b', width = barWidth,
        edgecolor ='grey', label ='ECE')
 
    plt.xlabel('Branch', fontweight ='bold', fontsize = 15)
    plt.ylabel('Students passed', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(IT))],
        list(p0_scores.keys()))
 
    plt.xlabel("Victory points")
    plt.ylabel("No. of games played")
    plt.title("MCTS vs. random performance")
    plt.savefig('results.png', dpi = 100)


def run_10_games_parallel():
    global A, B, C, D, E
    import multiprocessing

    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default

    pool = multiprocessing.Pool(processes=cpus)
    pool.map(execute_game, [False] * 10)

    # for _ in range(10):
    #     clean_previous_images()
    #     execute_game(None)
    flush_to_excel()


def run_single_game_and_plot_map():
    clean_previous_images()
    execute_game(plot_map=True)
    get_scores()


if __name__ == '__main__':
    run_single_game_and_plot_map()