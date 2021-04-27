from __future__ import division

import time
import math
import numpy as np
import random
import sys
from typing import Callable, List

from algorithms.abstract_state import AbstractState, AbstractMove
from players.abstract_player import AbstractPlayer

class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isFinal = state.is_final()
        self.isFullyExpanded = self.isFinal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):
        s=[]
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isFinal: %s"%(self.isFinal))
        s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))

class mcts():
    def __init__(self, is_maximizing_player: Callable[[AbstractPlayer], bool],
            evaluate_heuristic_value: Callable[[AbstractState], float], timeout_seconds=None, 
            iterationLimit=5, explorationConstant=1 / math.sqrt(2),
            filter_moves: Callable[[List[AbstractMove]], List[AbstractMove]]=lambda l: l):
        self.state = None
        self._is_maximizing_player = is_maximizing_player
        self.filter_moves = filter_moves

        if timeout_seconds != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            self.timeout_seconds = timeout_seconds
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'

        self.explorationConstant = explorationConstant
        self.evaluate_heuristic_value = evaluate_heuristic_value

    def get_best_move(self, state: AbstractState):
        """
        get best move, based on the expectimax with alpha-beta pruning algorithm
        with given heuristic function
        :param state: the Game, an interface with necessary methods
        (see AbstractState for details)
        :param max_depth: the maximum depth the algorithm will reach in the game tree
        :return: the best move
        """
        assert isinstance(state, AbstractState)

        self.state = state
        best_move = self.search()
        return best_move

    def search(self, needDetails=False):
        self.root = treeNode(self.state, None)

        if self.limitType == 'time':
            time_limit = time.time() + self.timeout_seconds
            while time.time() < time_limit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
        reward, _ = self.rollout(node.state, False)
        self.backpropogate(node, reward)

    def rollout(self, state, is_random_event):
        """terminal = 0"""
        
        if self.state.is_final():
            print(self.evaluate_heuristic_value(self.state))
            return self.evaluate_heuristic_value(self.state), None

        """if is_random_event:
            random_move = random.choice(self.state.get_next_random_moves())
            self.state.make_random_move(random_move)
            terminal = self.rollout(state, False)
            self.state.unmake_random_move(random_move)

        else:
            move = random.choice(self.filter_moves(self.state.get_next_moves(), self.state))
            self.state.make_move(move)
            terminal = self.rollout(state, True)
            self.state.unmake_move(move)
        
        print(terminal)
        return terminal"""
        if is_random_event:
            random_move = random.choice(self.state.get_next_random_moves())
            self.state.make_random_move(random_move)
            terminal = self.rollout(state, False)
            self.state.unmake_random_move(random_move)

        elif self._is_maximizing_player(self.state.get_current_player()):
            terminal = -math.inf
            best_move = None
            for move in self.filter_moves(self.state.get_next_moves(), self.state):
                self.state.make_move(move)
                val, _ = self.rollout(state, True)
                if val > terminal:
                    terminal = val
                    best_move = move
                self.state.unmake_move(move)
            return terminal, best_move
        
        else:
            terminal = math.inf
            for move in self.filter_moves(self.state.get_next_moves(), self.state):
                self.state.make_move(move)
                val, _ = self.rollout(state, True)
                terminal = min(terminal, val)
                self.state.unmake_move(move)
            return terminal, None


    def selectNode(self, node):
        while not node.isFinal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.get_next_moves()
        for action in actions:
            if action not in node.children:
                node.state.make_move(action)
                newNode = treeNode(node.state, node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                node.state.unmake_move(action)
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            curr = 1 if self._is_maximizing_player(self.state.get_current_player()) else -1
            nodeValue = curr * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)