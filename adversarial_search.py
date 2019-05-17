# -----------------------------------------------------------------------------
# Name:     adversarial_search
# Purpose:  Homework 5 - Implement adversarial search algorithms
#
# Author: Jenil Thakker
#
# -----------------------------------------------------------------------------
"""
Adversarial search algorithms implementation

Your task for homework 5 is to implement:
1.  minimax
2.  alphabeta
3.  abdl (alpha beta depth limited)
"""
import random
import sys

def rand(game_state):
    """
    Generate a random move.
    :param game_state: GameState object
    :return:  a tuple representing the row column of the random move
    """
    done = False
    while not done:
        row = random.randint(0, game_state.size - 1)
        col = random.randint(0, game_state.size - 1)
        if game_state.available(row,col):
            done = True
    return row, col


def minimax(game_state):
    """
    Find the best move for our AI agent by applying the minimax algorithm
    (searching the entire tree from the current game state)
    :param game_state: GameState object
    :return:  a tuple representing the row column of the best move
    """
    v = None
    for move in game_state.possible_moves():
        if not v:
            v = (value(game_state.successor(move, 'AI'), 'user'), move)
        else:
            val, spot = v
            if value(game_state.successor(move, 'AI'), 'user') > val:
                v = (value(game_state.successor(move, 'AI'), 'user'), move)
    val, spot = v
    return spot

def value(game_state, player):
    """
    Calculate the minimax value for any state under the given agent's control
    :param game_state: GameState object - state may be terminal or non-terminal
    :param player: (string) 'user' or 'AI' - AI is max
    :return: (integer) value of that state -1, 0 or 1
    """
    if game_state.is_win('AI'):
        return 1
    elif game_state.is_win('user'):
        return -1
    elif game_state.is_tie():
        return 0
    else:
        if player == 'AI':
            return max_value(game_state)
        else:
            return min_value(game_state)

def max_value(game_state):
    """
    Calculate the minimax value for a non-terminal state under Max's
    control (AI agent)
    :param game_state: non-terminal GameState object
    :return: (integer) value of that state -1, 0 or 1
    """
    v = float("-inf")

    for m in game_state.possible_moves():
        v = max(v, value(game_state.successor(m, 'AI'), 'user'))
    return v


def min_value(game_state):
    """
    Calculate the minimax value for a non-terminal state under Min's
    control (user)
    :param game_state: non-terminal GameState object
    :return: (integer) value of that state -1, 0 or 1
    """
    v = float("inf")

    for m in game_state.possible_moves():
        v = min(v, value(game_state.successor(m, 'user'), 'AI'))
    return v



def alphabeta(game_state):
    """
    Find the best move for our AI agent by applying the minimax algorithm
    with alpha beta pruning.
    :param game_state: GameState object
    :return:  a tuple representing the row column of the best move
    """
    v = None
    for move in game_state.possible_moves():
        if not v:
            v = (ab_value(game_state.successor(move, 'AI'), 'user', float("-inf"), float("inf")), move)
        else:
            val, spot = v
            if ab_value(game_state.successor(move, 'AI'), 'user', float("-inf"), float("inf")) > val:
                v = (ab_value(game_state.successor(move, 'AI'), 'user', float("-inf"), float("inf")), move)
    val, spot = v
    return spot


def ab_value(game_state, player, alpha, beta):
    """
    Calculate the minimax value for any state under the given agent's control
    using alpha beta pruning
    :param game_state: GameState object - state may be terminal or non-terminal
    :param player: (string) 'user' or 'AI' - AI is max
    :return: (integer) value of that state -1, 0 or 1
    """
    if game_state.is_win('AI'):
        return 1
    elif game_state.is_win('user'):
        return -1
    elif game_state.is_tie():
        return 0
    else:
        if player == 'AI':
            return abmax_value(game_state, alpha, beta)
        else:
            return abmin_value(game_state, alpha, beta)


def abmax_value(game_state, alpha, beta):
    """
    Calculate the minimax value for a non-terminal state under Max's
    control (AI agent) using alpha beta pruning
    :param game_state: non-terminal GameState object
    :return: (integer) value of that state -1, 0 or 1
    """
    v = float("-inf")

    for m in game_state.possible_moves():
        v = max(v, ab_value(game_state.successor(m, 'AI'), 'user', alpha, beta))
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v


def abmin_value(game_state, alpha, beta):
    """
    Calculate the minimax value for a non-terminal state under Min's
    control (user) using alpha beta pruning
    :param game_state: non-terminal GameState object
    :return: (integer) value of that state -1, 0 or 1
    """
    v = float("inf")

    for m in game_state.possible_moves():
        v = min(v, ab_value(game_state.successor(m, 'user'), 'AI', alpha, beta))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v


def abdl(game_state, depth):
    """
    Find the best move for our AI agent by limiting the alpha beta search to
    the given depth and using the evaluation function game_state.eval()
    :param game_state: GameState object
    :return:  a tuple representing the row column of the best move
    """
    v = None
    for move in game_state.possible_moves():
        if not v:
            v = (abdl_value(game_state.successor(move, 'AI'), 'user', float("-inf"), float("inf"), depth), move)
        else:
            val, spot = v
            if abdl_value(game_state.successor(move, 'AI'), 'user', float("-inf"), float("inf"), depth) > val:
                v = (abdl_value(game_state.successor(move, 'AI'), 'user', float("-inf"), float("inf"), depth), move)
    val, spot = v
    return spot



def abdl_value(game_state, player, alpha, beta, depth):
    """
    Calculate the utility for any state under the given agent's control
    using depth limited alpha beta pruning and the evaluation
    function game_state.eval()
    :param game_state: GameState object - state may be terminal or non-terminal
    :param player: (string) 'user' or 'AI' - AI is max
    :return: (integer) utility of that state
    """
    if game_state.is_win('AI'):
        return 1
    elif game_state.is_win('user'):
        return -1
    elif game_state.is_tie():
        return 0
    else:
        if player == 'AI':
            return abdlmax_value(game_state, alpha, beta, depth)
        else:
            return abdlmin_value(game_state, alpha, beta, depth)


def abdlmax_value(game_state, alpha, beta, depth):
    """
    Calculate the utility for a non-terminal state under Max's control
    using depth limited alpha beta pruning and the evaluation
    function game_state.eval()
    :param game_state: non-terminal GameState object
    :return: (integer) utility (evaluation function) of that state
    """
    if depth <= 0:
        return game_state.eval()

    v = float("-inf")
    for m in game_state.possible_moves():
        v = max(v, abdl_value(game_state.successor(m, 'AI'), 'user', alpha, beta, depth - 1))
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v


def abdlmin_value( game_state, alpha, beta, depth):
    """
    Calculate the utility for a non-terminal state under Min's control
    using depth limited alpha beta pruning and the evaluation
    function game_state.eval()
    :param game_state: non-terminal GameState object
    :return: (integer) utility (evaluation function) of that state
    """
    if depth <= 0:
        return game_state.eval()

    v = float("inf")
    for m in game_state.possible_moves():
        v = min(v, abdl_value(game_state.successor(m, 'user'), 'AI', alpha, beta, depth - 1))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v

