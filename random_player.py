import random

# MARK_X = 'X'
# MARK_O = 'O'
EMPTY_MARK = '-'

def get_next_move(board):
    free_squares = [idx for idx, square in enumerate(board) if square == EMPTY_MARK]
    return random.choice(free_squares)
