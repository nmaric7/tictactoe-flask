import random

# MARK_X = 'X'
# MARK_O = 'O'
EMPTY_MARK = '-'

def get_next_move(board):
    free_squares = []
    for idx, square in enumerate(board):
        if EMPTY_MARK == square:
            free_squares.append(idx)
    return random.choice(free_squares)
