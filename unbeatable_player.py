import random 

MARK_X = 'X'
MARK_O = 'O'
EMPTY_SQUARE = '-'

WINNING_COMBINATIONS = [
    # horizontal
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    # vertical
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    # diagonal
    [0, 4, 8],
    [2, 4, 6]
]

CENTER = 4
CORNERS = [0, 2, 6, 8]
SIDES = [1, 3, 5, 7]
OPPOSITE_CORNERS = [[0, 8], [2, 6], [6, 2], [8, 0]]

def get_next_move(board, mark):
    # check for winning move
    moves = winning_moves(board, mark)

    # block opponent win
    if len(moves) == 0:
        moves = winning_moves(board, get_opponent_mark(mark))

    # try to fork
    if len(moves) == 0:
        moves = fork_moves(board, mark)

    # block opponent fork
    if len(moves) == 0:
        moves = fork_moves(board, get_opponent_mark(mark))
        # if there is more than one fork place 
        if len(moves) == 2:
            # O - -
            # - X -
            # - - O
            # take one of sides
            moves = [x for x in SIDES if board[x] == EMPTY_SQUARE]
        elif len(moves) == 4:
            # X - -
            # - O -
            # - - O
            # take one of free corners
            moves = [x for x in CORNERS if board[x] == EMPTY_SQUARE]

    # take the center
    if len(moves) == 0 and board[CENTER] == EMPTY_SQUARE:
        moves = [CENTER]

    # take the opposite corner from opponent
    if len(moves) == 0:
        moves = opposite_corner_moves(board, get_opponent_mark(mark))

     # available corner
    if len(moves) == 0:
        moves = available_corner_moves(board)      

    # available side
    if len(moves) == 0:
        moves = available_side_moves(board)

    # return random empty space if all checks returns empty list. 
    # It should not be happenned ever
    return random.choice(moves) if len(moves) > 0 else random.choice(get_empty_squares(board))

def winning_moves(board, mark):
    return [idx for idx in get_empty_squares(board) if has_win(board, mark, idx)]

def fork_moves(board, mark):
    return [idx for idx in get_empty_squares(board) if has_fork(board, mark, idx)]

def opposite_corner_moves(board, mark):
    return [oc[0] for oc in OPPOSITE_CORNERS if oc[0] == EMPTY_SQUARE and oc[1] == mark]

def available_corner_moves(board):
    return [i for i in CORNERS if board[i] == EMPTY_SQUARE]

def available_side_moves(board):
    return [i for i in SIDES if board[i] == EMPTY_SQUARE]

# helpers
def prepare_board(board, mark, id):
    c_board = [i for i in board]
    c_board[id] = mark
    return c_board

def get_opponent_mark(mark):
    return MARK_X if mark == MARK_O else MARK_O     

def get_occupied_squares_number(comb, board, mark):
    return len([p for p in comb if board[p] == mark])

def get_occupied_squares_number(comb, board, mark):
    return len([p for p in comb if board[p] == mark])

def get_potential_winning_combinations(board, mark):
    return [w_comb for w_comb in WINNING_COMBINATIONS if not is_combination_contains_mark(w_comb, board, get_opponent_mark(mark))]

def is_combination_contains_mark(comb, board, mark):
    return len([idx for idx in comb if board[idx] == mark]) > 0

def get_empty_squares(board):
    return [idx for idx, s in enumerate(board) if s == EMPTY_SQUARE]

def has_win(board, mark, idx):
    return has_combination(board, mark, idx, 3, 0)

def has_fork(board, mark, idx):
    return has_combination(board, mark, idx, 2, 1)

def has_combination(board, mark, idx, number_of_squares, number_of_combinations):
    c_board = prepare_board(board, mark, idx)
    w_combinations = [w_comb for w_comb in get_potential_winning_combinations(board, mark) if get_occupied_squares_number(w_comb, c_board, mark) == number_of_squares]
    return len(w_combinations) > number_of_combinations

 