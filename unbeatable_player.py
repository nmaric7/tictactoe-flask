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

def get_next_move(board, mark):
    # check for winning move
    move = winning_move(board, mark)
    if move != -1:
        return move     
    # block opponent win
    move = block_opponent_win(board, mark)
    if move != -1:
        return move     
    # try to fork
    move = fork_move(board, mark)
    if move != -1:
        return move  
    # block opponent fork
    move = block_opponent_fork(board, mark)
    if move != -1:
        return move  
    # take the center
    if board[4] == EMPTY_SQUARE:
        return 4    
    # take the opposite corner from opponent
    move = opposite_corner(board, mark)
    if move != -1:
        return move 
    # available corner
    move = available_corner(board)    
    if move != -1:
        return move 
    # available side
    move = available_side(board)    
    return move

def winning_move(board, mark):
    for idx, s in enumerate(board):
        if s == EMPTY_SQUARE:
            c_board = copy_board(board)
            c_board[idx] = mark
            if is_win(c_board, mark):
                return idx 
    return -1

def block_opponent_win(board, mark):
    opponent_mark = get_opponent_mark(mark)
    for idx, s in enumerate(board):
        if s == EMPTY_SQUARE:
            c_board = copy_board(board)
            c_board[idx] = opponent_mark
            if is_win(c_board, opponent_mark):
                return idx 
    return -1

def fork_move(board, mark):
    for idx, s in enumerate(board):
        if s == EMPTY_SQUARE:
            c_board = copy_board(board)
            c_board[idx] = mark
            if is_fork(c_board, mark):
                return idx
    return -1

def block_opponent_fork(board, mark):
    opponent_mark = get_opponent_mark(mark)
    for idx, s in enumerate(board):
        if s == EMPTY_SQUARE:
            c_board = copy_board(board)
            c_board[idx] = opponent_mark
            if is_fork(c_board, opponent_mark):
                return idx
    return -1

def opposite_corner(board, mark):
    opponent_mark = get_opponent_mark(mark)
    if board[0] == EMPTY_SQUARE and board[8] == opponent_mark: 
        return 0
    elif board[2] == EMPTY_SQUARE and board[6] == opponent_mark: 
        return 2
    elif board[2] == opponent_mark and board[6] == EMPTY_SQUARE: 
        return 6
    elif board[0] == opponent_mark and board[8] == EMPTY_SQUARE: 
        return 8

    return -1

def available_corner(board):
    if board[0] == EMPTY_SQUARE:
        return 0
    elif board[2] == EMPTY_SQUARE:
        return 2
    elif board[6] == EMPTY_SQUARE:
        return 6
    elif board[8] == EMPTY_SQUARE:
        return 8
    return -1    

def available_side(board):
    if board[1] == EMPTY_SQUARE:
        return 1
    elif board[3] == EMPTY_SQUARE:
        return 3
    elif board[5] == EMPTY_SQUARE:
        return 5
    elif board[7] == EMPTY_SQUARE:
        return 7
    return -1   

# helpers

def copy_board(board):
    # initialize copy of board
    c_board = []
    # copy values from origin board
    for i in board:
        c_board.append(i)
    # return copy of board    
    return c_board

def get_opponent_mark(mark):
    if mark == MARK_X:
        return MARK_O

    return MARK_X

def is_win(board, mark):
    # check each winning combination
    for w_comb in WINNING_COMBINATIONS:
        # number of occupied squares by mark is equal to 3, that is winning combination
        if get_occupied_squares_number(w_comb, board, mark) == 3:
            return True
    # there is no win
    return False

def get_occupied_squares_number(comb, board,mark):
    occupied_squares_by_mark = 0
    # check each square in combination
    for p in comb:
        # if square is occupied by mark, increase counter
        if board[p] == mark:
            occupied_squares_by_mark = occupied_squares_by_mark + 1

    return occupied_squares_by_mark

def is_fork(board, mark):
    fork_combinations = 0
    for w_comb in WINNING_COMBINATIONS:
        if get_occupied_squares_number(w_comb, board, mark) == 2:
            fork_combinations = fork_combinations + 1
    return fork_combinations > 1        
