from . import utils as u
import os

def load_inputs_and_labels(data_path, data_name):
    nnInputs = []
    labels = []

    boards_with_results = load_boards_with_results_from_file(data_path, data_name)

    for board in boards_with_results:
        tokens = board.split(',')
        nnInputs.append(mask_board_by_mark(tokens[0], "X") + mask_board_by_mark(tokens[0], "O"))
        labels.append(mask_output(tokens[1]))
    return nnInputs, labels

def load_boards_with_results_from_file(data_path, data_name):
    u.change_working_directory(data_path)
    return u.load_data(data_name)

def mask_board_by_mark(board, mark):
    mask = []
    board = board.rstrip('\n')
    for letter in board:
        mask.append(1 if letter == mark else 0)
    return mask

def mask_output(output):
    output = output.rstrip('\n')
    outputValue = int(output)
    outputArray = []
    for i in range(9):
        outputArray.append(1 if i == outputValue else 0)    
    return outputArray    

def append_training_data(data_path, data_name, new_boards):
    print('append_training_data', data_path, data_name, new_boards)
    u.change_working_directory(data_path)

    with open(data_name, "a+") as myfile:
        for b in new_boards:
            myfile.write(b + "\n")

def prepare_sample_board(board):
    sample_boards = []
    sample_boards.append(mask_board_by_mark(board, 'X') + mask_board_by_mark(board, 'O'))
    return sample_boards