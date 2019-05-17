import os 

def change_working_directory(path):
    # fetch current directory
    dir_path = os.path.dirname(__file__)
    # go one level up, for root directory
    dir_path = os.path.dirname(dir_path)
    os.chdir(os.path.join(dir_path, path))

def load_data(file_name):
    with open(file_name) as f:
        content = f.readlines()
    return content

def append_training_data(data_path, data_name, board, move):
    change_working_directory(data_path)
    
    # with open(data_name) as f:
    #     content = f.readlines()
    #     for line in content:
    #         print(line, txt, line == txt)

    with open(data_name, "a+") as f:
        print('open file', data_name)
        txt = board + "," + str(move)
    #     content = f.readlines()
    #     for line in content:
    #         print(line, txt, line == txt)
        f.write(txt + "\n")
