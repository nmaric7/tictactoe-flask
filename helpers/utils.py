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