import tensorflow as tf
import os
from ..helpers import board as b, model as model_helper

LEARNING_RATE = 0.001
BATCH_SIZE = 100
DISPLAY_STEP = 50
EPOCHS = 551

MODEL_PATH = "v2\\model"
MODEL_NAME = "tictactoe-2h"
DATA_PATH = "v2\\data"
X_TRAIN_BOARDS = "XBoardsWithResults.txt"

# Network Parameters
n_input = 18 # data input (X board, O board: 9+9)
n_classes = 9 # total positions (0-8)

n_hidden_1 = 36 # 1st layer number of features
n_hidden_2 = 18 # 2nd layer number of features

def init_model():
    init_default_graph()
    init_graph_inputs_and_outputs()
    init_graph_weights_and_biases()
    create_model()
    init_graph_cost_and_optimizer()
    init_saver()
    init_variables()

def init_default_graph():
    global graph_v2
    graph_v2 = tf.get_default_graph()

def init_graph_inputs_and_outputs():
    global X_v2, Y_v2, Y_pred_v2
    X_v2 = tf.placeholder(tf.float32, [None, n_input])
    Y_v2 = tf.placeholder(tf.float32, [None, n_classes])
    Y_pred_v2 = tf.placeholder(tf.int32, [None])

def init_graph_weights_and_biases():
    global weights_v2, biases_v2
    weights_v2 = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases_v2 = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }  

def create_model():
    global pred_v2
    pred_v2 = model_helper.multilayer_perceptron(X_v2, weights_v2, biases_v2)

def init_graph_cost_and_optimizer():
    global cost_v2, optimizer_v2
    cost_v2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_v2, labels=Y_v2))
    optimizer_v2 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost_v2)

def init_saver():
    global saver_v2
    # 'Saver' op to save and restore all the variables
    saver_v2 = tf.train.Saver()

def init_variables():
    global tf_init_v2
    tf_init_v2 = tf.global_variables_initializer()

# dictionary with all variables
def get_dictionary():
    return {
        'model_path': MODEL_PATH,
        'model_name': MODEL_NAME,
        'data_path': DATA_PATH,
        'data_name': X_TRAIN_BOARDS,
        'graph': graph_v2,
        'X': X_v2,
        'Y': Y_v2,
        'Y_pred': Y_pred_v2,
        'weights': weights_v2,
        'biases': biases_v2,
        'pred': pred_v2,
        'cost': cost_v2,
        'optimizer': optimizer_v2,
        'saver': saver_v2,
        'init': tf_init_v2
        }
