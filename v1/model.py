import flask
import numpy as np
import tensorflow as tf
from ..helpers import model as model_helper

import os



MODEL_PATH = "v1\\model"
MODEL_NAME = "tictactoe-2h"
DATA_PATH = "v1\\data"
X_TRAIN_BOARDS = "XBoardsWithResults.txt"


LEARNING_RATE = 0.001

# Network Parameters
n_hidden_1 = 36 # 1st layer number of features
n_hidden_2 = 18 # 2nd layer number of features
n_input = 18 # data input (X board, O board: 9+9)
n_classes = 9 # total positions (0-8)

def init_model():
    init_default_graph()
    init_graph_inputs_and_outputs()
    init_graph_weights_and_biases()
    create_model()
    init_graph_cost_and_optimizer()
    init_saver()
    # Initialize the variables at the end, because of saver
    init_variables()

def init_default_graph():
    global graph_v1
    graph_v1 = tf.get_default_graph()

def init_graph_inputs_and_outputs():
    global X_v1, Y_v1, Y_pred_v1
    X_v1 = tf.placeholder(tf.float32, [None, n_input])
    Y_v1 = tf.placeholder(tf.float32, [None, n_classes])
    Y_pred_v1 = tf.placeholder(tf.int32, [None])

def init_graph_weights_and_biases():
    global weights_v1, biases_v1
    weights_v1 = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases_v1 = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

def create_model():
    global pred_v1
    pred_v1 = model_helper.multilayer_perceptron(X_v1, weights_v1, biases_v1)

def init_graph_cost_and_optimizer():
    global cost_v1, optimizer_v1
    cost_v1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_v1, labels=Y_v1))
    optimizer_v1 = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost_v1)

def init_saver():
    global saver_v1
    # 'Saver' op to save and restore all the variables
    saver_v1 = tf.train.Saver()

def init_variables():
    global tf_init_v1
    tf_init_v1 = tf.global_variables_initializer()

# dictionary with all variables
def get_dictionary():
    return {
        'model_path': MODEL_PATH,
        'model_name': MODEL_NAME,
        'data_path': DATA_PATH,
        'data_name': X_TRAIN_BOARDS,
        'graph': graph_v1,
        'X': X_v1,
        'Y': Y_v1,
        'Y_pred': Y_pred_v1,
        'weights': weights_v1,
        'biases': biases_v1,
        'pred': pred_v1,
        'cost': cost_v1,
        'optimizer': optimizer_v1,
        'saver': saver_v1,
        'init': tf_init_v1
        }

