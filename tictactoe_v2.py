import tensorflow as tf
import os
from . import model_v2 as m
from . import board as b

DATA_PATH = "data\\tictactoe_v2"
MODEL_PATH = "model\\tictactoe_v2"
MODEL_NAME = "tictactoe_v2"
X_TRAIN_BOARDS = "XBoardsWithResults.txt"
O_TRAIN_BOARDS = "OBoardsWithResults.txt"

def accuracy():
    print("tictactoe_v2.accuracy")
    # create tenserflow session
    with tf.Session(graph=m.graph) as sess:
        # register global variables
        sess.run(m.init)
        # check for model
        check_model_exists(sess, MODEL_PATH, MODEL_NAME, DATA_PATH, X_TRAIN_BOARDS)
        # load model
        m.load_model(sess, MODEL_PATH, MODEL_NAME)
        # load training inputs and labels
        inputs, labels = b.load_inputs_and_labels(DATA_PATH, X_TRAIN_BOARDS)
        # calculate accuracy
        acc = calculate_accuracy(inputs, labels)

        return float(acc)


# helpers
def check_model_exists(sess, model_path, model_name, data_path, data_name):
    # check is model exists
    model_exists = m.check_model_exists(sess, model_path, model_name)
    # if model does not exists, train new model and save it
    if model_exists == False:
        # train model  
        m.train_model(sess, data_path, data_name)
        # save model
        m.save_model(sess, model_path, model_name)

def calculate_accuracy(inputs, labels):
    # convert logits to label indexes
    correct_pred = tf.equal(tf.argmax(m.pred, 1), tf.argmax(m.Y, 1))
    # define accouracy metric
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print("calculate_accuracy")
    # print
    acc = accuracy.eval({m.X: inputs, m.Y: labels})

    return float(acc)
