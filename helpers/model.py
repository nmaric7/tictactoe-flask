import tensorflow as tf
import numpy as np
import os

from . import utils as u
from . import board as b

BATCH_SIZE = 100
DISPLAY_STEP = 50
EPOCHS = 501

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # out_layer = tf.contrib.layers.fully_connected(out_layer, 9, tf.nn.relu)
    return out_layer

def checkpoint_exists(d):
    # change working directory
    u.change_working_directory(d['model_path'])
    # check for checkpoint
    return tf.train.checkpoint_exists(d['model_name'])

def create_and_save_default_model(d):
    with tf.Session(graph=d['graph']) as sess:
        sess.run(d['init'])
        train_model(sess, d)
        save_model(sess, d)    

def train_model(sess, d):
    # print("train_model")
    # load training inputs and labels
    inputs, labels = b.load_inputs_and_labels(d['data_path'], d['data_name'])
    # Training cycle
    for epoch in range(EPOCHS):
        avg_cost = 0.
        total_batch = int(len(inputs)/BATCH_SIZE)
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([d['optimizer'], d['cost']], 
            feed_dict={d['X']: inputs[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)], 
            d['Y']: labels[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)]})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % DISPLAY_STEP == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

# save model
def save_model(sess, d):
    # prepare working directory. one level up, because this is helpers directory
    dir_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(dir_path, d['model_path'])
    # Save model weights to disk
    save_path = d['saver'].save(sess, os.path.join(model_path, d['model_name']))
    print("Model saved in file: %s" % save_path)

# load model
def load_model(sess, d):
    # prepare working directory
    dir_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(dir_path, d['model_path'])
    # Restore model weights from previously saved model
    d['saver'].restore(sess, os.path.join(model_path, d['model_name']))

def accuracy(d):
     # create tenserflow session
    with tf.Session(graph=d['graph']) as sess:
        # register global variables
        sess.run(d['init'])

        if checkpoint_exists(d) == True:
            load_model(sess, d)
            inputs, labels = b.load_inputs_and_labels(d['data_path'], d['data_name'])
            correct_pred = tf.equal(tf.argmax(d['pred'], 1), tf.argmax(d['Y'], 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            acc = accuracy.eval({d['X']: inputs, d['Y']: labels})
            return float(acc)

        return float(-1)

def train_NN(d):
    with tf.Session(graph=d['graph']) as sess:
        sess.run(d['init'])
        load_model(sess, d)
        train_model(sess, d)
        save_model(sess, d)


def predict(board, d):
    with tf.Session(graph=d['graph']) as sess:
        # print('init')
        sess.run(d['init'])
        # print('load_model')
        load_model(sess, d)
        # print('sample_boards')
        sample_boards = b.prepare_sample_board(board)
        # print('predicted')
        predicted = sess.run(d['pred'], feed_dict={d['X']: sample_boards})[0]
        # print('find first available')
        move = find_first_available(board, np.argsort(-predicted))
        return int(move)

def find_first_available(board, indexes):
    board = board.rstrip('\n')
    for i in indexes:
        if board[i] == '-' or board[i] == ' ':
            return i
    return 0