import flask
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os

ROOT_PATH = "c:\\Users\\Nikica\\Desktop\\Tensorflow-Template\\data\\tictactoe"
MODEL_PATH = "C:\\Users\\Nikica\\Desktop\\Tensorflow-Template\\Model\\tictactoe-2h"
MODEL_NAME = "tictactoe-2h"
TRAIN_BOARDS_PATH = "XRandomVsRandomWithResults.txt"

LEARNING_RATE = 0.001
BATCH_SIZE = 100
DISPLAY_STEP = 50

# Network Parameters
n_hidden_1 = 36 # 1st layer number of features
n_hidden_2 = 18 # 2nd layer number of features
n_input = 18 # data input (X board, O board: 9+9)
n_classes = 9 # total positions (0-8)

def init_model():
    # Initialize default graph
    init_default_graph()
    # Initialize the variables (i.e. assign their default value)
    init_variables()
    global weights, biases, pred, cost, optimizer, train_boards
    # Initialize inputs, outputs and predicted outputs
    init_graph_inputs_and_outputs()
    # Initialize graphs weights and biases
    init_graph_weights_and_biases()
    # Construct the Model
    construct_model()
    # Initialize cost and optimizer
    init_graph_cost_and_optimizer()
    # Load trainig boards
    load_training_boards()

def init_default_graph():
    global graph
    graph = tf.get_default_graph()

def init_variables():
    global tf_init
    tf_init = tf.global_variables_initializer()

def init_graph_inputs_and_outputs():
    global X, Y, Y_pred
    X = tf.placeholder(tf.float32, [None, n_input])
    Y = tf.placeholder(tf.float32, [None, n_classes])
    Y_pred = tf.placeholder(tf.int32, [None])

def init_graph_weights_and_biases():
    global weights, biases
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

def construct_model():
    global pred
    pred = multilayer_perceptron(X, weights, biases)

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

def init_graph_cost_and_optimizer():
    global cost, optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

def load_training_boards():
    global train_boards
    train_boards = load_data(os.path.join(ROOT_PATH, TRAIN_BOARDS_PATH))

def load_data(filename):
    with open(filename) as f:
        content = f.readlines()
    return content

def transform_boards_to_NN_data(boards):
    nnInputs = []
    labels = []
    valueLabels = []
    for board in boards:
        tokens = board.split(',')
        nnInputs.append(mask_board_by_mark(tokens[0], "X") + mask_board_by_mark(tokens[0], "O"))
        outputValue, outputArray = mask_output(tokens[1])
        labels.append(outputArray)
        valueLabels.append(outputValue)
    return nnInputs, labels, valueLabels

def load_model(sess):
    global saver
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()
    # Restore model weights from previously saved model
    saver.restore(sess, os.path.join(MODEL_PATH, MODEL_NAME))

def predict(board):
    with tf.Session(graph=graph) as sess:
        # session init
        init_session(sess)
        # load model
        load_model(sess)
        print('prepare sample boards')
        sample_boards = []
        sample_boards.append(mask_board_by_mark(board, 'X') + mask_board_by_mark(board, 'O'))
        # print("sample_boards", sample_boards)
        # Run the "correct_pred" operation
        predicted = sess.run(pred, feed_dict={X: sample_boards})[0]
        move = find_first_available(board, np.argsort(-predicted))
        
        # print('board', board, 'type', type(predicted), 'predicted', predicted, 'move', move)
        return int(move)

def find_first_available(board, indexes):
    print('indexes', indexes)
    board = board.rstrip('\n')
    for i in indexes:
        if board[i] == ' ':
            return i
    return 0

def mask_board_by_mark(board, mark):
    mask = []
    board = board.rstrip('\n')
    for letter in board:
        if letter == mark:
            mask.append(1)
        else:
            mask.append(0)
    return mask

def mask_output(output):
    # print(output)
    output = output.rstrip('\n')
    outputValue = int(output)
    outputArray = []
    for i in range(9):
        outputArray.append(1 if i == outputValue else 0)
    
    return outputValue, outputArray    

def accuracy():
   with tf.Session(graph=graph) as sess:
        sess.run(tf_init)

        load_model(sess)

        inputs, labels, valueLabels = transform_boards_to_NN_data(train_boards)

        # convert logits to label indexes
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # define accouracy metric
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # print
        acc = accuracy.eval({X: inputs, Y: labels})

        return float(acc)

def train():
    with tf.Session(graph=graph) as sess:
        init_session(sess)
        # Load model
        load_model(sess)
        # Train
        train_model(sess, 551)
        # Save model
        save_model(sess)

def init_session(sess):
    # Initialize variables
    sess.run(tf_init)

def train_model(sess, epochs):

    inputs, labels, valueLabels = transform_boards_to_NN_data(train_boards)

    # Training cycle
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(len(inputs)/BATCH_SIZE)
        # Loop over all batches
        for i in range(total_batch):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: inputs[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)],
                                                          Y: labels[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)]})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % DISPLAY_STEP == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=",                 "{:.9f}".format(avg_cost))

def save_model(sess):
    # Save model weights to disk
    save_path = saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME))
    print("Model saved in file: %s" % save_path)