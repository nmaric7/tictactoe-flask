import tensorflow as tf
import os
from . import board as b
from . import utils as u

LEARNING_RATE = 0.001
BATCH_SIZE = 100
DISPLAY_STEP = 50
EPOCHS = 251

DATA_PATH = "data\\tictactoe_v2"

# Network Parameters
n_input = 18 # data input (X board, O board: 9+9)
n_classes = 9 # total positions (0-8)

n_hidden_1 = 36 # 1st layer number of features
n_hidden_2 = 18 # 2nd layer number of features

def init_model():
    print("model_v2.init_model")
    # Initialize default graph
    init_default_graph()
    # Initialize inputs, outputs and predicted outputs
    init_graph_inputs_and_outputs()
    # Initialize graphs weights and biases
    init_graph_weights_and_biases()
    # Construct the Model
    create_model()
    # Initialize cost and optimizer
    init_graph_cost_and_optimizer()
    # Initialize saver
    init_saver()
    # Initialize the variables (i.e. assign their default value)
    init_variables()

def init_default_graph():
    print("model_v2.init_default_graph")
    global graph
    graph = tf.get_default_graph()

def init_variables():
    global init
    init = tf.global_variables_initializer()
    
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

def create_model():
    global pred
    pred = multilayer_perceptron()

# Create Multi-Layer perceptron model
def multilayer_perceptron():
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
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
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

def init_saver():
    global saver
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()

# save model
def save_model(sess, model_path, model_name):
    # prepare working directory
    dir_path = os.path.dirname(__file__)
    model_path = os.path.join(dir_path, model_path)
    # Save model weights to disk
    save_path = saver.save(sess, os.path.join(model_path, model_name))
    print("Model saved in file: %s" % save_path)

# load model
def load_model(sess, model_path, model_name):
    # prepare working directory
    dir_path = os.path.dirname(__file__)
    model_path = os.path.join(dir_path, model_path)
    # Restore model weights from previously saved model
    saver.restore(sess, os.path.join(model_path, model_name))

def check_model_exists(sess, model_path, model_name):
    # change working directory
    u.change_working_directory(model_path)
    # check for checkpoint
    checkpoint_exists = tf.train.checkpoint_exists(model_name)

    return checkpoint_exists

def train_model(sess, data_path, data_name):
    # load training inputs and labels
    inputs, labels = b.load_inputs_and_labels(data_path, data_name)
    # Training cycle
    for epoch in range(EPOCHS):
        avg_cost = 0.
        total_batch = int(len(inputs)/BATCH_SIZE)
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: inputs[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)],
                                                          Y: labels[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)]})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % DISPLAY_STEP == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
