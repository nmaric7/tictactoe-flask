
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
from skimage import data, transform, color
import numpy as np
import matplotlib.pyplot as plt
import random


# In[4]:


def loadData(filename):
    with open(filename) as f:
        content = f.readlines()
    return content


# In[5]:


ROOT_PATH = "c:\\Users\\Nikica\\Desktop\\Tensorflow-Template\\data\\tictactoe"
train_boards = loadData(os.path.join(ROOT_PATH, "XRandomVsRandomWithResults.txt"))


# In[6]:


def maskBoardByMark(board, mark):
    mask = []
    board = board.rstrip('\n')
    for letter in board:
        if letter == mark:
            mask.append(1)
        else:
            mask.append(0)
    return mask

def maskOutput(output):
    # print(output)
    output = output.rstrip('\n')
    outputValue = int(output)
    outputArray = []
    for i in range(9):
        outputArray.append(1 if i == outputValue else 0)
    
    return outputValue, outputArray


# In[7]:


def transformBoardsToNNData(boards):
    nnInputs = []
    labels = []
    valueLabels = []
    for board in boards:
        tokens = board.split(',')
        # nnInput = [] 
        # nnInput.append(maskBoardByMark(tokens[0], "X"))
        # nnInput.append(maskBoardByMark(tokens[0], "O"))
        nnInputs.append(maskBoardByMark(tokens[0], "X") + maskBoardByMark(tokens[0], "O"))
        outputValue, outputArray = maskOutput(tokens[1])
        labels.append(outputArray)
        valueLabels.append(outputValue)
    return nnInputs, labels, valueLabels


# In[8]:


inputs, labels, valueLabels = transformBoardsToNNData(train_boards)
# print("in", len(x))
# print("y", len(y))


# In[9]:


## plotLabels = np.array(valueLabels)

# Make a histogram with 9 bins of the `labels` data
## plt.hist(plotLabels, 9)

# Show the plot
## plt.show()


# In[10]:


# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 50
model_path = "C:\\Users\\Nikica\\Desktop\\Tensorflow-Template\\Model\\tictactoe-2h"


# In[11]:


# Network Parameters
n_hidden_1 = 36 # 1st layer number of features
n_hidden_2 = 18 # 2nd layer number of features
n_input = 18 # data input (X board, O board: 9+9)
n_classes = 9 # total positions (0-8)


# In[12]:


# tf Graph input
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])
Y_pred = tf.placeholder(tf.int32, [None])


# In[13]:


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


# In[14]:


# Store layers weight & bias
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


# In[15]:


# Construct model
pred = multilayer_perceptron(X, weights, biases)


# In[16]:


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# # convert logits to label indexes
# correct_pred = tf.argmax(pred, 1)

# # define accouracy metric
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[17]:


# Initialize the variables (i.e. assign their default value)
## init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
## saver = tf.train.Saver()


# In[18]:


def train(sess, epochs, inputs, labels):
    # Training cycle
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(len(inputs)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: inputs[(i * batch_size):((i + 1) * batch_size)],
                                                          Y: labels[(i * batch_size):((i + 1) * batch_size)]})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=",                 "{:.9f}".format(avg_cost))


# In[19]:


def test(inputs, labels):
    # convert logits to label indexes
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # define accouracy metric
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # print
    print("Accuracy:", accuracy.eval({X: inputs, Y: labels}))


# In[32]:


def saveModel(sess):
    # Save model weights to disk
    save_path = saver.save(sess, os.path.join(model_path, "tictactoe-2h"))
    print("Model saved in file: %s" % save_path)


# In[33]:


def initAndCreateModel():
    print("Starting 1st session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        # Train
        train(sess, 151, inputs, labels)
        # Test, check accuracy
        test(inputs, labels)
        # Save model
        saveModel(sess)


# In[34]:


## initAndCreateModel()


# In[39]:


def loadModel(sess):
    # 'Saver' op to save and restore all the variables
    print('saver1 initialize')
    saver1 = tf.train.Saver()
    print('saver1 restore')
    # Restore model weights from previously saved model
    saver1.restore(sess, os.path.join(model_path, "tictactoe-2h"))
    # print("Model restored from file: %s" % save_path)


# In[40]:


def loadAndTrainModel():
    # Running a new session
    print("Starting 2nd session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        # Load model
        loadModel(sess)
        # Train
        train(sess, 251, inputs, labels)
        # Test
        test(inputs, labels)
        # Save model
        saveModel(sess)


# In[41]:


## loadAndTrainModel()


# In[23]:


def predict(board):
    print('predict(board)', board)
    with tf.Session() as sess:
        print('session init')
        sess.run(tf.global_variables_initializer())
        print('load model')
        loadModel(sess)
        print('prepare sample boards')
        sample_boards = []
        sample_boards.append(maskBoardByMark(board, 'X') + maskBoardByMark(board, 'O'))
        print('predict')
        # Run the "correct_pred" operation
        predicted = sess.run([tf.argmax(pred, 1)], feed_dict={X: sample_boards})[0]
        print('board', board, 'predicted', predicted)
        return predicted


# In[ ]:


# Pick 10 random boards
sample_indexes = random.sample(range(len(inputs)), 500)
sample_boards = [inputs[i] for i in sample_indexes]
sample_labels = [valueLabels[i] for i in sample_indexes]

# Run the "correct_pred" operation
## predicted = sess.run([tf.argmax(pred, 1)], feed_dict={X: sample_boards})[0]

""" count = 0
# Display the predictions and the ground truth visually.
for i in range(len(sample_boards)):
    truth = sample_labels[i]
    prediction = predicted[i]
    if (truth == prediction):
        count += 1
    # color='\033[92m' if truth == prediction else '\033[91m'
    # print("{0}Truth: {1} Prediction: {2}\033[00m".format(color, truth, prediction))
print(str(count) + '/' + str(len(sample_boards)))
 """

# In[42]:


## predict(' XOOX O  ')


# In[196]:


def accuracy():
    with tf.Session() as sess:
        sess.run(init)
        loadModel(sess)
        test(inputs, labels)


# In[197]:


## accuracy()

