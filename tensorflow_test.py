from __future__ import print_function
import numpy as np
import time

print("Generating dataset...")

tests = 100000

data = np.random.uniform(size = (tests, 100))
targets = data

learning_rate = 0.001
training_epochs = 100
bsz = 100000
display_step = 1

# =====================================================
import tensorflow as tf

n_input = 100
n_hidden_1 = 400
n_hidden_2 = 200
n_hidden_3 = 200
n_hidden_4 = 200
n_hidden_5 = 100

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_hidden_5])

# Create model
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.tanh(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.tanh(layer_4)

    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.tanh(layer_5)

    return layer_5

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
}

# ===================================================

print("Preparing model...")
ts = time.clock()

pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

print("Time elapsed: ", time.clock() - ts)

# ===================================================

print("Calculating...")
ts = time.clock()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(tests/bsz)
        # Loop over all batches
        for b in range(total_batch):
            batch_x, batch_y = data[b * bsz : (b + 1) * bsz], targets[b * bsz : (b + 1) * bsz]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})

print("Time elapsed: ", time.clock() - ts)