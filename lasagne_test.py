from __future__ import print_function
import numpy as np
import time

print("Generating dataset...")
X = np.random.uniform(size = (100000, 100)).astype(np.float32)
y = X

# =====================================================

print("Preparing...")
ts = time.clock()

import numpy as np
import theano
import theano.tensor as T

import lasagne

x = T.matrix('x')
targets = T.matrix("targets")

l_in = lasagne.layers.InputLayer(shape=(None, 100), input_var = x)
l_hid1 = lasagne.layers.DenseLayer(
        l_in, num_units=400,
        nonlinearity=lasagne.nonlinearities.tanh,
        W=lasagne.init.GlorotUniform())

l_hid2 = lasagne.layers.DenseLayer(
        l_hid1, num_units=200,
        nonlinearity=lasagne.nonlinearities.tanh,
        W=lasagne.init.GlorotUniform())

l_hid3 = lasagne.layers.DenseLayer(
        l_hid2, num_units=200,
        nonlinearity=lasagne.nonlinearities.tanh,
        W=lasagne.init.GlorotUniform())

l_hid4 = lasagne.layers.DenseLayer(
        l_hid3, num_units=200,
        nonlinearity=lasagne.nonlinearities.tanh,
        W=lasagne.init.GlorotUniform())

model = lasagne.layers.DenseLayer(
        l_hid4, num_units=100,
        nonlinearity=lasagne.nonlinearities.tanh,
        W=lasagne.init.GlorotUniform())

prediction = lasagne.layers.get_output(model)
loss = lasagne.objectives.squared_error(prediction, targets)
loss = loss.mean()

params = lasagne.layers.get_all_params(model, trainable=True)

updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

train_fn = theano.function([x, targets], loss, updates=updates)

print("Time elapsed: ", time.clock() - ts)

# ===================================================

print("Calculating...")
ts = time.clock()

bsz = 100000
bs = 100000 / bsz

ts = time.clock()

for _ in range(10):
    for b in range(bs):
        train_fn(X[b * bsz : (b + 1) * bsz], y[b * bsz : (b + 1) * bsz])

print("Time elapsed: ", time.clock() - ts)