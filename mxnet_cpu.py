from __future__ import print_function
import numpy as np
import time

print("Generating dataset...")
X = np.random.uniform(size = (100000, 100)).astype(np.float32)
y = X

# =====================================================

print("Preparing...")
ts = time.clock()

import mxnet as mx

targets = mx.symbol.Variable('softmax_label')

data = mx.symbol.Variable('data')
data = mx.sym.Flatten(data=data)
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=400)
act1 = mx.symbol.Activation(data = fc1, name='a1', act_type="tanh")

fc2  = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=200)
act2 = mx.symbol.Activation(data = fc2, name='a2', act_type="tanh")

fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=200)
act3 = mx.symbol.Activation(data = fc3, name='a3', act_type="tanh")

fc4  = mx.symbol.FullyConnected(data = act3, name='fc4', num_hidden=200)
act4 = mx.symbol.Activation(data = fc4, name='a4', act_type="tanh")

fc5  = mx.symbol.FullyConnected(data = act4, name='fc5', num_hidden=100)
mlp = mx.symbol.LogisticRegressionOutput(data = fc5, label = targets, name='model')

model = mx.model.FeedForward(
        ctx                = mx.cpu(),
        symbol             = mlp,
        num_epoch          = 1,
        learning_rate      = 0.01,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        )

trainIter = mx.io.NDArrayIter(data = X, label = y, batch_size = 1000)

print("Time elapsed: ", time.clock() - ts)

# ===================================================

print("Calculating...")
ts = time.clock()

for _ in range(5):
        model.fit(
            X = trainIter, 
            #eval_data=valIter, 
            batch_end_callback=None, 
            epoch_end_callback=None, 
            eval_metric='mse'
            )


print("Time elapsed: ", time.clock() - ts)