from __future__ import print_function
import numpy as np
import time

print("Generating dataset...")
X = np.random.uniform(size = (100000, 100)).astype(np.float32)
y = X

# =====================================================

print("Preparing...")
ts = time.clock()

import cntk
from cntk.layers import *

print(cntk.try_set_default_device(cntk.gpu(0)))

x = cntk.input_variable(100, np.float32)
target = cntk.input_variable(100, np.float32)

model = Sequential ([
    Dense(400, activation=tanh),
    Dense(200, activation=tanh),
    Dense(200, activation=tanh),
    Dense(200, activation=tanh),
    Dense(100, activation=tanh)
]) (x)

loss = cntk.squared_error(model, target)
learner = cntk.sgd(model.parameters, cntk.learning_rate_schedule(0.01, cntk.UnitType.minibatch))

trainer = cntk.Trainer(model, (loss, loss), [learner])

print("Time elapsed: ", time.clock() - ts)

# ===================================================

print("Calculating...")
ts = time.clock()

for _ in range(100):
	trainer.train_minibatch({x : X, target : y})

print("Time elapsed: ", time.clock() - ts)