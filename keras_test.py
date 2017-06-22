from __future__ import print_function
import numpy as np
import time

print("Generating dataset...")
X = np.random.uniform(size = (100000, 100)).astype(np.float32)
y = X

# =====================================================

print("Preparing...")
ts = time.time()

import tensorflow as tf
with tf.device('/gpu:0'):
    import keras as K
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(400, activation='tanh', input_dim=100))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(100, activation='tanh'))

    #model.summary()

    model.compile(loss='mean_squared_error',
                  optimizer='sgd')

    print("Time elapsed: ", time.time() - ts)

    # ===================================================

    print("Calculating...")
    ts = time.time()

    model.fit(  X, y,
                batch_size=100000,
                epochs=100,
                verbose=0)

    print("Time elapsed: ", time.time() - ts)