import numpy as np
import time

print("Generating dataset...")
X = np.random.uniform(size = (100000, 100))
y = X

#=====================================================

from neon.callbacks.callbacks import Callbacks
from neon.data import ArrayIterator
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, Explin, CrossEntropyBinary, Misclassification, Tanh, Softmax, Accuracy
from neon.backends import gen_backend
from neon.transforms import SumSquared, MeanSquared

backend = 'cpu'
print("backend: ", backend)

be = gen_backend(backend = backend)

be.bsz = 10000

train = ArrayIterator(X = X, y = y, make_onehot = False)

print("Init mlp...")

# setup weight initialization function
init_norm = Gaussian(loc=0.0, scale=0.1)

layers = [
            Affine(nout = 400, init=init_norm, activation=Tanh()),
            Affine(nout = 200, init=init_norm, activation=Tanh()),
            Affine(nout = 200, init=init_norm, activation=Tanh()),
            Affine(nout = 200, init=init_norm, activation=Tanh()),
            Affine(nout = 100, init=init_norm, activation=Tanh())
         ]

# setup cost function
cost = GeneralizedCost(costfunc=SumSquared())

# setup optimizer
optimizer = GradientDescentMomentum(
    0.1, momentum_coef=0.9)

# initialize model object
mlp = Model(layers=layers)

ts = time.clock()

# run fit
mlp.fit(train, 
        callbacks=Callbacks(mlp),
        optimizer=optimizer,
        num_epochs=10, 
        cost=cost
        )

print("Time elapsed: ", time.clock() - ts)