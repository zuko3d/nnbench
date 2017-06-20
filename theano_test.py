from __future__ import print_function
import numpy as np
import time

print("Generating dataset...")
X = np.random.uniform(size = (100000, 100)).astype(np.float32)
yv = X

# =====================================================
import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import nnet
rng = numpy.random

x = T.matrix("x")
y = T.matrix("y")

inp_sz = 100

w_1 = theano.shared(rng.randn(inp_sz, 400), name="w1")
b_1 = theano.shared(numpy.zeros((400,)), name="b1")

w_2 = theano.shared(rng.randn(400, 200), name="w2")
b_2 = theano.shared(numpy.zeros((200,)), name="b2")

w_3 = theano.shared(rng.randn(200, 200), name="w3")
b_3 = theano.shared(numpy.zeros((200,)), name="b3")

w_4 = theano.shared(rng.randn(200, 200), name="w4")
b_4 = theano.shared(numpy.zeros((200,)), name="b4")

w_5 = theano.shared(rng.randn(200, 100), name="w5")
b_5 = theano.shared(numpy.zeros((100,)), name="b5")

p_1 = nnet.sigmoid(-T.dot(nnet.sigmoid(-T.dot(nnet.sigmoid(-T.dot(nnet.sigmoid(-T.dot(nnet.sigmoid(-T.dot(
    x, w_1)-b_1), w_2)-b_2), w_3)-b_3), w_4)-b_4), w_5)-b_5)

xent = -y*T.log(p_1) - (1-y)*T.log(1 - p_1)
cost = xent.mean()

gw_1, gb_1, gw_2, gb_2, gw_3, gb_3, gw_4, gb_4, gw_5, gb_5 = T.grad(cost, [w_1, b_1, w_2, b_2, w_3, b_3, w_4, b_4, w_5, b_5])

print("Preparing model...")
ts = time.clock()

train = theano.function(
        inputs = [x, y],
        outputs = [p_1, xent], 
        #allow_input_downcast=True,
        # updates = {
        #         w_1 : w_1-0.1*gw_1, 
        #         b_1 : b_1-0.1*gb_1, 
        #         w_2 : w_2-0.1*gw_2, 
        #         b_2 : b_2-0.1*gb_2,
        #         w_3 : w_3-0.1*gw_3, 
        #         b_3 : b_3-0.1*gb_3,
        #         w_4 : w_4-0.1*gw_4, 
        #         b_4 : b_4-0.1*gb_4,
        #         w_5 : w_5-0.1*gw_5, 
        #         b_5 : b_5-0.1*gb_5
        #     }
        updates = [
            (w_1 , w_1-0.1*gw_1), 
            (b_1 , b_1-0.1*gb_1), 
            (w_2 , w_2-0.1*gw_2), 
            (b_2 , b_2-0.1*gb_2),
            (w_3 , w_3-0.1*gw_3), 
            (b_3 , b_3-0.1*gb_3),
            (w_4 , w_4-0.1*gw_4), 
            (b_4 , b_4-0.1*gb_4),
            (w_5 , w_5-0.1*gw_5), 
            (b_5 , b_5-0.1*gb_5)
        ]
    )
print("Time elapsed: ", time.clock() - ts)

# ===================================================

print("Calculating...")
ts = time.clock()

bsz = 100000
bs = 100000 / bsz

ts = time.clock()

for _ in range(1):
    for b in range(bs):
        pred, err = train(X[b * bsz : (b + 1) * bsz], yv[b * bsz : (b + 1) * bsz])

print("Time elapsed: ", time.clock() - ts)