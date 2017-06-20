from __future__ import print_function
import numpy as np
import time

print("Generating dataset...")
X = np.random.uniform(size = (100000, 100)).astype(np.float32)
y = X

# =====================================================

print("Preparing...")
ts = time.clock()

from caffe2.python import core, workspace, utils, model_helper
from caffe2.python import brew
import numpy as np

model = model_helper.ModelHelper(name="train")

workspace.FeedBlob('X', X)
workspace.FeedBlob('y', y)

fc1 = brew.fc(model, 'X', "fc1", 100, 400)
fc2 = brew.fc(model, fc1, "fc2", 400, 200)
fc3 = brew.fc(model, fc2, "fc3", 200, 200)
fc4 = brew.fc(model, fc3, "fc4", 200, 200)
fc5 = brew.fc(model, fc4, "fc5", 200, 100)

model.Validate()

dist = model.SquaredL2Distance(['fc5', 'y'], "dist")
loss = model.AveragedLoss(dist, "loss")
gradient_map = model.AddGradientOperators([loss])

workspace.RunNetOnce(model.param_init_net)

workspace.CreateNet(model)

print("Time elapsed: ", time.clock() - ts)

# ===================================================

print("Calculating...")
ts = time.clock()

for i in range(100):
    workspace.RunNet(model.net)

print("Time elapsed: ", time.clock() - ts)