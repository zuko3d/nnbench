from __future__ import print_function
import numpy as np
import time

print("Generating dataset...")
X = np.random.uniform(size = (100000, 100))
y = X

# =====================================================
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch.utils.data as data_utils

net = torch.nn.Sequential(
        nn.Linear(100, 400),
        nn.Linear(400, 200),
        nn.Linear(200, 200),
        nn.Linear(200, 200),
        nn.Linear(200, 100)
    )
net.cuda()

data = torch.FloatTensor(X)
target = torch.FloatTensor(y)

train = data_utils.TensorDataset(data, target)
train_loader = data_utils.DataLoader(train, batch_size=5000, shuffle=False)

criterion = nn.MSELoss()

optimizer = opt.SGD(net.parameters(), lr=0.01)

# ===================================================

print("Calculating...")

ts = time.clock()

for epoch in range(100):
    for minibatch in train_loader:
        data, target = minibatch
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    


print("Time elapsed: ", time.clock() - ts)