from __future__ import print_function
import numpy as np
import time

print("Generating dataset...")
X = np.random.uniform(size = (100000, 100))
y = X

#=====================================================
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch.utils.data as data_utils

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 100)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

net = Net()
print(net)

data = torch.FloatTensor(X)
target = torch.FloatTensor(y)

train = data_utils.TensorDataset(data, target)
train_loader = data_utils.DataLoader(train, batch_size=1000, shuffle=False)

criterion = nn.MSELoss()

ts = time.clock()

for epoch in range(10):
    for minibatch in train_loader:
        data, target = minibatch
        optimizer = opt.SGD(net.parameters(), lr=0.01)

        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(Variable(data))
        loss = criterion(output, Variable(target))
        loss.backward()
        optimizer.step()    # Does the update


print("Time elapsed: ", time.clock() - ts)