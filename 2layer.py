import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from numpy.linalg import norm
import random
import copy


# d：dimension of data
# m: number of hidden layer

# f(x) = V\sigma(Wx), x: d*1, W: m*d, V: 2*m
class TwoLayerNet(nn.Module):   
    def __init__(self, m, d):
        super(TwoLayerNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(d, m, bias=False), nn.ReLU(True),
            nn.Linear(m, 2, bias=False))    
    
    def forward(self, x):
        y_pred = self.layer(x)
        return y_pred

# Weight-initialization
def weight_init(module):
    if isinstance(module, nn.Linear):
        size = module.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        std = np.sqrt(2.0 / (fan_in + fan_out))
        module.weight.data.normal_(0, std)

# Generate 2n data
def initialize_data(d, n):
    X0 = np.random.randn(n,d)+np.ones([n,d])
    X1 = np.random.randn(n,d)-np.ones([n,d])
    X = np.vstack((X0, X1))
    for j in range(2*n):
        X[j] /= np.linalg.norm(X[j])
    return X

# Calculate the product of norm of weights
def norm_product(model):
    q = 1
    l = list(model.parameters())
    for i in range(len(l)):
        q *= norm(l[i].detach().numpy(), 2)
    return q

# Mixup dataset
def mixup(X, n):
    NewData = []
    MixFactor = []
    for i in range(n):
        for j in range(n):
            t = random.random()
            Y = t * X[i] + (1-t) * X[n+j]
            NewData.append(Y)
            MixFactor.append(t)
    return NewData, MixFactor

# Generate mixed labels
def creat_labels(l):
    ml = []
    for i in range(len(l)):
        ml.append([l[i],1-l[i]])
    return ml

m = 10
d = 10
n = 1
Lossi = []
Normi = []
MixLossi = []
MixNormi = []

# Generate two different dataset: X and Y
M = initialize_data(d, n)
T = mixup(M, n)
X = torch.from_numpy(M).double()
Y = torch.tensor(T[0]).double()


# Generate two models with same initialization
model = TwoLayerNet(m, d).double()
model.apply(weight_init)
mixmodel = copy.deepcopy(model)

# Two loss functions
loss = nn.CrossEntropyLoss()
mixloss = nn.MultiLabelSoftMarginLoss()

# Generate two separate optimizers
optimizer = optim.SGD(model.parameters(), lr=0.01)
mixoptimizer = optim.SGD(mixmodel.parameters(), lr=0.01)

# Generate two different targets
tar = [0]*n + [1]*n
target = torch.tensor(tar)
Mixtar = torch.tensor(creat_labels(T[1]))


# Optimization for the original model
for t in range(1000):
    Loss = loss(model(X), target)
    Lossi.append(Loss)
    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()
    nor = norm_product(model)
    Normi.append(nor)
    if Loss < 0.01:
       break


# Optimization for the mixup model
for t in range(8000):
    MixLoss = mixloss(mixmodel(Y), Mixtar)
    MixLossi.append(MixLoss)
    mixoptimizer.zero_grad()
    MixLoss.backward()
    mixoptimizer.step()
    Mixnor = norm_product(mixmodel)
    MixNormi.append(Mixnor)
    if MixLoss < 0.01:
        break


# Visualization 
plt.figure()
plt.plot(Normi[:])
plt.figure()
plt.plot(Lossi[:])
plt.figure()
plt.plot(MixNormi[:])
plt.figure()
plt.plot(MixLossi[:])