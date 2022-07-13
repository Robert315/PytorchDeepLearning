import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


transform = transforms.ToTensor()
train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transform)
# print(train_data, tes  t_data)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10,shuffle=False)

# Define layers
conv1 = nn.Conv2d(1, 6, 3, 1)  # 1 Color channel, 6 Filters(output channels, 3by3 Kernel, Stride=1
conv2 = nn.Conv2d(6, 16, 3, 1)  # 6 inpput Filters conv1, 16 Filters, 3by3 Kernel, Stride=1

# Grab the first MNIST record
for i, (X_train, y_train) in enumerate(train_data):
    break

# create a rank 4 tensor to be passed into model
x = X_train.view(1, 1, 28, 28)

# Perform the first convolution/activation
x = F.relu(conv1(x))
# Run the first pooling layer
x = F.max_pool2d(x, 2, 2)
# Perform the second convolution/activation
x = F.relu(conv2(x))
# Run the second pooling layer
x = F.max_pool2d(x, 2, 2)  # 2by2 kernel, stride=2
x = x.view(-1, 16 * 5 * 5)
print(x.shape)