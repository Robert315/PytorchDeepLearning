import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt


class Model(nn.Module):
    # to inheriting nn module
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        # How many Layers?
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)  # input layer
        self.fc2 = nn.Linear(h1, h2)  # hidden layer
        self.out = nn.Linear(h2, out_features)  # output layer
        # fc1(fully connected layer
        # Input Layer (4 features) --> Hidden Layer 1 --> Hidden Layer 2 --> output (3 classes)

    def forward(self, x):   # set our propagation method
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))   # F.relu (rectified linear unit function)
        x = self.out

        return x

torch.manual_seed(32)
model = Model()

df = pd.read_csv('~/PycharmProjects/PytorchDeepLearning/Iris.csv')
print(df.head(), df.tail())




