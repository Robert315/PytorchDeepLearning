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
import time

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transform)
# print(train_data, tes  t_data)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

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


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

torch.manual_seed(42)
model = ConvolutionalNetwork()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import time

start_time = time.time()

epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1

        # Apply the model
        y_pred = model(X_train)  # we don't flatten X-train here
        loss = criterion(y_pred, y_train)

        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        if b % 600 == 0:
            print(f'epoch: {i:2}  batch: {b:4} [{10 * b:6}/60000]  loss: {loss.item():10.8f}  \
accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            # Apply the model
            y_val = model(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed

# Extract the data all at once, not in batches
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)  # we don't flatten the data this time
        predicted = torch.max(y_val,1)[1]
        correct += (predicted == y_test).sum()
print(f'Test accuracy: {correct.item()}/{len(test_data)} = {correct.item()*100/(len(test_data)):7.3f}%')

#  We can track the index positions of "missed" predictions, and extract the corresponding image and label.
#  We'll do this in batches to save screen space.
misses = np.array([])
for i in range(len(predicted.view(-1))):
    if predicted[i] != y_test[i]:
        misses = np.append(misses, i).astype('int64')

# Display the number of misses
len(misses)

x = 2019
plt.figure(figsize=(1, 1))
plt.imshow(test_data[x][0].reshape((28, 28)), cmap="gist_yarg")
plt.show()

model.eval()
with torch.no_grad():
    new_pred = model(test_data[x][0].view(1, 1, 28, 28)).argmax()
print("Predicted value:", new_pred.item())
