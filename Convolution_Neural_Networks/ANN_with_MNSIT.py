import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)

test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transform)
# print(train_data[0])
image, label = train_data[0]
print('Shape', image.shape, '\nLabel:', label)
plt.imshow(train_data[0][0].reshape(28, 28), cmap='gray')
# plt.show()
torch.manual_seed(101)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

from torchvision.utils import make_grid
np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))  # to widen the printed array

# Grab the first batch of images
for images, labels in train_loader:
    break

# Print the first 12 labels
# print('Labels: ', labels[:12].numpy())

# Print the first 12 images
im = make_grid(images[:12], nrow=12)  # the default nrow is 8
plt.figure(figsize=(10,4))
# We need to transpose the images from CWH to WHC
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=784, out_sz=10, layers=[100,84]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], out_sz)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


torch.manual_seed(101)
model = MultilayerPerceptron()
# print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# print(images.shape)
images.view(100, -1).size()

import time
start_time = time.time()

epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Run the training batches
    for batch, (X_train, y_train) in enumerate(train_loader):
        batch += 1
        pass
        # Apply the model
        y_pred = model(X_train.view(100, -1))  # Here we flatten X_train
        loss = criterion(y_pred, y_train)

        # Totally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        # Update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 200 == 0:
            acc = trn_corr.item() * 100 / (100 * batch)
            print(f'Epoch: {i}  batch {batch}  loss: {loss.item()}  accuracy: {acc}')

    train_losses.append(loss)
    train_correct.append(trn_corr)
    test_correct.append(tst_corr)

total_time = time.time() - start_time
print(f'Duration: {total_time} seconds')

print(test_correct) # contains the results of all 10 epochs
print()
print(f'Test accuracy: {test_correct[-1].item()*100/10000:.3f}%')