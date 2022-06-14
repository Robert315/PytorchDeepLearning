import matplotlib
import torch

# x = torch.tensor(2.0, requires_grad=True)
# y = 2*x**4 + x**3 + 3*x**2 + 5*x + 1
# print(y.backward, x.grad)

# x = torch.tensor([[1., 2., 3.], [3., 2., 1.]], requires_grad=True)
# print(x)
#
# y = 3*x + 2
# print(y)
#
# z = 2*y**2
# print(z)
#
# out = z.mean()
# out.backward()
# print(x.grad)


##### Linear Regression #####

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


X = torch.linspace(1,50,50).reshape(-1, 1)
# print(X)
torch.manual_seed(71)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
y = 2*X + 1 + e

plot = plt.scatter(X.numpy(), y.numpy())


torch.manual_seed(59)

model = nn.Linear(in_features=1,out_features=1)
# print(model.weight,model.bias)

class Model(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features,out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

torch.manual_seed(59)

model = Model(1,1)
# print(model.linear.weight)
# print(model.linear.bias)

x = torch.tensor([2.0])
print(model.forward(x))

x1 = np.linspace(0.0, 50.0, 50)
w1 = 0.1059
b1 = 0.9637

y1 = w1*x1 + b1

plt.scatter(X.numpy(), y.numpy())
plt.plot(x1, y1, 'r')

criterion = nn.MSELoss() # Loss function (Mean Squared Error)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#  After we set a Loss Funtions and we have a optimization, now we need to TRAINING the Model

epochs = 50 # Set a reasonably large number of passes
losses = []  # create a list to store loss values. This will let us view our prgress afterward

for i in range(epochs):
    i = i + 1
    y_pred = model.forward(X)  # create a prediciton set by running X through the current model parameters
    loss = criterion(y_pred, y)  # calculate our loss(ERROR)
    losses.append(loss)
    print(f'epoch {i} loss: {loss.item()} weight: {model.linear.weight.item()} bias: {model.linear.bias.item()}')
    #Gradients accumulate with every backprop.
    # To prevent compounding we need to reset the stored gradient for each new epoch.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

w1,b1 = model.linear.weight.item(), model.linear.bias.item()
print(f'Current weight: {w1:.8f}, Current bias: {b1:.8f}')
print()

y1 = x1*w1 + b1
print(x1)
print(y1)
plt.scatter(X.numpy(), y.numpy())
plt.plot(x1, y1, 'r')
plt.title('Current Model')
plt.ylabel('y')
plt.xlabel('x')
plt.show()