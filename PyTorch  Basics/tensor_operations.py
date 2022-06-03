import torch
import numpy as np

x = torch.arange(6).reshape(3, 2)

# print(type(x[1,1]))

x = torch.arange(10)

# print(x.view(2, 5))
# print(x.shape)  # to see dimensions

a = torch.tensor([1., 2., 3])
b = torch.tensor([4., 5., 6])

c = torch.add(a, b)
# print(c)

d = a.mul(b)  # a * b
a = a.mul_(b)  # redefine a as a * b

a = torch.tensor([[0, 2, 4], [1, 3, 5]])
b = torch.tensor([[6, 7], [8, 9], [10, 11]])
c = torch.mm(a, b)  # matrix multiplication, dosen`t metter if the matrixs aren't the same shape

x = torch.Tensor([1, 2, 3, 4])
# print(x.norm())
# print(c.numel()) # number of all elements

###### Exercices #######

np.random.seed(42)
manual_seed = torch.manual_seed(42)

arr = np.random.randint(0,5,6)

x = torch.from_numpy(arr)

x = x.reshape(3,2)
print(x)

print(x[:, 1:])

y = torch.randint(low=0,high=5,size=(2,3))
print(y)

z = torch.mm(a,b)
print(z)