import torch
import numpy as np

# print(torch.__version__)

arr = np.array([1, 2, 3, 4, 5])

x = torch.from_numpy(arr)

arr2d = np.arange(0.0, 12.0)

arr2d = arr2d.reshape(4, 3)

x2 = torch.from_numpy(arr2d)  # to convert in a TENSOR / float64 mean 64 bit precision on that floating point data

my_arr = np.arange(0, 10)
my_tensor = torch.tensor(my_arr)  # create a copy of the matrix
my_second_tensor = torch.from_numpy(my_arr)
# print(my_tensor, my_second_tensor)

my_arr[0] = 99
# print(my_tensor, my_second_tensor)


new_arr = np.array([1, 2, 3])

my_tnsr = torch.tensor(new_arr)
my_sec_tnsr = torch.Tensor(new_arr)  # torch with 'T' change your array in a float type
# print(my_tnsr, my_sec_tnsr)

empty_torch = torch.zeros(4, 3, dtype=torch.int64)  # by default is float

x_torch = torch.arange(0, 18, 2).reshape(3, 3)

x2_torch = torch.tensor([1, 2, 3])  # to convert a normal python list in a tensor quicly
x2_torch = x2_torch.type(torch.int32)  # convert

rand_torch = torch.rand(4,3)  # create a tensor with random numbers between 0-1

randn_torch = torch.randn(4,3)  # create a tensor with random numbers between 0-1

rand_int = torch.randint(low=0, high=10, size=(5, 5))

manual_seed = torch.manual_seed(42)  # keep the same values of a random tensor
manual = torch.rand(2, 3)
print(manual)