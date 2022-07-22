import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

path = '../Data/CATS_DOGS/'
img_names = []

for folder, subfolders, filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder + '/' + img)

# print('Images: ', len(img_names))

# Start by creating a list
img_sizes = []
rejected = []

for item in img_names:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)
    except:
        rejected.append(item)

# print(f'Images:  {len(img_sizes)}')
# print(f'Rejects: {len(rejected)}')

# Convert the list to a DataFrame
df = pd.DataFrame(img_sizes)
# print(df.head())
# Run summary statistics on image widths
# print(df[0].describe())

dog = Image.open('../Data/CATS_DOGS/train/DOG/14.jpg')
# print(dog.size)
r, g, b = dog.getpixel((0, 0))
# print(r,g,b)

transform = transforms.Compose([
    transforms.ToTensor()
])
im = transform(dog)
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 3)))
