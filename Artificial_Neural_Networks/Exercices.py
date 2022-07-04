import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time

df = pd.read_csv('/home/robert/DeepLearning - Course/PYTORCH_NOTEBOOKS/Data/income.csv')

# 1. Separate continuous, categorical and label column names
# print(df.columns)

ctg_cols = ['sex', 'education', 'marital-status', 'workclass', 'occupation']
cont_cols = ['age', 'hours-per-week']
y_col = ['label']
# print(f'ctg_cols  has {len(ctg_cols)} columns')
# print(f'cont_cols has {len(cont_cols)} columns')
# print(f'y_col     has {len(y_col)} column')

# 2. Convert categorical columns to category dtypes
# print(df.dtypes)
for ctg in ctg_cols:
    df[ctg] = df[ctg].astype('category')

# 3. Set the embedding sizes
# Create a variable "cat_szs" to hold the number of categories in each variable.
# Then create a variable "emb_szs" to hold the list of (category size, embedding size) tuples.
ctg_szs = [len(df[col].cat.categories) for col in ctg_cols]
emb_szs = [(size, min(50, size+1)//2) for size in ctg_szs]
# print(emb_szs)

# 4. Create an array of categorical values

sex = df['sex'].cat.codes.values
education = df['education'].cat.codes.values
marital_status = df['marital-status'].cat.codes.values
workclass = df['workclass'].cat.codes.values
occupation = df['occupation'].cat.codes.values
ctgs = np.stack([sex, education, marital_status, workclass, occupation], axis=1)

# 5. Convert "cats" to a tensor
ctgs = torch.tensor(ctgs, dtype=torch.int64)

# 6. Create an array of continuous values
conts = np.stack([df[col].values for col in cont_cols], axis=1)

# 7. Convert "conts" to a tensor
conts = torch.tensor(conts, dtype=torch.float32)
# print(conts)

# 8. Create a tensor called "y" from the values in the label column
y = torch.tensor(df[y_col].values).flatten()

# 9. Create train and test sets from ctgs, conts, and y
batch_size = 30000
test_size = 5000

ctg_train = ctgs[:batch_size - test_size]
ctg_test = ctgs[batch_size - test_size:batch_size]
cont_train = conts[:batch_size - test_size]
cont_test = conts[batch_size - test_size:batch_size]
y_train = y[:batch_size - test_size]
y_test = y[batch_size - test_size:batch_size]


class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        # Call the parent __init__
        super().__init__()

        # Set up the embedding, dropout, and batch normalization layer attributes
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        # Assign a variable to hold a list of layers
        layerlist = []

        # Assign a variable to store the number of embedding and continuous layers
        n_emb = sum((nf for ni, nf in emb_szs))
        n_in = n_emb + n_cont

        # Iterate through the passed-in "layers" parameter (ie, [200,100]) to build a list of layers
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))

        # Convert the list of layers into an attribute
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        # Extract embedding values from the incoming categorical data
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
        x = torch.cat(embeddings, 1)
        # Perform an initial dropout on the embeddings
        x = self.emb_drop(x)

        # Normalize the incoming continuous data
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)

        # Set up model layers
        x = self.layers(x)
        return x

# 10. Set the random seed
torch.random.manual_seed(33)

# 11. Create a TabularModel instance
model = TabularModel(emb_szs, conts.shape[1], 2, [50], p=0.4)

# 12. Define the loss and optimization functions
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
start_time = time.time()

epochs = 300
losses = []

for i in range(epochs):
    i += 1
    y_pred = model(ctg_train, cont_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    if i % 10 == 0:
        print(f'epochs {i}  loss is {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

duration = time.time() - start_time
print(f'Training took {duration} seconds')

# 13. Plot the Cross Entropy Loss against epochs
# plt.plot(range(epochs), losses)
# plt.show()


with torch.no_grad():
    y_val = model(ctg_test, cont_test)
    loss = criterion(y_val, y_test)
print(loss)

# 15. Calculate the overall percent accuracy
rows = len(y_test)
correct = 0

for i in range(rows):
    if y_val[i].argmax().item() == y_test[i]:
        correct +=1

pct_corr = 100*correct/rows
print(pct_corr)