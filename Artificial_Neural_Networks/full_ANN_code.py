import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/robert/Documents/PyThorch_Bootcamp/PYTORCH_NOTEBOOKS/Data/NYCTaxiFares.csv')


def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers

    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[long2] - df[long1])

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = (r * c)  # in kilometers

    return d


df['dist_km'] = haversine_distance(df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])  # convert in a datetime object

my_time = df['pickup_datetime'][0]
df['EDTdate'] = df['pickup_datetime'] - pd.Timedelta(hours=4)
df['Hour'] = df['EDTdate'].dt.hour
df['AMorPM'] = np.where(df['Hour'] < 12, 'AM', 'PM')
df['Weekday'] = df['EDTdate'].dt.strftime('%a')

cat_cols = ['Hour', 'AMorPM', "Weekday"]
cont_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
             'dropoff_latitude', 'passenger_count', 'dist_km']
y_col = ['fare_amount']

#  converting categorical values to numerical codes
for cat in cat_cols:
    df[cat] = df[cat].astype('category')

# Transform in numpy array
hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values  # 0 or 1 for am or pm
wkdy = df['Weekday'].cat.codes.values  # 0,...,6

cats = np.stack([hr, ampm, wkdy], axis=1)  # to combine the three categorical columns into one input array

# Convert continuous variables to a tensor
cats = torch.tensor(cats, dtype=torch.int64)
conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts = torch.tensor(conts, dtype=torch.float)

# Convert labels to a tensor
y = torch.tensor(df[y_col].values, dtype=torch.float)

# This will set embedding sizes for Hours, AMvsPM and Weekdays
cat_sizes = [len(df[col].cat.categories) for col in cat_cols]
emb_sizes = [(size, min(50, (size + 1) // 2)) for size in cat_sizes]

catz = cats[:2]
selfembeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_sizes])
print(selfembeds)

#  FORWARD METHOD(cats)
embeddingz = []

for i, e in enumerate(selfembeds):
    embeddingz.append(e(catz[:, i]))

z = torch.cat(embeddingz, 1)
selfembdrop = nn.Dropout(0.4)
z = selfembdrop(z)


class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()

        #  Set up the embedded layers with torch.nn.ModuleList() and torch.nn.Embedding()
        # Categorical data will be filtered through these Embeddings in the forward section.
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])

        # Set up a dropout function for the embeddings with torch.nn.Dropout() The default p-value=0.5
        self.emb_drop = nn.Dropout(p)
        # Set up a normalization function for the continuous variables with torch.nn.BatchNorm1d()
        self.bn_cont = nn.BatchNorm1d(n_cont)
        # Set up a sequence of neural network layers where each level includes a Linear function, an activation function
        # (we'll use ReLU), a normalization step, and a dropout layer. We'll combine the list of layers with torch.nn.Sequential()
        layerlist = []
        n_emb = sum(nf for ni, nf in emb_szs)
        n_in = n_emb + n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
          embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x

torch.manual_seed(33)
model = TabularModel(emb_szs, conts.shape[1], 1, [200, 100], p=0.4)
print(model)
criterion = nn.MSELoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 60000
test_size = int(batch_size * .2)

cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size-test_size:batch_size]
con_train = conts[:batch_size-test_size]
con_test = conts[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]

import time

start_time = time.time()
epochs = 300
losses = []

for i in range(epochs):
    i += 1
    y_pred = model(cat_train, con_train)
    loss = torch.sqrt(criterion(y_pred, y_train)) # RMSE
    losses.append(loss)

    if i % 10 == 0:
        print(f'epoch {i} loss is {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

duration = time.time() - start_time
print(f'Training took {duration/60} minutes')


