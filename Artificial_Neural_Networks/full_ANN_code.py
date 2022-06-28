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
