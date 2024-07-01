
import os
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy

from utils import plot_class_distribution, Data
from model import Net

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

width = 13  # mfcc features/coefficients
height = 32  # (max) length of utterance
classes = 11  # digits
SEED = 1234
EPOCHS = 101

# Load data_train
X_train = np.load('X_train.npy')
X_train = X_train[:,1:]
X_train[X_train == -9999999.] = 0
y_train = np.load('y_train.npy')

def log_transform(data):
    data[data == -9999999] = 0  # replace placeholders before log transform
    data = np.log1p(data - data.min())  # log1p is log(1 + x) to handle zero values
    return data

# X_train = log_transform(X_train)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1)) # StandardScaler()
X_train = scaler.fit_transform(X_train)


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# no_zero = np.where(X_train==-9999999, 0, X_train)[:,1:]

# print(no_zero.max(), no_zero.min(), no_zero.mean(), no_zero.std())
print(X_train[0].shape, X_train.max(), X_train.min(), X_train.mean(), X_train.std())
print(y_train[0].shape, y_train.max(), y_train.min(), y_train.mean(), y_train.std())

# X_train = np.reshape(X_train[:,1:], (-1, height, width))

X_train = np.reshape(X_train, (-1, 1, height, width))
y_train = y_train[:,1]
y_train = y_train - y_train.min()

print(X_train[0].shape, X_train.max(), X_train.min(), X_train.mean(), X_train.std())
print(y_train[0].shape, y_train.max(), y_train.min(), y_train.mean(), y_train.std())

classes = np.unique(y_train)

print(y_train[6670])
# y_train_hot = F.one_hot(torch.tensor(y_train - y_train.min()), len(classes))
# print(y_train_hot[6670])
y_train_hot = OneHotEncoder(handle_unknown='ignore').fit_transform(y_train.reshape(-1, 1) ).toarray()
print(y_train_hot.shape)
# plot_class_distribution(y_train)

print(X_train.shape)

# Split data_train
XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED
                                                        , stratify=y_train)

train_dataset = Data(XX_train, yy_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

metric = MulticlassAccuracy()
model = Net().to(device)

summary(model, input_size=(128, 1, 32, 13))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for ep in range(EPOCHS):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        metric.update(outputs.argmax(1), labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # break

    if ep % 5 == 0:

        print(f'Epoch {ep}, Loss: {loss.item():.4f}, Accuracy: {metric.compute().item():.4f}')

print()
print(outputs.argmax(1), labels)



# # With square kernels and equal stride
# m = nn.Conv2d(16, 33, 3)
# # m = nn.Conv2d(16, 33, 3, stride=2)
# # # non-square kernels and unequal stride and with padding
# # m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# # non-square kernels and unequal stride and with padding and dilation
# # m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
# input = torch.randn(20, 16, 50, 100)
# output = m(input)
# print(input.size(), output.size())