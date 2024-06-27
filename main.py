
import os
import torch
import torch.nn.functional as F
import numpy as np


from utils import plot_class_distribution

width = 13  # mfcc features/coefficients
height = 32  # (max) length of utterance
classes = 11  # digits

# Load data_train
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

print(X_train[0].shape, X_train.max(), X_train.min(), X_train.mean(), X_train.std())
print(y_train[0].shape, y_train.max(), y_train.min(), y_train.mean(), y_train.std())

# X_train = np.reshape(X_train[:,1:], (-1, height, width))
X_train = X_train[:,1:]
X_train = np.reshape(X_train, (-1, height, width))
y_train = y_train[:,1]


print(X_train[0].shape, X_train.max(), X_train.min(), X_train.mean(), X_train.std())
print(y_train[0].shape, y_train.max(), y_train.min(), y_train.mean(), y_train.std())

classes = np.unique(y_train)

print(y_train[6670])
y_train_hot = F.one_hot(torch.tensor(y_train - y_train.min()), len(classes))
print(y_train_hot[6670])
plot_class_distribution(y_train)