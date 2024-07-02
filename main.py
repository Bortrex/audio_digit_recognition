import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer, SimpleImputer
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from torchinfo import summary

from model import Net
from utils import Data

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

width = 13  # mfcc features/coefficients
height = 32  # (max) length of utterance
classes = 11  # digits
SEED = 1234
EPOCHS = 101

# imp = IterativeImputer(max_iter=10, random_state=SEED, missing_values=-9999999)
# imp = SimpleImputer(missing_values=-9999999, strategy='mean')

# Load data_train
X_train = np.load('X_train.npy')
X_train = X_train[:, 1:]  # removing index column
# Change missing values to 0
X_train[X_train == -9999999.] = 0
y_train = np.load('y_train.npy')
y_train = y_train[:, 1]  # removing index column

# Normalize data_train
scaler = MinMaxScaler(feature_range=(-1, 1))  # StandardScaler()
X_train = scaler.fit_transform(X_train)

# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print("Dataset max(), min(), mean(), std():")
print(X_train[0].shape, X_train.max(), X_train.min(), X_train.mean(), X_train.std())

X_train = np.reshape(X_train, (-1, 1, height, width))
y_train = y_train - y_train.min()
print(f"Shape of dataset: {X_train.shape}")

# plot_class_distribution(y_train)


# Split data_train
XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED
                                                        , stratify=y_train)
test_dataset = Data(XX_test, yy_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

train_dataset = Data(XX_train, yy_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

metric = MulticlassAccuracy()
model = Net().to(device)

print("\nModel summary:")
summary(model, input_size=(128, 1, 32, 13))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)  # , nesterov=True, momentum=0.9)


# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True, factor=0.5, min_lr=1e-4)


def train(model, train_loader, optimizer, criterion, metric):
    model.train()
    metric.reset()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        metric.update(outputs, labels)

    return loss, metric.compute()


def evaluate(model, test_loader, metric):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            metric.update(outputs, labels)

    return metric.compute()


if __name__ == "__main__":

    print("\nTraining...")
    for ep in range(EPOCHS):

        loss, metric_train = train(model, train_loader, optimizer, criterion, metric)
        # scheduler.step(loss)
        metric_test = evaluate(model, test_loader, metric)
        if ep % 5 == 0:
            print(f'Epoch {ep}, Loss: {loss.item():.4f}, Accuracy: {metric_train.item():.4f}')
            print(f'\tAccuracy: {metric_test.item():.4f}')
