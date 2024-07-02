import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = torch.tensor(self.inputs[idx], dtype=torch.float32)
        label_data = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_data, label_data


def plot_class_distribution(data, **kwargs):
    """
    Plots the distribution of classes in a NumPy array.

    Args:
      data: The NumPy array containing the class labels (up to 10k values).
      **kwargs: Additional keyword arguments to be passed to the plotting function.

    """

    # Create a count of each class (assuming integer labels)
    bins, counts = np.unique(data, return_counts=True)
    pprint.pprint(dict(zip(bins, counts)))

    # Plot the histogram with labels
    plt.bar(bins, counts, label="Class Distribution", **kwargs)
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.title("Distribution of Classes")
    plt.xticks(bins)  # Set x-axis ticks to class labels
    plt.legend()
    plt.show()
