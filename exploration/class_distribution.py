import pprint

import matplotlib.pyplot as plt
import numpy as np


def plot_class_distribution(data, **kwargs):
    """Print class counts and plot them; pass keyword arguments to plt.bar."""

    bins, counts = np.unique(data, return_counts=True)
    pprint.pprint(dict(zip(bins, counts)))

    plt.bar(bins, counts, label="Class Distribution", **kwargs)
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.title("Distribution of Classes")
    plt.xticks(bins)
    plt.legend()
    plt.show()
