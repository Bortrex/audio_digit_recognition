import numpy as np
import pprint
import matplotlib.pyplot as plt


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
