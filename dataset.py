import torch
from torch.utils.data import Dataset


class Data(Dataset):
    """Convert NumPy samples and labels to tensors when indexed."""

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = torch.tensor(self.inputs[idx], dtype=torch.float32)
        label_data = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_data, label_data
