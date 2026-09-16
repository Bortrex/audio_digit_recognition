"""MFCC preprocessing, label mapping, and scaler persistence."""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

MFCC_SHAPE = (32, 13)
MISSING_VALUE = -9999999

# Class 0 is non-digit/other (-1); classes 1..10 are spoken digits 0..9.
CLASS_TO_LABEL = dict(enumerate(range(-1, 10)))
LABEL_TO_CLASS = {label: index for index, label in CLASS_TO_LABEL.items()}


def encode_labels(labels):
    """Map competition labels to model classes."""
    return np.array([LABEL_TO_CLASS[int(label)] for label in labels], dtype=np.int64)


def load_training_data(data_dir="."):
    """Load features without IDs, replace padding, and encode labels."""
    data_dir = Path(data_dir)
    inputs = np.load(data_dir / "X_train.npy")[:, 1:]
    inputs[inputs == MISSING_VALUE] = 0
    labels = np.load(data_dir / "y_train.npy")[:, 1]
    return inputs, encode_labels(labels)


def prepare_training_data(inputs, labels, full_data=False, seed=1234):
    """Split before scaling, or fit on all labelled samples in full-data mode."""
    if full_data:
        train_inputs, train_labels = inputs, labels
        validation_inputs, validation_labels = None, None
    else:
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
            inputs, labels, test_size=0.1, random_state=seed, stratify=labels
        )

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_inputs = scaler.fit_transform(train_inputs).reshape(-1, 1, *MFCC_SHAPE)
    if validation_inputs is not None:
        validation_inputs = scaler.transform(validation_inputs).reshape(-1, 1, *MFCC_SHAPE)
    return train_inputs, train_labels, validation_inputs, validation_labels, scaler


def scaler_state(scaler):
    """Use tensors and primitives so a checkpoint needs no sklearn-object pickle."""
    state = {
        "feature_range": list(scaler.feature_range),
        "copy": scaler.copy,
        "clip": scaler.clip,
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": int(scaler.n_samples_seen_),
    }
    for name in ("scale_", "min_", "data_min_", "data_max_", "data_range_"):
        state[name] = torch.from_numpy(getattr(scaler, name).copy())
    return state


def restore_scaler(state):
    """Reconstruct the fitted scaler without refitting or losing float precision."""
    scaler = MinMaxScaler(
        feature_range=tuple(state["feature_range"]),
        copy=state["copy"], clip=state["clip"],
    )
    for name in ("scale_", "min_", "data_min_", "data_max_", "data_range_"):
        setattr(scaler, name, state[name].detach().cpu().numpy().copy())
    scaler.n_features_in_ = state["n_features_in_"]
    scaler.n_samples_seen_ = state["n_samples_seen_"]
    return scaler
