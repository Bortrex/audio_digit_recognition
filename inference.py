"""Checkpoint restoration and single-WAV classification."""

from pathlib import Path
import pickle

import numpy as np
import torch

from audio_preprocessing import wav_to_features
from model import Net
from preprocessing import CLASS_TO_LABEL, MFCC_SHAPE, MISSING_VALUE, restore_scaler


def load_checkpoint(path, device="cpu"):
    """Restore a supported checkpoint on CPU by default, without fitting a scaler."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint["format_version"] != 2:
            raise ValueError("only checkpoint format version 2 is supported")
        if checkpoint["model"]["output_activation"] != "identity":
            raise ValueError("expected unrestricted model logits")
        metadata = checkpoint["preprocessing"]
        expected = {
            "mfcc_shape": list(MFCC_SHAPE), "input_channels": 1, "id_column": 0,
            "missing_value": MISSING_VALUE, "replacement_value": 0,
            "reshape_order": "C", "tensor_dtype": "float32",
            "scaling_before_tensor_conversion": True,
        }
        if any(metadata[key] != value for key, value in expected.items()):
            raise ValueError("unsupported input preprocessing metadata")
        if checkpoint["class_to_label"] != CLASS_TO_LABEL:
            raise ValueError("unsupported class/label mapping")
        scaler = restore_scaler(checkpoint["scaler"])
        if scaler.n_features_in_ != np.prod(MFCC_SHAPE):
            raise ValueError("scaler must have 416 features")
        for name in ("scale_", "min_", "data_min_", "data_max_", "data_range_"):
            values = getattr(scaler, name)
            if values.shape != (416,) or not np.isfinite(values).all():
                raise ValueError(f"invalid scaler attribute: {name}")
        model = Net().to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
    except (OSError, EOFError, pickle.UnpicklingError, KeyError, TypeError,
            AttributeError, ValueError, RuntimeError, IndexError) as error:
        raise ValueError(f"Incompatible or unreadable checkpoint '{path}': {error}") from error
    return model, scaler, checkpoint


def prepare_wav_input(path, scaler, metadata):
    """Apply saved scaling to one padded MFCC row and create a model tensor."""
    shape = tuple(metadata["mfcc_shape"])
    features = wav_to_features(path, shape, metadata["missing_value"])
    features[features == metadata["missing_value"]] = metadata["replacement_value"]
    features = scaler.transform(features.reshape(1, -1))
    inputs = features.reshape(1, metadata["input_channels"], *shape,
                              order=metadata["reshape_order"])
    return torch.as_tensor(inputs, dtype=getattr(torch, metadata["tensor_dtype"]))


def decode_class(class_index, class_to_label):
    label = class_to_label[class_index]
    return "other" if label == -1 else str(label)


def predict_wav(path, checkpoint_path="checkpoints/model.pt", device="cpu", return_logits=False):
    """Return a decoded label, optionally with CPU logits from the same forward pass."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")
    model, scaler, checkpoint = load_checkpoint(checkpoint_path, device)
    inputs = prepare_wav_input(path, scaler, checkpoint["preprocessing"]).to(device)
    with torch.no_grad():
        logits = model(inputs)
        class_index = logits.argmax(dim=1).item()
    label = decode_class(class_index, checkpoint["class_to_label"])
    return (label, logits[0].cpu()) if return_logits else label
