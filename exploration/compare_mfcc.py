"""Local-only diagnostic: python -m exploration.compare_mfcc WAV --sample-id ID."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audio_preprocessing import wav_to_features
from inference import decode_class, load_checkpoint
from preprocessing import MISSING_VALUE


def compare(wav, dataset, sample_id, checkpoint_path):
    """Compare raw features by ID, then classify both through the same checkpoint."""
    rows = np.load(dataset, mmap_mode="r", allow_pickle=False)
    matches = np.flatnonzero(rows[:, 0] == sample_id)
    if len(matches) != 1:
        raise ValueError(f"Expected one row with sample ID {sample_id}; found {len(matches)}")
    reference = rows[matches[0], 1:].copy()
    reconstructed = wav_to_features(wav)
    if reference.shape != reconstructed.shape:
        raise ValueError(f"Feature shapes differ: {reference.shape} vs {reconstructed.shape}")

    model, scaler, checkpoint = load_checkpoint(checkpoint_path)
    metadata = checkpoint["preprocessing"]
    report = {
        "sample_id": sample_id,
        "checkpoint": str(checkpoint_path),
        "training": checkpoint["training"],
    }
    masks = []
    for name, features in (("organizer", reference), ("wav", reconstructed)):
        mask = features == MISSING_VALUE
        masks.append(mask)
        frame_mask = mask.reshape(metadata["mfcc_shape"])
        inputs = features.copy()
        inputs[mask] = metadata["replacement_value"]
        inputs = scaler.transform(inputs.reshape(1, -1)).reshape(
            1, metadata["input_channels"], *metadata["mfcc_shape"],
            order=metadata["reshape_order"],
        )
        with torch.no_grad():
            logits = model(torch.as_tensor(inputs, dtype=getattr(torch, metadata["tensor_dtype"])))
        report[name] = {
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "non_padding_frames": int((~frame_mask).any(axis=1).sum()),
            "padding_count": int(mask.sum()),
            "padding_positions_zero_based": np.flatnonzero(mask).tolist(),
            "first_values": features[:13].tolist(),
            "prediction": decode_class(logits.argmax(1).item(), checkpoint["class_to_label"]),
            "logits": logits[0].tolist(),
        }
    valid = ~masks[0] & ~masks[1]
    difference = np.abs(reference[valid] - reconstructed[valid])
    report["comparison"] = {
        "padding_masks_identical": bool(np.array_equal(*masks)),
        "compared_non_padding_values": int(valid.sum()),
        "max_absolute_difference": float(difference.max()) if difference.size else None,
        "mean_absolute_difference": float(difference.mean()) if difference.size else None,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("X_test.npy"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    args = parser.parse_args()
    print(json.dumps(compare(args.wav, args.dataset, args.sample_id, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
