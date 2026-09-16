"""Non-interactive plots for validation and recorded predictions."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch

from preprocessing import CLASS_TO_LABEL

DISPLAY_LABELS = ["other" if label == -1 else str(label) for label in CLASS_TO_LABEL.values()]


def _save(fig, path):
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def save_prediction_plot(logits, label, path):
    """Softmax is for display only; the decision comes from the original logits."""
    scores = torch.softmax(logits.detach().cpu().reshape(-1), dim=0).numpy()
    if scores.shape != (11,):
        raise ValueError("Expected 11 class logits")
    winner = int(logits.argmax().item())
    colors = ["#2878a5" if index == winner else "#b8c7d1" for index in range(11)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(DISPLAY_LABELS, scores, color=colors)
    ax.set(ylim=(0, 1), ylabel="Softmax score", xlabel="Class", title=f"Prediction: {label}")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, path)


def save_spectrogram(wav_path, path):
    """Plot the recorded audio; this representation is never used for inference."""
    import librosa
    import librosa.display
    from scipy.io import wavfile

    sample_rate, signal = wavfile.read(wav_path)
    power = librosa.feature.melspectrogram(
        y=signal.astype(np.float32), sr=sample_rate, n_mels=128, fmax=sample_rate / 2
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    image = librosa.display.specshow(
        librosa.power_to_db(power, ref=np.max), sr=sample_rate,
        x_axis="time", y_axis="mel", fmax=sample_rate / 2, ax=ax,
    )
    ax.set(title="Recorded audio: Mel spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    _save(fig, path)


def save_confusion_matrix(true_labels, predicted_labels, path):
    counts = confusion_matrix(true_labels, predicted_labels, labels=list(CLASS_TO_LABEL))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(counts, display_labels=DISPLAY_LABELS).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set(title="Validation confusion matrix", xlabel="Predicted label", ylabel="True label")
    _save(fig, path)
    return counts


def save_validation_confusion_matrix(model, loader, device,
                                     path="runtime_outputs/confusion_matrix.png"):
    """Evaluate once without updating metrics or leaving the model in another mode."""
    was_training = model.training
    true_labels, predicted_labels = [], []
    try:
        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                predicted_labels.extend(model(inputs.to(device)).argmax(1).cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
    finally:
        model.train(was_training)
    return save_confusion_matrix(true_labels, predicted_labels, path)
