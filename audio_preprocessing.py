"""Convert a WAV recording to the organizer's time-major MFCC layout."""

from pathlib import Path

import numpy as np

from preprocessing import MFCC_SHAPE, MISSING_VALUE


def to_mono(signal):
    """Match the organizer's squeeze and channel-averaging behavior."""
    signal = np.squeeze(signal)
    if signal.ndim <= 1:
        return np.atleast_1d(signal)
    axis = 0 if signal.shape[0] == 2 else 1
    return np.mean(signal, axis=axis)


def extract_mfcc(path, n_coefficients=MFCC_SHAPE[1]):
    """Keep PCM amplitude and use the organizer's reflected waveform boundaries."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")
    try:
        import librosa
        from scipy.io import wavfile
    except ImportError as error:
        raise RuntimeError("WAV prediction requires librosa and scipy; install requirements-tools.txt") from error

    try:
        sample_rate, signal = wavfile.read(path)
        signal = to_mono(signal).astype(np.float32)
        if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
            raise ValueError("expected nonempty, finite mono/stereo audio")
        # librosa 0.4.3 used reflection; modern defaults zero-pad STFT boundaries.
        # This is separate from padding missing MFCC frames with the sentinel.
        return librosa.feature.mfcc(
            y=signal, sr=sample_rate, n_mfcc=n_coefficients, pad_mode="reflect"
        )
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"Cannot read or process WAV '{path}': {error}") from error


def mfcc_to_features(mfcc, shape=MFCC_SHAPE, padding=MISSING_VALUE):
    """Pad short sequences; truncate long ones to the first 32 frames by default."""
    frames, coefficients = shape
    mfcc = np.asarray(mfcc)
    if mfcc.ndim != 2 or mfcc.shape[0] != coefficients or mfcc.shape[1] == 0:
        raise ValueError(f"Expected {coefficients} MFCC coefficients and at least one frame")
    if not np.isfinite(mfcc).all():
        raise ValueError("MFCC values must be finite")
    features = np.full((frames, coefficients), padding, dtype=np.float64)
    count = min(frames, mfcc.shape[1])
    features[:count] = mfcc[:, :count].T
    return features.reshape(-1)


def wav_to_features(path, shape=MFCC_SHAPE, padding=MISSING_VALUE):
    """Return a flat feature row without an ID column or fitted scaling."""
    return mfcc_to_features(extract_mfcc(path, shape[1]), shape, padding)
