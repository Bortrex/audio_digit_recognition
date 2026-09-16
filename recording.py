"""Capture one microphone recording and reuse WAV prediction."""

from pathlib import Path
import time

import numpy as np

from inference import predict_wav
from plots import save_prediction_plot, save_spectrogram

SAMPLE_RATE = 16000
FRAMES = 16000


def record_and_predict(checkpoint_path="checkpoints/model.pt", output_dir="recordings"):
    from scipy.io import wavfile
    try:
        import sounddevice as sd
    except (ImportError, OSError) as error:
        raise RuntimeError("Microphone capture requires sounddevice/PortAudio; install requirements-tools.txt") from error
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        sd.check_input_settings(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    except Exception as error:
        raise RuntimeError(f"Microphone unavailable: {error}. Check the default input device and microphone permissions.") from error

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "latest.wav"
    spectrogram_path = output_dir / "latest_spectrogram.png"
    prediction_path = output_dir / "latest_prediction.png"
    print("Ready...", flush=True)
    for number in (3, 2, 1):
        print(number, flush=True)
        time.sleep(1)
    print("Speak!", flush=True)
    try:
        audio = sd.rec(FRAMES, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        print("Recording...", flush=True)
        sd.wait()
    except Exception as error:
        raise RuntimeError(f"Microphone recording failed: {error}. Check the input device and permissions.") from error
    if audio.shape != (FRAMES, 1) or audio.dtype != np.int16:
        raise ValueError("Expected one second of mono int16 microphone audio")
    # Remove previous plots so a failed prediction cannot leave stale scores for new audio.
    spectrogram_path.unlink(missing_ok=True)
    prediction_path.unlink(missing_ok=True)
    wavfile.write(wav_path, SAMPLE_RATE, audio[:, 0])
    label, logits = predict_wav(wav_path, checkpoint_path, return_logits=True)
    save_spectrogram(wav_path, spectrogram_path)
    save_prediction_plot(logits, label, prediction_path)
    print(f"Prediction: {label}")
    print(f"Recording: {wav_path}")
    print(f"Spectrogram: {spectrogram_path}")
    print(f"Prediction plot: {prediction_path}")
    return label
