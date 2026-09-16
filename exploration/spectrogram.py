"""Explore one competition recording; run from the repository root."""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load("wav_train/train-9_7_1.wav")
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
fig = plt.figure(figsize=(10, 4))
img = librosa.display.specshow(
    librosa.power_to_db(S, ref=np.max),
    y_axis="mel", x_axis="time", fmax=8_000,
)

fig.colorbar(img)

plt.title("Mel spectrogram")
plt.tight_layout()
plt.show()