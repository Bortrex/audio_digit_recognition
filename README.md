# Spoken-digit recognition

Spoken-digit recognition using a convolutional neural network and MFCC features,
developed for the 2017 iML challenge at ULiÃƒÂ¨ge.

## Data

The challenge organizers supplied `X_train.npy`, `y_train.npy`, and `X_test.npy`.
They generated the MFCC representation using the supplied `wav2mfcc.py` and
`transform_wav.py` scripts. Normal training uses these precomputed arrays;
regenerating MFCCs is not required. The scripts are kept unchanged.
Raw WAV recordings and NumPy datasets are competition data and are not
redistributed in this repository.

| File | Shape | Contents |
| --- | --- | --- |
| `X_train.npy` | `(14721, 417)` | Sample ID followed by 416 MFCC features |
| `X_test.npy` | `(22568, 417)` | Sample ID followed by 416 MFCC features |
| `y_train.npy` | `(14721, 2)` | Sample ID and label |

Column 0 is an ID, not a feature. Each sample has 32 time steps Ãƒâ€” 13 MFCC
coefficients. Padding uses `-9999999`, which training replaces with zero.
Label `-1` means other/non-digit; labels `0`Ã¢â‚¬â€œ`9` are spoken digits. Model class
`0` corresponds to label `-1`, and classes `1`Ã¢â‚¬â€œ`10` correspond to digits `0`Ã¢â‚¬â€œ`9`.

The competition test set contains unseen speakers. Development validation uses
a random stratified split; speaker-grouped validation is not implemented.

## Training

Install `requirements.txt` in your Python environment and place `X_train.npy`
and `y_train.npy` in the repository root, or specify `--data-dir`.

```sh
python main.py train
python main.py train --full-data
python main.py train --epochs 1 --checkpoint checkpoints/development-smoke.pt
python main.py train --full-data --epochs 1 --checkpoint checkpoints/full-data-smoke.pt
python main.py train --epochs 6 --num-workers 0
python main.py train --help
```

- **Development mode:** split labelled samples 90/10 with stratification, fit
  `MinMaxScaler(feature_range=(-1, 1))` only on training features, and transform
  both subsets with that scaler. Validation values can lie outside `[-1, 1]`;
  clipping is disabled.
- **Full-data mode:** skip validation, fit the scaler on all labelled features,
  and train on every labelled sample.

Both modes reshape features to `(N, 1, 32, 13)` and save the final model and fitted
scaler. Training never reads `X_test.npy`. Defaults are SGD with learning rate
`0.05`, 101 epochs, batch size 128, zero loader workers, and output
`checkpoints/model.pt`. Choose separate checkpoint paths to retain both modes;
existing files at the selected path are overwritten. `--num-workers` controls both
loaders (default: 0). Each run reports wall-clock time for the epoch training and
validation loop, excluding setup and checkpoint saving; CUDA is synchronized at
the timer boundaries. Compare worker counts on your own machine before tuning.

The network returns unrestricted logits with shape `(N, 11)`, including `N=1`.
Training uses cross-entropy and reports sample-weighted mean loss and accuracy
every five epochs. Development mode also reports validation loss and accuracy.

Python, NumPy, PyTorch, and loader generators use seed 1234. Workers seed Python
and NumPy from their PyTorch worker seed. cuDNN benchmarking is disabled and
its deterministic setting is enabled. Results are not guaranteed to be bit-for-bit
identical across PyTorch, CUDA, or hardware environments.

Imports and help commands do not load datasets or start setup. When calling
`run_training(...)` from another script, use an `if __name__ == "__main__":`
guard for Windows multiprocessing.

## WAV prediction

```sh
python main.py predict path/to/audio.wav
python main.py predict test-17744_1308_2.wav --checkpoint checkpoints/model.pt
python main.py predict --help
```

Install `requirements-tools.txt` for librosa and SciPy. Prediction runs on CPU and
prints a digit or `other`. It requires a format-version-2 checkpoint.

WAV loading uses SciPy, preserves the recording's sample rate and PCM amplitude,
averages stereo channels as in the organizer script, and calls librosa for 13
MFCC coefficients with reflected waveform boundaries (`pad_mode="reflect"`). Frames are flattened in time-major order. Short recordings
are padded with `-9999999`; recordings over 32 frames keep only the first 32.
Padding is then replaced with zero, the checkpoint scaler is applied without
refitting, and features become a float32 `(1, 1, 32, 13)` tensor. There is no
resampling, silence detection, or segmentation. Librosa 0.4.3 used reflection padding; [librosa 0.9 changed the default to zero
padding](https://librosa.org/doc/0.11.0/changelog.html#v0-9-0). Explicit reflection
matches local organizer row ID 17744 to a maximum absolute error of `0.0001171`
(mean `0.000007592`), compared with `13.3497` (mean `0.296817`) under zero padding.
This is a close numerical match on that recording, not bitwise equality or a
claim covering every competition sample.

To compare local-only audio and arrays without adding them to the test suite:

```sh
python -m exploration.compare_mfcc test-17744_1308_2.wav --sample-id 17744
```

## Checkpoints

Format version 2 stores tensors and plain metadata:

- `model_state_dict` and model output semantics (unrestricted logits).
- Fitted scaler attributes, preserving their numerical precision and settings.
- Both class/label mappings and input preprocessing metadata.
- Training mode, seed, completed epochs, and batch size.

```python
import torch
from model import Net
from preprocessing import restore_scaler

checkpoint = torch.load("checkpoints/model.pt", map_location="cpu", weights_only=True)
assert checkpoint["format_version"] == 2
model = Net()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
scaler = restore_scaler(checkpoint["scaler"])
```

Optimizer and RNG states are not saved, so checkpoints do not support exact
training resumption. Earlier version-1 checkpoints used a final ReLU. Their
parameter shapes still match, but loading them into this model changes output
behavior; retrain to use the corrected workflow.

## Files and checks

- `main.py`: CLI, loaders, timed training/evaluation, and checkpoint saving.
- `audio_preprocessing.py` / `inference.py`: WAV features and checkpoint prediction.
- `preprocessing.py`: loading, splitting/scaling, label mapping, and scaler state.
- `model.py` / `dataset.py`: convolutional network and tensor dataset adapter.
- `exploration/`: class-count and spectrogram plotting utilities.
- `wav2mfcc.py` / `transform_wav.py`: organizer-supplied preprocessing scripts.
- `tests/`: training and WAV inference regression tests.

```sh
python -m unittest discover -s tests -v
```

Tests cover output shapes and negative logits, train-only scaler fitting,
full-data routing, sample-weighted losses, seeded sampling, label mapping,
checkpoint round trips, WAV feature layout and decoding, worker configuration,
timer placement, and imports/help without data access.

Dependencies are unpinned. Optional audio/plotting dependencies are listed in
`requirements-tools.txt`. If torchinfo's summary encounters a Windows terminal
encoding error, set `$env:PYTHONIOENCODING = "utf-8"` in PowerShell.

## Next steps

Microphone input, submission generation, and speaker-grouped validation are not
implemented. Microphone support still needs capture/device handling and a defined
recording length. Check additional recordings when validating broader MFCC compatibility.
A separate performance pass can compare worker counts over representative runs.

## Author

[@Bortrex](https://github.com/Bortrex)
