# Spoken-digit recognition

Spoken-digit recognition using a convolutional neural network and MFCC features,
developed for the 2017 iML challenge at ULiège.

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

Column 0 is an ID, not a feature. Each sample has 32 time steps × 13 MFCC
coefficients. Padding uses `-9999999`, which training replaces with zero.
Label `-1` means other/non-digit; labels `0`–`9` are spoken digits. Model class
`0` corresponds to label `-1`, and classes `1`–`10` correspond to digits `0`–`9`.

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
`0.05`, 101 epochs, batch size 128, four loader workers, and output
`checkpoints/model.pt`. Choose separate checkpoint paths to retain both modes;
existing files at the selected path are overwritten.

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

- `main.py`: CLI, loaders, training, evaluation, and checkpoint saving.
- `preprocessing.py`: loading, splitting/scaling, label mapping, and scaler state.
- `model.py` / `dataset.py`: convolutional network and tensor dataset adapter.
- `exploration/`: class-count and spectrogram plotting utilities.
- `wav2mfcc.py` / `transform_wav.py`: organizer-supplied preprocessing scripts.
- `tests/test_training.py`: focused regression tests.

```sh
python -m unittest discover -s tests -v
```

Tests cover output shapes and negative logits, train-only scaler fitting,
full-data routing, sample-weighted losses, seeded sampling, label mapping,
checkpoint round trips, and imports/help without data access.

Dependencies are unpinned. Optional audio/plotting dependencies are listed in
`requirements-tools.txt`. If torchinfo's summary encounters a Windows terminal
encoding error, set `$env:PYTHONIOENCODING = "utf-8"` in PowerShell.

## Next steps

Prediction, microphone input, submission generation, and speaker-grouped
validation are not implemented. Before single-WAV inference, verify audio-library
compatibility and reproduce the organizers' MFCC extraction, coefficient ordering,
and padding/truncation rules against the supplied arrays. Reuse the checkpoint's
scaler and label mapping without refitting them.

## Author

[@Bortrex](https://github.com/Bortrex)
