# Spoken-digit recognition

A convolutional neural network for spoken digits, developed for the 2017 iML
challenge at ULiège. The model combines 2D and 1D convolutions to classify
**digits 0-9 and other/non-digit** from 13 MFCC coefficients over 32 time steps.

## Install

```sh
python -m pip install -r requirements.txt
# WAV and microphone prediction, plus audio tools:
python -m pip install -r requirements-tools.txt
```

Microphone capture uses sounddevice and the system's default input device.
Allow microphone access in your operating system; PortAudio may need to be
installed separately on some platforms.

## Data

The organizers supplied `X_train.npy` (14721 x 417), `X_test.npy` (22568 x 417),
and `y_train.npy` (sample ID and label). Feature column 0 is the sample ID;
the remaining 416 values are 32 time steps x 13 MFCC coefficients. Label `-1`
means other; labels `0` through `9` are digits. Padding (`-9999999`) becomes zero.

Training uses these arrays directly. The supplied `wav2mfcc.py` and
`transform_wav.py` generated the MFCC representation and remain unchanged.
Competition WAVs and datasets are not redistributed.

## Train

Place the labelled arrays in the repository root, or use `--data-dir`.

```sh
python main.py train
python main.py train --full-data
```

Development training uses a stratified 90/10 split and fits the scaler only on
training samples. Full-data training uses all labelled samples without validation.
Both save the model and fitted scaler to `checkpoints/model.pt`.

Defaults: SGD, learning rate 0.05, 101 epochs, batch size 128, and zero loader
workers. Options include `--epochs`, `--batch-size`, `--num-workers`, and
`--checkpoint`. Use separate checkpoint paths to keep different runs. Training
reports epoch-loop time, loss, and accuracy.

## Predict a WAV

```sh
python main.py predict path/to/audio.wav
```

Use `--checkpoint` to select another trained model. Prediction runs on CPU and
prints a digit or `other`. WAV preprocessing retains the sample rate and PCM
amplitude, converts stereo to mono, and explicitly uses the boundary-padding
behavior required to reproduce the organizers' MFCC representation. Short inputs
are padded; inputs over 32 MFCC frames are truncated. The saved scaler is reused
without refitting.

## Predict from the microphone

```sh
python main.py record
```

After the countdown and **Speak!**, capture lasts one second at 16 kHz, mono,
signed 16-bit PCM. `--checkpoint` selects a different trained model. Each call
overwrites the same three local files:

- `recordings/latest.wav`
- `recordings/latest_spectrogram.png`
- `recordings/latest_prediction.png`

These let you listen to the capture, inspect its time-frequency representation,
and compare softmax scores across all 11 classes. Scores are not calibrated
confidence estimates; the decision uses the model's logits.

## Results and limitations

Development training writes `runtime_outputs/confusion_matrix.png` once after
the final epoch, using validation-set counts. Full-data training does not create
one. Both `recordings/` and `runtime_outputs/` are ignored by Git.
Representative figures can later be selected manually and copied to `docs/images/`
for documentation; runtime outputs are never copied there automatically.

Microphone and unseen-speaker accuracy depend on the trained checkpoint and
recording conditions. The competition test set used unseen speakers, while the
validation split here is random and stratified. There is no continuous listening,
silence detection, speech segmentation, or submission generation.

## Structure and tests

- `main.py`: CLI, training, evaluation, and checkpoint saving.
- `model.py`, `dataset.py`, `preprocessing.py`: network and array preparation.
- `audio_preprocessing.py`, `inference.py`: shared WAV prediction pipeline.
- `recording.py`, `plots.py`: microphone capture and generated figures.
- `exploration/`: optional visualization and local diagnostics.
- `tests/`: regression tests using synthetic inputs and mocked capture.

```sh
python main.py --help
python -m unittest discover -s tests -v
```

## Author

[@Bortrex](https://github.com/Bortrex)
