"""WAV layout and checkpoint inference tests without competition recordings."""

from pathlib import Path
import tempfile
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from scipy.io import wavfile

from audio_preprocessing import extract_mfcc, mfcc_to_features, to_mono, wav_to_features
from inference import decode_class, load_checkpoint, predict_wav, prepare_wav_input
from main import main, save_checkpoint
from model import Net
from preprocessing import CLASS_TO_LABEL, MISSING_VALUE


class AudioTests(unittest.TestCase):
    def test_padding_order_and_size(self):
        mfcc = np.arange(13 * 3).reshape(13, 3)
        result = mfcc_to_features(mfcc)
        self.assertEqual(result.shape, (416,))
        np.testing.assert_array_equal(result[:39], mfcc.T.reshape(-1))
        np.testing.assert_array_equal(result[39:], MISSING_VALUE)

    def test_long_sequence_keeps_first_32_frames(self):
        mfcc = np.arange(13 * 40).reshape(13, 40)
        np.testing.assert_array_equal(mfcc_to_features(mfcc), mfcc[:, :32].T.reshape(-1))

    def test_mocked_extraction_reaches_feature_layout(self):
        with patch("audio_preprocessing.extract_mfcc", return_value=np.ones((13, 32))) as extract:
            features = wav_to_features("example.wav")
        extract.assert_called_once_with("example.wav", 13)
        np.testing.assert_array_equal(features, np.ones(416))

    def test_mono_and_extraction_preserve_pcm_scale_and_sample_rate(self):
        signal = np.array([[1000, 3000], [2000, 4000], [3000, 5000]], dtype=np.int16)
        np.testing.assert_array_equal(to_mono(signal), [2000, 3000, 4000])
        np.testing.assert_array_equal(to_mono(signal.T), [2000, 3000, 4000])
        mfcc = Mock(return_value=np.zeros((13, 3)))
        fake_librosa = SimpleNamespace(feature=SimpleNamespace(mfcc=mfcc))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            path.touch()
            with patch.dict("sys.modules", {"librosa": fake_librosa}), \
                 patch.object(wavfile, "read", return_value=(16000, signal)):
                extract_mfcc(path)
        kwargs = mfcc.call_args.kwargs
        self.assertEqual(kwargs["sr"], 16000)
        self.assertEqual(kwargs["n_mfcc"], 13)
        self.assertEqual(kwargs["pad_mode"], "reflect")
        self.assertEqual(kwargs["y"].dtype, np.float32)
        np.testing.assert_array_equal(kwargs["y"], [2000, 3000, 4000])

    def test_missing_and_invalid_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.wav"
            with self.assertRaisesRegex(FileNotFoundError, "WAV file not found"):
                extract_mfcc(path)
            path.write_bytes(b"not audio")
            with patch.dict("sys.modules", {"librosa": SimpleNamespace()}):
                with self.assertRaisesRegex(ValueError, "Cannot read or process WAV"):
                    extract_mfcc(path)
                with patch.object(wavfile, "read", return_value=(16000, np.array([]))):
                    with self.assertRaisesRegex(ValueError, "nonempty"):
                        extract_mfcc(path)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.pt"
        model = Net()
        with torch.no_grad():
            model.conv1d4.weight.zero_()
            model.conv1d4.bias.zero_()
            model.conv1d4.bias[7] = 5
        self.scaler = MinMaxScaler(feature_range=(-1, 1)).fit(
            np.array([np.zeros(416), np.full(416, 100.0)])
        )
        save_checkpoint(self.path, model, self.scaler, 1, 128)

    def test_input_uses_saved_scaler_without_fit(self):
        features = mfcc_to_features(np.full((13, 2), 25.0))
        expected = features.copy()
        expected[expected == MISSING_VALUE] = 0
        expected = self.scaler.transform(expected.reshape(1, -1)).reshape(1, 1, 32, 13)
        with patch.object(MinMaxScaler, "fit", side_effect=AssertionError("Refitted scaler")), \
             patch.object(MinMaxScaler, "fit_transform", side_effect=AssertionError("Refitted scaler")), \
             patch("inference.wav_to_features", return_value=features):
            model, scaler, checkpoint = load_checkpoint(self.path)
            inputs = prepare_wav_input("mock.wav", scaler, checkpoint["preprocessing"])
        self.assertEqual(tuple(inputs.shape), (1, 1, 32, 13))
        self.assertEqual(inputs.dtype, torch.float32)
        np.testing.assert_array_equal(inputs.numpy(), expected.astype(np.float32))
        self.assertFalse(model.training)
        self.assertEqual(next(model.parameters()).device.type, "cpu")
        with torch.no_grad():
            self.assertEqual(tuple(model(inputs).shape), (1, 11))

    def test_decoding_all_classes(self):
        self.assertEqual(decode_class(0, CLASS_TO_LABEL), "other")
        for index in range(1, 11):
            self.assertEqual(decode_class(index, CLASS_TO_LABEL), str(index - 1))

    def test_prediction_decodes_logits_without_gradients(self):
        wav = Path(self.directory.name) / "mock.wav"
        wav.touch()
        original_forward = Net.forward
        def checked_forward(model, inputs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(model.training)
            return original_forward(model, inputs)
        with patch("inference.wav_to_features", return_value=np.zeros(416)), \
             patch.object(Net, "forward", checked_forward):
            self.assertEqual(predict_wav(wav, self.path), "6")

    def test_cli_output_and_file_errors(self):
        output = io.StringIO()
        with patch("inference.predict_wav", return_value="other"), contextlib.redirect_stdout(output):
            main(["predict", "mock.wav"])
        self.assertEqual(output.getvalue(), "Prediction: other\n")
        wav = Path(self.directory.name) / "missing.wav"
        for args, message in [
            (["predict", str(wav)], "WAV file not found"),
            (["predict", str(self.path), "--checkpoint", str(wav)], "Checkpoint not found"),
        ]:
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertIn(message, errors.getvalue())

    def test_logits_are_returned_from_a_single_forward_pass(self):
        wav = Path(self.directory.name) / "mock.wav"
        wav.touch()
        model, scaler, checkpoint = load_checkpoint(self.path)
        with patch("inference.load_checkpoint", return_value=(model, scaler, checkpoint)), \
             patch("inference.wav_to_features", return_value=np.zeros(416)), \
             patch.object(model, "forward", wraps=model.forward) as forward:
            label, logits = predict_wav(wav, self.path, return_logits=True)
        forward.assert_called_once()
        self.assertEqual(label, "6")
        self.assertEqual(tuple(logits.shape), (11,))
        self.assertEqual(logits.argmax().item(), 7)
        self.assertFalse(logits.requires_grad)

    def test_missing_corrupt_and_incompatible_checkpoints(self):
        with self.assertRaisesRegex(FileNotFoundError, "Checkpoint not found"):
            load_checkpoint(self.path.with_name("missing.pt"))
        checkpoint = torch.load(self.path, weights_only=True)
        checkpoint["format_version"] = 1
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "only checkpoint format version 2"):
            load_checkpoint(self.path)
        checkpoint["format_version"] = 2
        checkpoint["preprocessing"]["mfcc_shape"] = [13, 32]
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "preprocessing metadata"):
            load_checkpoint(self.path)
        self.path.write_bytes(b"not a checkpoint")
        with self.assertRaisesRegex(ValueError, "unreadable checkpoint"):
            load_checkpoint(self.path)


if __name__ == "__main__":
    unittest.main()
