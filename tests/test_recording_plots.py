"""Recording and plot tests use synthetic inputs and temporary outputs only."""

import contextlib
import io
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from scipy.io import wavfile
import torch

from plots import (DISPLAY_LABELS, save_prediction_plot, save_confusion_matrix,
                   save_validation_confusion_matrix)
from recording import record_and_predict


class RecordingPlotTests(unittest.TestCase):
    def test_recording_overwrites_three_files_and_uses_wav_pipeline(self):
        sd = Mock()
        sd.rec.return_value = np.zeros((16000, 1), dtype=np.int16)
        librosa = ModuleType("librosa")
        display = ModuleType("librosa.display")
        librosa.display = display
        librosa.feature = Mock()
        librosa.feature.melspectrogram.return_value = np.ones((128, 32))
        librosa.power_to_db = Mock(return_value=np.zeros((128, 32)))
        display.specshow = lambda values, ax, **kwargs: ax.imshow(values, aspect="auto")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.touch()
            output = Path(directory) / "recordings"
            logits = torch.arange(11, dtype=torch.float32)
            with patch.dict("sys.modules", {"sounddevice": sd, "librosa": librosa,
                                            "librosa.display": display}), \
                 patch("recording.time.sleep"), \
                 patch("recording.predict_wav", return_value=("9", logits)) as predict, \
                 contextlib.redirect_stdout(io.StringIO()):
                for value in (10, 20):
                    sd.rec.return_value.fill(value)
                    record_and_predict(checkpoint, output)
                    self.assertEqual(sorted(p.name for p in output.iterdir()),
                                     ["latest.wav", "latest_prediction.png", "latest_spectrogram.png"])
                    rate, audio = wavfile.read(output / "latest.wav")
                    self.assertEqual(rate, 16000)
                    self.assertEqual(audio.shape, (16000,))
                    self.assertEqual(audio.dtype, np.int16)
                    self.assertTrue((audio == value).all())
                    for name in ("latest_prediction.png", "latest_spectrogram.png"):
                        self.assertTrue((output / name).read_bytes().startswith(b"\x89PNG"))
                self.assertEqual(predict.call_count, 2)
                predict.assert_called_with(output / "latest.wav", checkpoint, return_logits=True)
            sd.rec.assert_called_with(16000, samplerate=16000, channels=1, dtype="int16")
        self.assertEqual(plt.get_fignums(), [])

    def test_microphone_failure_is_actionable(self):
        sd = Mock()
        sd.check_input_settings.side_effect = RuntimeError("No input device")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.touch()
            with patch.dict("sys.modules", {"sounddevice": sd}):
                with self.assertRaisesRegex(RuntimeError, "Microphone unavailable.*permissions"):
                    record_and_predict(checkpoint, Path(directory) / "recordings")
        sd.rec.assert_not_called()

    def test_prediction_plot_has_ordered_softmax_scores(self):
        logits = torch.arange(11, dtype=torch.float32)
        captured = {}
        original = Axes.bar
        def capture(ax, labels, scores, **kwargs):
            captured.update(labels=list(labels), scores=scores.copy(), colors=kwargs["color"])
            return original(ax, labels, scores, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(Axes, "bar", capture):
            path = Path(directory) / "prediction.png"
            save_prediction_plot(logits, "9", path)
            self.assertTrue(path.is_file())
        self.assertEqual(captured["labels"], ["other"] + list(map(str, range(10))))
        np.testing.assert_allclose(captured["scores"], torch.softmax(logits, 0).numpy())
        self.assertNotEqual(captured["colors"][-1], captured["colors"][0])
        self.assertEqual(plt.get_fignums(), [])

    def test_confusion_matrix_includes_all_classes_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_outputs" / "confusion_matrix.png"
            counts = save_confusion_matrix([0, 0, 1, 10], [0, 1, 1, 10], path)
            self.assertTrue(path.is_file())
        self.assertEqual(counts.shape, (11, 11))
        self.assertEqual(counts.sum(), 4)
        self.assertEqual(counts[0, 1], 1)
        self.assertEqual(DISPLAY_LABELS, ["other"] + list(map(str, range(10))))
        self.assertEqual(plt.get_fignums(), [])

    def test_validation_plot_uses_eval_no_grad_and_restores_mode(self):
        class CheckedModel(torch.nn.Module):
            def forward(inner, inputs):
                self.assertFalse(inner.training)
                self.assertFalse(torch.is_grad_enabled())
                return inputs
        model = CheckedModel()
        model.train()
        inputs = torch.eye(11)[:3]
        loader = [(inputs, torch.tensor([0, 1, 2]))]
        with tempfile.TemporaryDirectory() as directory:
            counts = save_validation_confusion_matrix(model, loader, "cpu", Path(directory) / "cm.png")
        self.assertTrue(model.training)
        self.assertEqual(counts.trace(), 3)


if __name__ == "__main__":
    unittest.main()
