"""Regression checks; run with python -m unittest discover -s tests."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import random

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from main import (
    create_loaders, save_checkpoint, train, evaluate, seed_random_generators,
    run_training, main,
)
from torcheval.metrics import MulticlassAccuracy
from model import Net
from preprocessing import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    encode_labels,
    load_training_data,
    prepare_training_data,
    restore_scaler,
)


class TrainingTests(unittest.TestCase):
    def test_import_and_help_do_not_set_up_training(self):
        # A fresh process avoids an already-imported module hiding side effects.
        script = '''
from unittest.mock import patch
import numpy as np
import torch
from model import Net
from dataset import Data
from sklearn.preprocessing import MinMaxScaler

benchmark = torch.backends.cudnn.benchmark
def forbidden(*args, **kwargs):
    raise AssertionError("Application setup during import/help")

with patch.object(np, "load", forbidden), \
     patch.object(torch.cuda, "empty_cache", forbidden), \
     patch.object(torch.cuda, "is_available", forbidden), \
     patch.object(torch.utils.data, "DataLoader", forbidden), \
     patch.object(Net, "__init__", forbidden), \
     patch.object(Data, "__init__", forbidden), \
     patch.object(MinMaxScaler, "fit_transform", forbidden):
    import preprocessing
    import main
    for args in (["--help"], ["train", "--help"]):
        try:
            main.main(args)
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError("Help did not exit")
assert torch.backends.cudnn.benchmark == benchmark
'''
        with tempfile.TemporaryDirectory() as directory:
            # No datasets are available in this subprocess's working directory.
            root = str(Path(__file__).resolve().parents[1])
            script = f"import sys; sys.path.insert(0, {root!r})\n" + script
            result = subprocess.run(
                [sys.executable, "-c", script], cwd=directory,
                capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_label_mapping(self):
        labels = np.arange(-1, 10, dtype=np.int64)
        np.testing.assert_array_equal(encode_labels(labels), labels - labels.min())
        self.assertEqual(CLASS_TO_LABEL[0], -1)
        self.assertEqual(CLASS_TO_LABEL[10], 9)
        for label, index in LABEL_TO_CLASS.items():
            self.assertEqual(CLASS_TO_LABEL[index], label)

    def test_loading_replaces_sentinel_and_preserves_mapping(self):
        inputs = np.arange(22 * 416, dtype=np.float64).reshape(22, 416)
        inputs[0, :13] = -9999999
        inputs[:, -1] = 5  # Include a constant feature in scaler coverage.
        labels = np.tile(np.arange(-1, 10), 2)
        ids = np.arange(len(labels))
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory) / "X_train.npy", np.column_stack((ids, inputs)))
            np.save(Path(directory) / "y_train.npy", np.column_stack((ids, labels)))
            actual, targets = load_training_data(directory)
        inputs[inputs == -9999999] = 0
        np.testing.assert_array_equal(actual, inputs)
        np.testing.assert_array_equal(targets, labels - labels.min())

    def test_checkpoint_round_trip(self):
        model = Net()
        inputs = np.arange(3 * 416, dtype=np.float64).reshape(3, 416)
        inputs[:, -1] = 7
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "model.pt"
            save_checkpoint(path, model, scaler, epochs=1, batch_size=128)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        restored_model = Net()
        restored_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, restored_model.state_dict()[name]))
        restored_scaler = restore_scaler(checkpoint["scaler"])
        # Includes values outside the fitted range; clip=False is preserved.
        probes = np.vstack((inputs, inputs[0] - 100, inputs[-1] + 100))
        np.testing.assert_array_equal(
            scaler.transform(probes), restored_scaler.transform(probes)
        )
        self.assertEqual(checkpoint["class_to_label"], CLASS_TO_LABEL)
        self.assertEqual(checkpoint["label_to_class"], LABEL_TO_CLASS)
        self.assertEqual(checkpoint["preprocessing"]["mfcc_shape"], [32, 13])
        self.assertEqual(checkpoint["training"]["epochs_completed"], 1)
        self.assertEqual(checkpoint["format_version"], 2)
        self.assertEqual(checkpoint["model"]["output_activation"], "identity")
        self.assertEqual(checkpoint["training"]["mode"], "development")

    def test_development_scaler_uses_only_training_samples(self):
        labels = np.tile(np.arange(11), 20)
        inputs = np.arange(220 * 416, dtype=np.float64).reshape(220, 416)
        train_ids, validation_ids = train_test_split(
            np.arange(220), test_size=0.1, random_state=1234, stratify=labels
        )
        # An extreme present only in validation must not affect the fitted scaler.
        inputs[validation_ids[0], 0] = 1e9
        train_x, train_y, val_x, val_y, scaler = prepare_training_data(inputs, labels)
        expected = MinMaxScaler(feature_range=(-1, 1)).fit(inputs[train_ids])
        self.assertEqual(scaler.n_samples_seen_, len(train_ids))
        np.testing.assert_array_equal(scaler.data_max_, expected.data_max_)
        np.testing.assert_array_equal(scaler.data_min_, expected.data_min_)
        np.testing.assert_array_equal(train_x, expected.transform(inputs[train_ids]).reshape(-1, 1, 32, 13))
        np.testing.assert_array_equal(val_x, expected.transform(inputs[validation_ids]).reshape(-1, 1, 32, 13))
        np.testing.assert_array_equal(train_y, labels[train_ids])
        np.testing.assert_array_equal(val_y, labels[validation_ids])
        self.assertGreater(val_x[0, 0, 0, 0], 1)

    def test_full_data_skips_split_and_validation_loader(self):
        inputs = np.arange(220 * 416, dtype=np.float64).reshape(220, 416)
        labels = np.tile(np.arange(11), 20)
        with patch("preprocessing.train_test_split", side_effect=AssertionError("Unexpected split")):
            train_x, train_y, val_x, val_y, scaler = prepare_training_data(
                inputs, labels, full_data=True
            )
        self.assertIsNone(val_x)
        self.assertIsNone(val_y)
        self.assertEqual(scaler.n_samples_seen_, len(labels))
        expected = MinMaxScaler(feature_range=(-1, 1)).fit_transform(inputs)
        np.testing.assert_array_equal(train_x, expected.reshape(-1, 1, 32, 13))
        np.testing.assert_array_equal(train_y, labels)
        loader, validation_loader = create_loaders(train_x, train_y)
        self.assertEqual(len(loader.dataset), len(labels))
        self.assertIsNone(validation_loader)

    def test_full_data_cli_and_orchestration_skip_evaluation(self):
        with patch("main.run_training") as run:
            main(["train", "--full-data", "--epochs", "1"])
            self.assertTrue(run.call_args.args[-1])
        inputs = np.arange(22 * 416, dtype=np.float64).reshape(22, 416)
        labels = np.tile(np.arange(11), 2)
        model = Net()
        with patch("main.load_training_data", return_value=(inputs, labels)), \
             patch("main.create_training_components", return_value=(model, None, None, None)), \
             patch("main.train", return_value=(1.0, torch.tensor(0.5))), \
             patch("main.evaluate", side_effect=AssertionError("Unexpected evaluation")), \
             tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full.pt"
            run_training(epochs=1, checkpoint_path=path, full_data=True)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["training"]["mode"], "full_data")
        self.assertEqual(checkpoint["scaler"]["n_samples_seen_"], 22)

    def test_loaders_and_rngs_are_repeatable(self):
        inputs = np.zeros((220, 1, 32, 13))
        labels = np.tile(np.arange(11), 20)
        first, validation = create_loaders(inputs, labels, inputs[:22], labels[:22])
        second, _ = create_loaders(inputs, labels)
        self.assertEqual(list(first.sampler), list(second.sampler))
        for loader in (first, validation):
            self.assertEqual(loader.batch_size, 128)
            self.assertEqual(loader.num_workers, 4)
            self.assertTrue(loader.pin_memory)
            self.assertFalse(loader.drop_last)
            self.assertIsNotNone(loader.worker_init_fn)
        self.assertIsInstance(first.sampler, torch.utils.data.RandomSampler)
        self.assertIsInstance(validation.sampler, torch.utils.data.SequentialSampler)
        seed_random_generators()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        seed_random_generators()
        self.assertEqual(expected[0], random.random())
        self.assertEqual(expected[1], np.random.rand())
        self.assertTrue(torch.equal(expected[2], torch.rand(3)))
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)

    def test_model_returns_logits_with_batch_dimension(self):
        model = Net()
        with torch.no_grad():
            model.conv1d4.weight.zero_()
            model.conv1d4.bias.fill_(-2)
            for training in (False, True):
                model.train(training)
                for count in (1, 4):
                    logits = model(torch.zeros(count, 1, 32, 13))
                    self.assertEqual(tuple(logits.shape), (count, 11))
                    self.assertTrue(torch.equal(logits, torch.full((count, 11), -2.0)))

    def test_losses_are_weighted_by_sample_count(self):
        # Three samples in batches of two and one, with different batch losses.
        inputs = torch.tensor([[3., 0.], [3., 0.], [0., 3.]])
        labels = torch.zeros(3, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, labels), batch_size=2
        )
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        criterion = torch.nn.CrossEntropyLoss()
        expected = criterion(inputs, labels).item()
        # Zero learning rate isolates aggregation from parameter updates in this test.
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        loss, accuracy = train(model, loader, optimizer, criterion, MulticlassAccuracy(), "cpu")
        self.assertAlmostEqual(loss, expected, places=6)
        self.assertAlmostEqual(accuracy.item(), 2 / 3, places=6)
        loss, accuracy = evaluate(model, loader, criterion, MulticlassAccuracy(), "cpu")
        self.assertAlmostEqual(loss, expected, places=6)
        self.assertAlmostEqual(accuracy.item(), 2 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
