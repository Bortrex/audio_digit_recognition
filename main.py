"""Command-line interface for training, WAV prediction, and microphone prediction."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from torchinfo import summary

from dataset import Data
from model import Net
from preprocessing import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    MFCC_SHAPE,
    MISSING_VALUE,
    load_training_data,
    prepare_training_data,
    scaler_state,
)

SEED = 1234
EPOCHS = 101
BATCH_SIZE = 128
NUM_WORKERS = 0


def seed_random_generators(seed=SEED):
    """Seed training RNGs and request reproducible cuDNN algorithm selection."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """Seed NumPy and Python from the worker seed assigned by PyTorch."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_loaders(train_inputs, train_labels, validation_inputs=None,
                   validation_labels=None, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Build seeded loaders; full-data mode has no validation loader."""
    train_loader = DataLoader(
        Data(train_inputs, train_labels),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(SEED), worker_init_fn=seed_worker,
    )
    validation_loader = None
    if validation_inputs is not None:
        validation_loader = DataLoader(
            Data(validation_inputs, validation_labels),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
            generator=torch.Generator().manual_seed(SEED), worker_init_fn=seed_worker,
        )
    return train_loader, validation_loader


def create_training_components(device):
    """Create the model, loss, optimizer, and accuracy metric."""
    metric = MulticlassAccuracy()
    model = Net().to(device)
    print("\nModel summary:")
    summary(model, input_size=(128, 1, *MFCC_SHAPE))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    return model, criterion, optimizer, metric


def train(model, train_loader, optimizer, criterion, metric, device):
    """Return sample-weighted mean loss and accuracy for one training epoch."""
    model.train()
    metric.reset()
    loss_sum = 0.0
    sample_count = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        metric.update(outputs, labels)
        loss_sum += loss.item() * labels.size(0)
        sample_count += labels.size(0)

    return loss_sum / sample_count, metric.compute()


def evaluate(model, validation_loader, criterion, metric, device):
    """Return sample-weighted mean loss and accuracy on validation samples."""
    model.eval()
    metric.reset()
    loss_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            metric.update(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            sample_count += labels.size(0)

    return loss_sum / sample_count, metric.compute()


def save_checkpoint(path, model, scaler, epochs, batch_size, full_data=False):
    """Save an inference checkpoint; optimizer/resume state is intentionally absent."""
    checkpoint = {
        "format_version": 2,
        "model": {"output_activation": "identity", "output_shape": ["batch_size", 11]},
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "scaler": scaler_state(scaler),
        "class_to_label": dict(CLASS_TO_LABEL),
        "label_to_class": dict(LABEL_TO_CLASS),
        "preprocessing": {
            "mfcc_shape": list(MFCC_SHAPE),
            "input_channels": 1,
            "id_column": 0,
            "missing_value": MISSING_VALUE,
            "replacement_value": 0,
            "reshape_order": "C",
            "tensor_dtype": "float32",
            "scaling_before_tensor_conversion": True,
        },
        "training": {
            "epochs_completed": epochs, "batch_size": batch_size, "seed": SEED,
            "mode": "full_data" if full_data else "development",
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def run_training(data_dir=".", epochs=EPOCHS, batch_size=BATCH_SIZE,
                 checkpoint_path="checkpoints/model.pt", full_data=False, num_workers=NUM_WORKERS):
    """Run setup and training explicitly, then persist the final model and scaler."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")

    torch.cuda.empty_cache()
    seed_random_generators()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs, labels = load_training_data(data_dir)
    train_inputs, train_labels, validation_inputs, validation_labels, scaler = (
        prepare_training_data(inputs, labels, full_data=full_data, seed=SEED)
    )
    train_loader, validation_loader = create_loaders(
        train_inputs, train_labels, validation_inputs, validation_labels, batch_size, num_workers
    )
    print(f"Training samples: {len(train_labels)}")
    if validation_labels is not None:
        print(f"Validation samples: {len(validation_labels)}")
    model, criterion, optimizer, metric = create_training_components(device)

    print("\nTraining...")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for ep in range(epochs):
        loss, metric_train = train(
            model, train_loader, optimizer, criterion, metric, device
        )
        if validation_loader is not None:
            validation_loss, metric_validation = evaluate(
                model, validation_loader, criterion, metric, device
            )
        if ep % 5 == 0:
            print(f"Epoch {ep}, Loss: {loss:.4f}, Accuracy: {metric_train.item():.4f}")
            if validation_loader is not None:
                print(f"\tValidation loss: {validation_loss:.4f}, Accuracy: {metric_validation.item():.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    minutes, seconds = divmod(elapsed, 60)
    print(f"Training completed in {int(minutes)}m {seconds:.1f}s ({elapsed / epochs:.2f}s/epoch)")

    save_checkpoint(checkpoint_path, model, scaler, epochs, batch_size, full_data)
    print(f"Checkpoint saved to {checkpoint_path}")
    if validation_loader is not None:
        from plots import save_validation_confusion_matrix

        save_validation_confusion_matrix(model, validation_loader, device)
        print("Confusion matrix: runtime_outputs/confusion_matrix.png")
    return model, scaler


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train from competition NumPy arrays")
    train_parser.add_argument("--data-dir", type=Path, default=Path("."),
                              help="Directory containing X_train.npy and y_train.npy (default: .)")
    train_parser.add_argument("--epochs", type=positive_int, default=EPOCHS,
                              help="Training epochs (default: 101)")
    train_parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE,
                              help="Loader batch size (default: 128)")
    train_parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"),
                              help="Output checkpoint, overwritten on success (default: checkpoints/model.pt)")
    train_parser.add_argument("--full-data", action="store_true",
                              help="Train on all labelled samples without validation")
    train_parser.add_argument("--num-workers", type=nonnegative_int, default=NUM_WORKERS,
                              help="DataLoader workers (default: 0)")
    predict_parser = commands.add_parser("predict", help="Classify one WAV recording")
    predict_parser.add_argument("wav", type=Path, help="WAV file to classify")
    predict_parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"),
                                help="Trained checkpoint (default: checkpoints/model.pt)")
    record_parser = commands.add_parser("record", help="Record one second and classify it")
    record_parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"),
                               help="Trained checkpoint (default: checkpoints/model.pt)")
    args = parser.parse_args(argv)
    if args.command == "train":
        run_training(args.data_dir, args.epochs, args.batch_size, args.checkpoint,
                     args.full_data, num_workers=args.num_workers)
    elif args.command == "predict":
        from inference import predict_wav

        try:
            label = predict_wav(args.wav, args.checkpoint)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        print(f"Prediction: {label}")
    elif args.command == "record":
        from recording import record_and_predict

        try:
            record_and_predict(args.checkpoint)
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
