"""
02_mixed_precision_training.py

Compare FP32 training with automatic mixed precision (AMP).

Experiments:
- FP32 baseline
- FP16 automatic mixed precision with gradient scaling
- BF16 automatic mixed precision where supported
- Training time, throughput, accuracy, and peak GPU memory

"""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 1e-3

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/mixed_precision")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_WORKERS = 0


# ---------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class MNISTCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def create_dataloaders(
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------
# Precision configuration
# ---------------------------------------------------------------------

def get_precision_configs(
    device: torch.device,
) -> list[tuple[str, torch.dtype | None]]:
    configs: list[tuple[str, torch.dtype | None]] = [
        ("FP32", None)
    ]

    if device.type == "cuda":
        configs.append(("FP16 AMP", torch.float16))

        if torch.cuda.is_bf16_supported():
            configs.append(("BF16 AMP", torch.bfloat16))

    return configs


# ---------------------------------------------------------------------
# Timing and memory
# ---------------------------------------------------------------------

def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None

    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[float, float]:
    model.eval()

    total_correct = 0
    total_examples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def run_experiment(
    name: str,
    initial_state: dict[str, torch.Tensor],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float | str | None]:
    print(f"\n{'=' * 65}")
    print(f"Experiment: {name}")
    print(f"{'=' * 65}")

    model = MNISTCNN().to(device)
    model.load_state_dict(initial_state)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # GradScaler is needed for FP16 AMP on CUDA to help prevent
    # small gradients from underflowing. It is not needed for BF16.
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(device.type == "cuda" and amp_dtype == torch.float16),
    )

    reset_peak_memory(device)
    synchronize(device)
    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_dtype,
        )

        synchronize(device)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2%}"
        )

    synchronize(device)
    training_seconds = time.perf_counter() - start_time

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        device,
        amp_dtype,
    )

    total_examples = len(train_loader.dataset) * EPOCHS
    throughput = total_examples / training_seconds

    peak_memory_mb = get_peak_memory_mb(device)

    checkpoint_path = OUTPUT_DIR / (
        name.lower().replace(" ", "_") + "_model.pt"
    )
    torch.save(model.state_dict(), checkpoint_path)

    result: dict[str, float | str | None] = {
        "name": name,
        "training_seconds": training_seconds,
        "examples_per_second": throughput,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "peak_memory_mb": peak_memory_mb,
    }

    print(f"Training Time: {training_seconds:.2f} seconds")
    print(f"Training Throughput: {throughput:.2f} examples/sec")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")

    if peak_memory_mb is not None:
        print(f"Peak GPU Memory: {peak_memory_mb:.2f} MiB")

    print(f"Checkpoint: {checkpoint_path}")

    return result


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

def print_comparison(
    results: list[dict[str, float | str | None]],
) -> None:
    print("\n" + "=" * 100)
    print("MIXED PRECISION COMPARISON")
    print("=" * 100)

    header = (
        f"{'Configuration':<14}"
        f"{'Time (s)':>12}"
        f"{'Examples/s':>14}"
        f"{'Test Accuracy':>16}"
        f"{'Peak Memory (MiB)':>22}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        memory = result["peak_memory_mb"]
        memory_text = f"{memory:.2f}" if memory is not None else "N/A"

        print(
            f"{str(result['name']):<14}"
            f"{float(result['training_seconds']):>12.2f}"
            f"{float(result['examples_per_second']):>14.2f}"
            f"{float(result['test_accuracy']):>15.2%}"
            f"{memory_text:>22}"
        )

    baseline = results[0]
    baseline_time = float(baseline["training_seconds"])

    print("\nRelative to FP32 baseline:")

    for result in results[1:]:
        optimized_time = float(result["training_seconds"])
        speedup = baseline_time / optimized_time

        print(
            f"{result['name']}: "
            f"{speedup:.2f}x speedup"
        )


def main() -> None:
    set_seed()
    device = get_device()

    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(
            "CUDA is unavailable. Running FP32 on CPU; "
            "mixed precision GPU comparisons will be skipped."
        )

    train_loader, test_loader = create_dataloaders(device)

    # Initialize one model and reuse its initial weights for each run.
    # This improves comparability between precision configurations.
    reference_model = MNISTCNN()
    initial_state = copy.deepcopy(reference_model.state_dict())

    results = []

    for name, amp_dtype in get_precision_configs(device):
        set_seed()
        result = run_experiment(
            name=name,
            initial_state=initial_state,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            amp_dtype=amp_dtype,
        )
        results.append(result)

    print_comparison(results)


if __name__ == "__main__":
    main()
