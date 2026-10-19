"""
04_efficient_training.py

Benchmark data loading and compiled execution.

Experiments:
1. Baseline eager execution.
2. Eager execution with optimized data loading.
3. Compiled execution where supported.

Run:
    uv run python "04-DL-Advanced/06. Advanced Training and GPU Optimization/04_efficient_training.py"
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

NUM_WORKERS = 2
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/efficient_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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

def create_datasets():
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

    return train_dataset, test_dataset


def create_loader(
    dataset,
    device: torch.device,
    optimized: bool,
    shuffle: bool,
) -> DataLoader:
    workers = NUM_WORKERS if optimized else 0

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=optimized and device.type == "cuda",
        persistent_workers=optimized and workers > 0,
    )


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------

def run_experiment(
    name: str,
    initial_state: dict[str, torch.Tensor],
    train_dataset,
    test_dataset,
    device: torch.device,
    optimized_loader: bool,
    use_compile: bool,
) -> dict[str, float | str | None]:
    print(f"\n{'=' * 72}")
    print(f"Experiment: {name}")
    print(f"{'=' * 72}")

    train_loader = create_loader(
        train_dataset,
        device,
        optimized=optimized_loader,
        shuffle=True,
    )
    test_loader = create_loader(
        test_dataset,
        device,
        optimized=optimized_loader,
        shuffle=False,
    )

    model = MNISTCNN().to(device)
    model.load_state_dict(initial_state)

    if use_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "torch.compile is not available in this PyTorch version."
            )

        print("Compiling model. The first run may take longer.")
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize(device)
    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        synchronize(device)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Accuracy: {train_accuracy:.2%}"
        )

    synchronize(device)
    training_seconds = time.perf_counter() - start_time

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    total_examples = len(train_dataset) * EPOCHS
    throughput = total_examples / training_seconds

    memory = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else None
    )

    # torch.compile returns a wrapped model. Save the original model
    # parameters when possible.
    saved_model = getattr(model, "_orig_mod", model)
    checkpoint_path = OUTPUT_DIR / (
        name.lower().replace(" ", "_") + ".pt"
    )
    torch.save(saved_model.state_dict(), checkpoint_path)

    result: dict[str, float | str | None] = {
        "name": name,
        "training_seconds": training_seconds,
        "throughput": throughput,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "peak_memory_mb": memory,
    }

    print(f"Training Time: {training_seconds:.2f} seconds")
    print(f"Throughput: {throughput:.2f} examples/sec")
    print(f"Test Accuracy: {test_accuracy:.2%}")

    if memory is not None:
        print(f"Peak GPU Memory: {memory:.2f} MiB")

    print(f"Checkpoint: {checkpoint_path}")

    return result


def print_comparison(
    results: list[dict[str, float | str | None]],
) -> None:
    print("\n" + "=" * 100)
    print("EFFICIENT TRAINING COMPARISON")
    print("=" * 100)

    print(
        f"{'Configuration':<30}"
        f"{'Time (s)':>12}"
        f"{'Examples/s':>14}"
        f"{'Accuracy':>12}"
        f"{'Peak Memory (MiB)':>22}"
    )
    print("-" * 100)

    for result in results:
        memory = result["peak_memory_mb"]
        memory_text = f"{memory:.2f}" if memory is not None else "N/A"

        print(
            f"{str(result['name']):<30}"
            f"{float(result['training_seconds']):>12.2f}"
            f"{float(result['throughput']):>14.2f}"
            f"{float(result['test_accuracy']):>11.2%}"
            f"{memory_text:>22}"
        )

    baseline_time = float(results[0]["training_seconds"])

    print("\nRelative to baseline:")
    for result in results[1:]:
        speedup = baseline_time / float(result["training_seconds"])
        print(f"{result['name']}: {speedup:.2f}x speedup")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    set_seed()
    device = get_device()

    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    train_dataset, test_dataset = create_datasets()

    reference_model = MNISTCNN()
    initial_state = copy.deepcopy(reference_model.state_dict())

    experiments = [
        {
            "name": "Baseline Eager",
            "optimized_loader": False,
            "use_compile": False,
        },
        {
            "name": "Optimized Data Loading",
            "optimized_loader": True,
            "use_compile": False,
        },
    ]

    # Compilation can be unsupported or unhelpful on some environments.
    # Keep it as a separate experiment and report failures clearly.
    if hasattr(torch, "compile"):
        experiments.append(
            {
                "name": "Compiled Execution",
                "optimized_loader": True,
                "use_compile": True,
            }
        )

    results = []

    for experiment in experiments:
        set_seed()

        try:
            result = run_experiment(
                name=experiment["name"],
                initial_state=initial_state,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                device=device,
                optimized_loader=experiment["optimized_loader"],
                use_compile=experiment["use_compile"],
            )
            results.append(result)

        except Exception as exc:
            print(
                f"\nExperiment '{experiment['name']}' failed: "
                f"{type(exc).__name__}: {exc}"
            )

    if results:
        print_comparison(results)
    else:
        print("No experiments completed successfully.")


if __name__ == "__main__":
    main()
