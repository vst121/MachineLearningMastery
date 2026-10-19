
"""
03_gpu_memory_optimization.py

Investigate gradient accumulation and activation checkpointing.

Experiments:
1. Baseline training.
2. Gradient accumulation with smaller microbatches.
3. Activation checkpointing.
4. Combined gradient accumulation and activation checkpointing.

Run:
    uv run python "04-DL-Advanced/06. Advanced Training and GPU Optimization/03_gpu_memory_optimization.py"
"""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
BATCH_SIZE = 128
MICROBATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICROBATCH_SIZE

EPOCHS = 3
LEARNING_RATE = 1e-3

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/gpu_memory_optimization")
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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class MNISTCNN(nn.Module):
    def __init__(self, activation_checkpointing: bool = False) -> None:
        super().__init__()

        self.activation_checkpointing = activation_checkpointing

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        if self.activation_checkpointing and self.training:
            x = checkpoint(
                self.classifier,
                x,
                use_reentrant=False,
            )
        else:
            x = self.classifier(x)

        return x


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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None

    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    accumulation_steps: int,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    optimizer.zero_grad(set_to_none=True)

    # Handle the final, possibly incomplete accumulation group.
    num_batches = len(loader)

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, num_batches - group_start)

        logits = model(images)
        loss = criterion(logits, labels)

        # Scale gradients to average over the current accumulation group.
        (loss / group_size).backward()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

        is_group_end = (
            (batch_index + 1) % accumulation_steps == 0
            or batch_index + 1 == num_batches
        )

        if is_group_end:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / total_examples, total_correct / total_examples


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
# Experiment runner
# ---------------------------------------------------------------------

def run_experiment(
    name: str,
    initial_state: dict[str, torch.Tensor],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    activation_checkpointing: bool,
    accumulation_steps: int,
) -> dict[str, float | str | None]:
    print(f"\n{'=' * 72}")
    print(f"Experiment: {name}")
    print(f"{'=' * 72}")

    model = MNISTCNN(
        activation_checkpointing=activation_checkpointing,
    ).to(device)

    model.load_state_dict(initial_state)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    reset_peak_memory(device)
    synchronize(device)
    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            accumulation_steps=accumulation_steps,
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

    total_examples = len(train_loader.dataset) * EPOCHS
    throughput = total_examples / training_seconds

    memory = peak_memory_mb(device)

    checkpoint_path = OUTPUT_DIR / (
        name.lower().replace(" ", "_").replace("+", "plus")
        + ".pt"
    )
    torch.save(model.state_dict(), checkpoint_path)

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
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")

    if memory is not None:
        print(f"Peak GPU Memory: {memory:.2f} MiB")

    print(f"Saved Model: {checkpoint_path}")

    return result


# ---------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------

def print_comparison(
    results: list[dict[str, float | str | None]],
) -> None:
    print("\n" + "=" * 105)
    print("GPU MEMORY OPTIMIZATION COMPARISON")
    print("=" * 105)

    print(
        f"{'Configuration':<34}"
        f"{'Time (s)':>12}"
        f"{'Examples/s':>14}"
        f"{'Accuracy':>12}"
        f"{'Peak Memory (MiB)':>22}"
    )
    print("-" * 105)

    for result in results:
        memory = result["peak_memory_mb"]
        memory_text = f"{memory:.2f}" if memory is not None else "N/A"

        print(
            f"{str(result['name']):<34}"
            f"{float(result['training_seconds']):>12.2f}"
            f"{float(result['throughput']):>14.2f}"
            f"{float(result['test_accuracy']):>11.2%}"
            f"{memory_text:>22}"
        )

    baseline = results[0]
    baseline_time = float(baseline["training_seconds"])
    baseline_memory = baseline["peak_memory_mb"]

    print("\nRelative to baseline:")

    for result in results[1:]:
        speedup = baseline_time / float(result["training_seconds"])
        print(f"{result['name']}: {speedup:.2f}x speedup")

        memory = result["peak_memory_mb"]

        if baseline_memory and memory is not None and baseline_memory > 0:
            reduction = (
                (baseline_memory - memory) / baseline_memory
            ) * 100
            print(f"  Peak memory reduction: {reduction:.2f}%")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    set_seed()
    device = torch.device(get_device())

    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Microbatch size: {MICROBATCH_SIZE}")
    print(f"Accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA unavailable. Running experiments on CPU.")

    train_loader, test_loader = create_dataloaders(device)

    reference_model = MNISTCNN()
    initial_state = copy.deepcopy(reference_model.state_dict())

    experiments = [
        {
            "name": "Baseline",
            "checkpointing": False,
            "accumulation_steps": 1,
        },
        {
            "name": "Gradient Accumulation",
            "checkpointing": False,
            "accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        },
        {
            "name": "Activation Checkpointing",
            "checkpointing": True,
            "accumulation_steps": 1,
        },
        {
            "name": "Accumulation + Checkpointing",
            "checkpointing": True,
            "accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        },
    ]

    results = []

    for experiment in experiments:
        set_seed()

        result = run_experiment(
            name=experiment["name"],
            initial_state=initial_state,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            activation_checkpointing=experiment["checkpointing"],
            accumulation_steps=experiment["accumulation_steps"],
        )

        results.append(result)

    print_comparison(results)


if __name__ == "__main__":
    main()
