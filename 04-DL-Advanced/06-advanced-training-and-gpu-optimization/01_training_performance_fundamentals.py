"""
Establish a reproducible baseline for deep learning training performance.

This experiment covers:
1. Building a small image classification model.
2. Measuring training time and examples processed per second.
3. Measuring inference throughput separately from training throughput.
4. Tracking peak GPU memory usage when CUDA is available.
5. Comparing warm-up and steady-state performance.
"""

import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

DATA_DIR = Path("data/mnist")
OUTPUT_DIR = Path("outputs/training_performance")


class ImageClassifier(nn.Module):
    """A compact CNN for MNIST classification."""

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def set_seed(seed: int) -> None:
    """Set random seeds for more reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    """Wait for pending CUDA operations before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_loaders(
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

    common_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **common_options,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_options,
    )

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    synchronize(device)
    start_time = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "seconds": elapsed_seconds,
        "examples_per_second": total_examples / elapsed_seconds,
        "examples": float(total_examples),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_correct = 0
    total_examples = 0

    synchronize(device)
    start_time = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)

        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "accuracy": total_correct / total_examples,
        "seconds": elapsed_seconds,
        "examples_per_second": total_examples / elapsed_seconds,
        "examples": float(total_examples),
    }


def main() -> None:
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(
            "GPU memory before model creation: "
            f"{torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB"
        )

    train_loader, test_loader = build_loaders(device)

    model = ImageClassifier().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    loss_function = nn.CrossEntropyLoss()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epoch_results: list[dict[str, float]] = []

    for epoch in range(EPOCHS):
        metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
        )
        epoch_results.append(metrics)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss: {metrics['loss']:.4f} | "
            f"accuracy: {metrics['accuracy']:.2%} | "
            f"time: {metrics['seconds']:.2f}s | "
            f"throughput: {metrics['examples_per_second']:.1f} "
            "examples/s"
        )

    evaluation = evaluate(model, test_loader, device)

    print("\nEvaluation:")
    print(f"Test accuracy: {evaluation['accuracy']:.2%}")
    print(f"Inference time: {evaluation['seconds']:.2f}s")
    print(
        "Inference throughput: "
        f"{evaluation['examples_per_second']:.1f} examples/s"
    )

    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        print("\nGPU memory:")
        print(f"Peak allocated: {peak_allocated / 1024**2:.2f} MiB")
        print(f"Peak reserved: {peak_reserved / 1024**2:.2f} MiB")

    checkpoint_path = OUTPUT_DIR / "baseline_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch_results": epoch_results,
            "evaluation": evaluation,
            "seed": SEED,
        },
        checkpoint_path,
    )

    # Final correctness checks.
    model.eval()
    with torch.no_grad():
        sample_images, _ = next(iter(test_loader))
        sample_logits = model(sample_images[:8].to(device))

    assert sample_logits.shape == (8, 10)
    assert torch.isfinite(sample_logits).all()
    assert all(result["seconds"] > 0 for result in epoch_results)
    assert all(
        result["examples_per_second"] > 0
        for result in epoch_results
    )

    print("\nVerification passed.")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
