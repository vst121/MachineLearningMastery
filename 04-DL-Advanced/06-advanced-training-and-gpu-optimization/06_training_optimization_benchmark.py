"""
06_training_optimization_benchmark.py

Compare baseline and optimized PyTorch training configurations.

Experiments:
1. FP32 baseline.
2. Automatic mixed precision (FP16 or BF16) when supported.
3. torch.compile when available.
4. Mixed precision + torch.compile when supported.

Metrics:
- Training time
- Training throughput
- Test loss and accuracy
- Peak allocated GPU memory
- Speedup and relative memory reduction versus baseline

"""

from __future__ import annotations

import copy
import csv
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

NUM_WORKERS = 0

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/training_optimization_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "benchmark_results.csv"


# ---------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


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
# Experiment configurations
# ---------------------------------------------------------------------

def create_configurations(
    device: torch.device,
) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = [
        {
            "name": "FP32 Baseline",
            "amp_dtype": None,
            "compile": False,
        }
    ]

    if device.type == "cuda":
        amp_dtypes = [torch.float16]

        if torch.cuda.is_bf16_supported():
            amp_dtypes.append(torch.bfloat16)

        for dtype in amp_dtypes:
            precision_name = (
                "FP16 AMP"
                if dtype == torch.float16
                else "BF16 AMP"
            )

            configs.append(
                {
                    "name": precision_name,
                    "amp_dtype": dtype,
                    "compile": False,
                }
            )

    if hasattr(torch, "compile"):
        configs.append(
            {
                "name": "Compiled FP32",
                "amp_dtype": None,
                "compile": True,
            }
        )

        if device.type == "cuda":
            configs.append(
                {
                    "name": "Compiled FP16 AMP",
                    "amp_dtype": torch.float16,
                    "compile": True,
                }
            )

            if torch.cuda.is_bf16_supported():
                configs.append(
                    {
                        "name": "Compiled BF16 AMP",
                        "amp_dtype": torch.bfloat16,
                        "compile": True,
                    }
                )

    return configs


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
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
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
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
    amp_dtype: torch.dtype | None,
) -> tuple[float, float]:
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

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
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_examples += batch_size

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


# ---------------------------------------------------------------------
# Benchmark one configuration
# ---------------------------------------------------------------------

def run_benchmark(
    config: dict[str, object],
    initial_state: dict[str, torch.Tensor],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    name = str(config["name"])
    amp_dtype = config["amp_dtype"]
    use_compile = bool(config["compile"])

    print(f"\n{'=' * 75}")
    print(f"Configuration: {name}")
    print(f"{'=' * 75}")

    set_seed()

    base_model = MNISTCNN().to(device)
    base_model.load_state_dict(initial_state)

    if use_compile:
        print(
            "torch.compile enabled. Compilation overhead is included "
            "in the measured training time."
        )
        model = torch.compile(base_model)
    else:
        model = base_model

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.CrossEntropyLoss()

    # Gradient scaling is enabled only for CUDA FP16 AMP.
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(
            device.type == "cuda"
            and amp_dtype == torch.float16
        ),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize(device)
    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            amp_dtype=amp_dtype,
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
        model=model,
        loader=test_loader,
        device=device,
        amp_dtype=amp_dtype,
    )

    total_examples = len(train_loader.dataset) * EPOCHS
    throughput = total_examples / training_seconds

    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else None
    )

    # Save the original module when the model is compiled.
    model_to_save = getattr(model, "_orig_mod", model)

    checkpoint_path = OUTPUT_DIR / (
        name.lower().replace(" ", "_").replace("+", "plus")
        + ".pt"
    )
    torch.save(model_to_save.state_dict(), checkpoint_path)

    result: dict[str, object] = {
        "name": name,
        "training_seconds": training_seconds,
        "throughput": throughput,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "peak_memory_mb": peak_memory_mb,
        "checkpoint": str(checkpoint_path),
    }

    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Throughput: {throughput:.2f} examples/sec")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")

    if peak_memory_mb is not None:
        print(f"Peak GPU memory: {peak_memory_mb:.2f} MiB")

    print(f"Checkpoint: {checkpoint_path}")

    return result


# ---------------------------------------------------------------------
# Comparison and report
# ---------------------------------------------------------------------

def print_comparison(
    results: list[dict[str, object]],
) -> None:
    baseline = results[0]

    baseline_time = float(baseline["training_seconds"])
    baseline_memory = baseline["peak_memory_mb"]

    print("\n" + "=" * 112)
    print("TRAINING OPTIMIZATION BENCHMARK")
    print("=" * 112)

    print(
        f"{'Configuration':<24}"
        f"{'Time (s)':>12}"
        f"{'Examples/s':>14}"
        f"{'Accuracy':>12}"
        f"{'Speedup':>11}"
        f"{'Peak Memory MiB':>19}"
        f"{'Memory Saved %':>17}"
    )
    print("-" * 112)

    for result in results:
        elapsed = float(result["training_seconds"])
        throughput = float(result["throughput"])
        accuracy = float(result["test_accuracy"])
        memory = result["peak_memory_mb"]

        speedup = baseline_time / elapsed

        if (
            baseline_memory is not None
            and memory is not None
            and float(baseline_memory) > 0
        ):
            memory_saved = (
                (float(baseline_memory) - float(memory))
                / float(baseline_memory)
            ) * 100
            memory_text = f"{float(memory):.2f}"
            saved_text = f"{memory_saved:.2f}%"
        else:
            memory_text = "N/A"
            saved_text = "N/A"

        print(
            f"{str(result['name']):<24}"
            f"{elapsed:>12.2f}"
            f"{throughput:>14.2f}"
            f"{accuracy:>11.2%}"
            f"{speedup:>10.2f}x"
            f"{memory_text:>19}"
            f"{saved_text:>17}"
        )

    print("\nInterpretation:")
    print("- Speedup greater than 1.00x means faster training than baseline.")
    print("- Positive memory savings mean lower peak allocated GPU memory.")
    print("- Compare accuracy before choosing an optimization.")


def save_results(
    results: list[dict[str, object]],
) -> None:
    fieldnames = [
        "name",
        "training_seconds",
        "throughput",
        "test_loss",
        "test_accuracy",
        "peak_memory_mb",
        "checkpoint",
    ]

    with RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    key: result.get(key)
                    for key in fieldnames
                }
            )

    print(f"\nBenchmark results saved to: {RESULTS_FILE}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    set_seed()
    device = get_device()

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(
            "CUDA is unavailable. Running FP32 and compiled FP32 "
            "where supported. GPU-specific comparisons are skipped."
        )

    train_loader, test_loader = create_dataloaders(device)

    reference_model = MNISTCNN()
    initial_state = copy.deepcopy(reference_model.state_dict())

    configurations = create_configurations(device)
    results: list[dict[str, object]] = []

    for config in configurations:
        try:
            result = run_benchmark(
                config=config,
                initial_state=initial_state,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
            )
            results.append(result)

        except Exception as exc:
            print(
                f"\nConfiguration '{config['name']}' failed: "
                f"{type(exc).__name__}: {exc}"
            )

            # A failed compilation or unsupported precision should not
            # prevent the remaining configurations from being tested.

    if not results:
        print("No benchmark configurations completed successfully.")
        return

    print_comparison(results)
    save_results(results)


if __name__ == "__main__":
    main()
