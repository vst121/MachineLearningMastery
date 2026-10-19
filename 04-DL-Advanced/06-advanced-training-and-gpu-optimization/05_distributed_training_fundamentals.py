
"""
05_distributed_training_fundamentals.py

Explore PyTorch DistributedDataParallel (DDP).

Experiments:
1. Single-process baseline.
2. Distributed training using one process per available GPU,
   or CPU processes when CUDA is unavailable.
3. Compare elapsed time, throughput, accuracy, and scaling efficiency.

Notes:
- Distributed execution requires multiple processes.
- CPU distributed execution is supported for learning.
- Multi-GPU speedup should be measured on a machine with multiple GPUs.
"""

from __future__ import annotations

import copy
import os
import random
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 1e-3

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/distributed_training")
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


def synchronize_device(device: torch.device) -> None:
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


def create_single_process_loader(
    dataset,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )


# ---------------------------------------------------------------------
# Single-process baseline
# ---------------------------------------------------------------------

def train_single_process(
    initial_state: dict[str, torch.Tensor],
    train_dataset,
    test_dataset,
    device: torch.device,
) -> dict[str, float | str]:
    print("\n" + "=" * 72)
    print("Single-process baseline")
    print("=" * 72)

    set_seed()

    train_loader = create_single_process_loader(
        train_dataset,
        shuffle=True,
        device=device,
    )
    test_loader = create_single_process_loader(
        test_dataset,
        shuffle=False,
        device=device,
    )

    model = MNISTCNN().to(device)
    model.load_state_dict(initial_state)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize_device(device)
    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_examples += batch_size

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {total_loss / total_examples:.4f} | "
            f"Accuracy: {total_correct / total_examples:.2%}"
        )

    synchronize_device(device)
    elapsed = time.perf_counter() - start_time

    test_accuracy = evaluate_accuracy(
        model,
        test_loader,
        device,
    )

    total_examples = len(train_dataset) * EPOCHS
    throughput = total_examples / elapsed

    checkpoint_path = OUTPUT_DIR / "single_process_model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    result = {
        "name": "Single process",
        "elapsed_seconds": elapsed,
        "throughput": throughput,
        "test_accuracy": test_accuracy,
    }

    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Throughput: {throughput:.2f} examples/sec")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Checkpoint: {checkpoint_path}")

    return result


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ---------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str,
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------
# Distributed worker
# ---------------------------------------------------------------------

def distributed_worker(
    rank: int,
    world_size: int,
    port: int,
    backend: str,
    initial_state: dict[str, torch.Tensor],
    train_dataset,
    test_dataset,
    result_queue,
) -> None:
    try:
        set_seed()

        if backend == "nccl":
            torch.cuda.set_device(rank)
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")

        setup_distributed(rank, world_size, backend, port)

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=SEED,
            drop_last=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )

        # Each rank evaluates only a disjoint subset, avoiding duplicate
        # evaluation work. Metrics are aggregated across ranks.
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            sampler=test_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )

        model = MNISTCNN().to(device)
        model.load_state_dict(initial_state)

        ddp_model = DistributedDataParallel(
            model,
            device_ids=[rank] if device.type == "cuda" else None,
        )

        optimizer = optim.Adam(
            ddp_model.parameters(),
            lr=LEARNING_RATE,
        )
        criterion = nn.CrossEntropyLoss()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        dist.barrier()
        synchronize_device(device)
        start_time = time.perf_counter()

        for epoch in range(1, EPOCHS + 1):
            train_sampler.set_epoch(epoch)
            ddp_model.train()

            total_loss = 0.0
            total_correct = 0
            total_examples = 0

            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits = ddp_model(images)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (
                    logits.argmax(dim=1) == labels
                ).sum().item()
                total_examples += batch_size

            # Aggregate training statistics across ranks.
            stats = torch.tensor(
                [
                    total_loss,
                    float(total_correct),
                    float(total_examples),
                ],
                dtype=torch.float64,
                device=device,
            )

            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            if rank == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Loss: {stats[0].item() / stats[2].item():.4f} | "
                    f"Accuracy: {stats[1].item() / stats[2].item():.2%}"
                )

        # Ensure all ranks have finished training before measuring time.
        dist.barrier()
        synchronize_device(device)
        elapsed = time.perf_counter() - start_time

        ddp_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                predictions = ddp_model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        evaluation_stats = torch.tensor(
            [float(correct), float(total)],
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(
            evaluation_stats,
            op=dist.ReduceOp.SUM,
        )

        if rank == 0:
            # The DistributedSampler may pad its indices to equalize
            # rank lengths. Compute accuracy from aggregated counts.
            test_accuracy = (
                evaluation_stats[0].item()
                / evaluation_stats[1].item()
            )

            # Save the shared model from rank zero.
            checkpoint_path = OUTPUT_DIR / "ddp_model.pt"
            torch.save(ddp_model.module.state_dict(), checkpoint_path)

            result_queue.put(
                {
                    "name": f"DDP ({world_size} processes)",
                    "elapsed_seconds": elapsed,
                    "throughput": (
                        len(train_dataset) * EPOCHS / elapsed
                    ),
                    "test_accuracy": test_accuracy,
                }
            )

            print(f"Distributed training time: {elapsed:.2f} seconds")
            print(
                "Aggregate throughput: "
                f"{len(train_dataset) * EPOCHS / elapsed:.2f} examples/sec"
            )
            print(f"Test accuracy: {test_accuracy:.2%}")
            print(f"Checkpoint: {checkpoint_path}")

    finally:
        cleanup_distributed()


# ---------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------

def print_comparison(
    baseline: dict[str, float | str],
    distributed: dict[str, float | str],
    world_size: int,
) -> None:
    baseline_time = float(baseline["elapsed_seconds"])
    distributed_time = float(distributed["elapsed_seconds"])

    speedup = baseline_time / distributed_time
    efficiency = speedup / world_size

    print("\n" + "=" * 72)
    print("DISTRIBUTED TRAINING COMPARISON")
    print("=" * 72)

    print(
        f"Single-process time: {baseline_time:.2f} seconds\n"
        f"Distributed time:    {distributed_time:.2f} seconds\n"
        f"Speedup:             {speedup:.2f}x\n"
        f"Parallel efficiency: {efficiency:.2%}\n"
        f"Single throughput:   {float(baseline['throughput']):.2f} examples/sec\n"
        f"DDP throughput:      {float(distributed['throughput']):.2f} examples/sec\n"
        f"Single accuracy:     {float(baseline['test_accuracy']):.2%}\n"
        f"DDP accuracy:        {float(distributed['test_accuracy']):.2%}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"PyTorch version: {torch.__version__}")
    print(f"Primary device: {device}")

    if device.type == "cuda":
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    else:
        gpu_count = 0
        print("CUDA unavailable. DDP will use CPU processes.")

    train_dataset, test_dataset = create_datasets()

    reference_model = MNISTCNN()
    initial_state = copy.deepcopy(reference_model.state_dict())

    baseline = train_single_process(
        initial_state=initial_state,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        device=device,
    )

    # Use up to two processes for this introductory experiment.
    # For CUDA, each process uses a separate GPU.
    world_size = min(gpu_count, 2) if gpu_count > 1 else 2
    backend = "nccl" if gpu_count > 1 else "gloo"

    print(
        f"\nStarting DDP with {world_size} processes "
        f"using backend '{backend}'."
    )

    port = find_free_port()
    result_queue = mp.SimpleQueue()

    mp.spawn(
        distributed_worker,
        args=(
            world_size,
            port,
            backend,
            initial_state,
            train_dataset,
            test_dataset,
            result_queue,
        ),
        nprocs=world_size,
        join=True,
    )

    distributed = result_queue.get()

    print_comparison(
        baseline=baseline,
        distributed=distributed,
        world_size=world_size,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
