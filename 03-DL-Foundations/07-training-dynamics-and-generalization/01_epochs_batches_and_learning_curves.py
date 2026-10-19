"""
Epochs, Batches, and Learning Curves
===================================

Experiments:
1. Compare epochs, batches, iterations, and optimizer updates.
2. Calculate iterations per epoch for different batch sizes.
3. Train a classifier using mini-batch gradient descent.
4. Track training and validation loss and accuracy.
5. Compare learning curves for different batch sizes.
6. Observe how batch size affects optimization behavior.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def section(title: str) -> None:
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")


def prepare_data():
    """Generate data and fit preprocessing on training data only."""
    features, labels = make_moons(
        n_samples=1800,
        noise=0.25,
        random_state=42,
    )

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )

    x_val, _, y_val, _ = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    def create_dataset(x, y):
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )

    return create_dataset(x_train, y_train), create_dataset(x_val, y_val)


class MoonClassifier(nn.Module):
    def __init__(self, input_features: int = 2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def show_batch_math(dataset_size: int, batch_sizes: list[int]) -> None:
    section("1. Epochs, Batches, and Iterations")

    print(f"Training samples: {dataset_size}")

    for batch_size in batch_sizes:
        batches = math.ceil(dataset_size / batch_size)

        print(
            f"Batch size: {batch_size:3d} | "
            f"Batches per epoch: {batches:3d} | "
            f"Updates over 30 epochs: {batches * 30:4d}"
        )

    print("\nIterations per epoch = ceil(dataset size / batch size).")
    print("Assumption: one optimizer update per batch.")


def calculate_metrics(model, data_loader, device):
    """Evaluate average loss and classification accuracy."""
    model.eval()
    loss_function = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = loss_function(logits, y_batch)

            predictions = (torch.sigmoid(logits) >= 0.5).long()

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_predictions.extend(predictions.cpu().numpy().ravel())
            all_labels.extend(y_batch.cpu().numpy().ravel())

    average_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy


def train_experiment(
    train_dataset,
    val_dataset,
    batch_size: int,
    epochs: int,
    device,
):
    """Train one model and return its learning history."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
    )

    model = MoonClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "updates": [],
    }

    updates_per_epoch = len(train_loader)
    total_updates = 0

    for epoch in range(epochs):
        model.train()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(x_batch)
            loss = loss_function(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_updates += 1

        train_loss, train_accuracy = calculate_metrics(
            model, train_eval_loader, device
        )
        val_loss, val_accuracy = calculate_metrics(
            model, val_loader, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["updates"].append(total_updates)

        if (epoch + 1) % 10 == 0:
            print(
                f"Batch size {batch_size:3d} | "
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Updates: {total_updates:4d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val accuracy: {val_accuracy:.4f}"
            )

    assert total_updates == epochs * updates_per_epoch

    return history


def plot_learning_curves(histories):
    section("3. Visualizing Learning Curves")

    epochs = range(1, len(next(iter(histories.values()))["train_loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    for batch_size, history in histories.items():
        axes[0].plot(
            epochs,
            history["train_loss"],
            label=f"Train, batch={batch_size}",
        )
        axes[0].plot(
            epochs,
            history["val_loss"],
            linestyle="--",
            label=f"Validation, batch={batch_size}",
        )

        axes[1].plot(
            epochs,
            history["val_accuracy"],
            label=f"Batch={batch_size}",
        )

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    figure.tight_layout()
    plt.show()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    train_dataset, val_dataset = prepare_data()

    epochs = 30
    batch_sizes = [16, 64, 256]

    show_batch_math(len(train_dataset), batch_sizes)

    section("2. Training Experiments")

    histories = {}

    for batch_size in batch_sizes:
        print(f"\nTraining with batch size {batch_size}")

        # Reset the seed to reduce initialization differences between runs.
        torch.manual_seed(42)

        histories[batch_size] = train_experiment(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
        )

    plot_learning_curves(histories)

    section("4. Final Comparison")

    for batch_size, history in histories.items():
        print(
            f"Batch size {batch_size:3d} | "
            f"Final train loss: {history['train_loss'][-1]:.4f} | "
            f"Final val loss: {history['val_loss'][-1]:.4f} | "
            f"Final val accuracy: {history['val_accuracy'][-1]:.4f} | "
            f"Total updates: {history['updates'][-1]}"
        )


if __name__ == "__main__":
    main()
