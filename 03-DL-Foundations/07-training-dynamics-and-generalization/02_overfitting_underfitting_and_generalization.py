"""
Overfitting, Underfitting, and Generalization
=============================================

Experiments:
1. Compare models with different capacities.
2. Track training and validation loss.
3. Identify underfitting, good fit, and overfitting.
4. Measure the generalization gap.
5. Implement early stopping using validation loss.
6. Restore the model checkpoint with the best validation performance.
7. Compare the selected model on an independent test set.
"""

import copy

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
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def prepare_data():
    """Create train, validation, and test splits without preprocessing leakage."""
    features, labels = make_moons(
        n_samples=1800,
        noise=0.30,
        random_state=42,
    )

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=0.40,
        random_state=42,
        stratify=labels,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    def make_dataset(x, y):
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )

    return (
        make_dataset(x_train, y_train),
        make_dataset(x_val, y_val),
        make_dataset(x_test, y_test),
    )


class Classifier(nn.Module):
    def __init__(self, hidden_features: int):
        super().__init__()

        layers = [
            nn.Linear(2, hidden_features),
            nn.ReLU(),
        ]

        if hidden_features >= 32:
            layers.extend([
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
            ])

        layers.append(nn.Linear(hidden_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def evaluate(model, loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    predictions = []
    actual_labels = []

    with torch.inference_mode():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = loss_function(logits, y_batch)

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predicted_classes = (logits >= 0).long()

            predictions.extend(predicted_classes.cpu().numpy().ravel())
            actual_labels.extend(y_batch.cpu().numpy().ravel())

    return (
        total_loss / total_samples,
        accuracy_score(actual_labels, predictions),
    )


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    device,
    max_epochs=200,
    patience=20,
    min_delta=1e-4,
):
    """Stop when validation loss fails to improve and restore best weights."""
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in range(max_epochs):
        model.train()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(x_batch)
            loss = loss_function(logits, y_batch)

            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = evaluate(
            model, train_loader, loss_function, device
        )
        val_loss, val_accuracy = evaluate(
            model, val_loader, loss_function, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Train acc: {train_accuracy:.3f} | "
                f"Val acc: {val_accuracy:.3f}"
            )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return history, best_epoch


def compare_model_capacities(train_dataset, val_dataset, device):
    section("1. Comparing Model Capacities")

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
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

    loss_function = nn.BCEWithLogitsLoss()
    results = {}

    # The number of hidden features controls model capacity.
    for hidden_features in [2, 16, 64]:
        print(f"\nModel hidden features: {hidden_features}")
        torch.manual_seed(42)

        model = Classifier(hidden_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_losses = []
        val_losses = []

        for epoch in range(100):
            model.train()

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = loss_function(logits, y_batch)
                loss.backward()
                optimizer.step()

            train_loss, _ = evaluate(
                model, train_eval_loader, loss_function, device
            )
            val_loss, _ = evaluate(
                model, val_loader, loss_function, device
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        generalization_gap = val_losses[-1] - train_losses[-1]

        results[hidden_features] = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "gap": generalization_gap,
        }

        print(f"Final training loss: {train_losses[-1]:.4f}")
        print(f"Final validation loss: {val_losses[-1]:.4f}")
        print(f"Generalization gap: {generalization_gap:.4f}")

    return results


def plot_capacity_comparison(results):
    section("2. Learning Curves by Model Capacity")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for hidden_features, result in results.items():
        epochs = range(1, len(result["train_loss"]) + 1)

        axes[0].plot(
            epochs,
            result["train_loss"],
            label=f"Hidden={hidden_features}",
        )
        axes[1].plot(
            epochs,
            result["val_loss"],
            label=f"Hidden={hidden_features}",
        )

        axes[2].bar(
            str(hidden_features),
            result["gap"],
            label=f"Hidden={hidden_features}",
        )

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")
    axes[2].set_title("Generalization Gap")

    for axis in axes[:2]:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Binary Cross-Entropy Loss")
        axis.legend()
        axis.grid(True)

    axes[2].set_xlabel("Hidden Features")
    axes[2].set_ylabel("Validation Loss - Training Loss")
    axes[2].axhline(0, linewidth=1)
    axes[2].grid(True, axis="y")

    figure.tight_layout()
    plt.show()


def demonstrate_early_stopping(train_dataset, val_dataset, device):
    section("3. Early Stopping and Best Checkpoint Restoration")

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
    )

    torch.manual_seed(42)
    model = Classifier(hidden_features=64).to(device)

    history, best_epoch = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=200,
        patience=20,
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["train_loss"], label="Training loss")
    axis.plot(history["val_loss"], label="Validation loss")
    axis.axvline(
        best_epoch - 1,
        linestyle="--",
        label=f"Best epoch: {best_epoch}",
    )
    axis.set_title("Early Stopping")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Binary Cross-Entropy Loss")
    axis.legend()
    axis.grid(True)
    figure.tight_layout()
    plt.show()

    return model


def final_test_evaluation(model, test_dataset, device):
    section("4. Independent Test Evaluation")

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
    )

    loss_function = nn.BCEWithLogitsLoss()
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        loss_function,
        device,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    train_dataset, val_dataset, test_dataset = prepare_data()

    capacity_results = compare_model_capacities(
        train_dataset,
        val_dataset,
        device,
    )
    plot_capacity_comparison(capacity_results)

    selected_model = demonstrate_early_stopping(
        train_dataset,
        val_dataset,
        device,
    )

    # The test set is used only for final evaluation.
    final_test_evaluation(
        selected_model,
        test_dataset,
        device,
    )


if __name__ == "__main__":
    main()
