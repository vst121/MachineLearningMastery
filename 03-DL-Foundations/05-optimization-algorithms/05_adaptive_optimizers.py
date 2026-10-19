"""
Adaptive Optimizers
===================

Experiments:
1. Compare SGD, RMSprop, Adam, and AdamW.
2. Train identical neural networks with different optimizers.
3. Compare training and validation loss.
4. Measure final classification accuracy.
5. Visualize convergence behavior.
"""

import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def prepare_data(seed=42):
    """Generate and standardize a binary classification dataset."""
    X, y = make_moons(
        n_samples=1200,
        noise=0.2,
        random_state=seed,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )

    val_X = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=generator,
    )

    return train_loader, val_X, val_y


class BinaryClassifier(nn.Module):
    """Small feedforward neural network."""

    def __init__(self, input_size=2, hidden_size=32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.network(x)


def create_optimizer(name, parameters, learning_rate):
    """Create the selected optimizer."""
    if name == "SGD":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
        )

    if name == "RMSprop":
        return torch.optim.RMSprop(
            parameters,
            lr=learning_rate,
        )

    if name == "Adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
        )

    if name == "AdamW":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=1e-2,
        )

    raise ValueError(f"Unknown optimizer: {name}")


def evaluate(model, X, y, criterion):
    """Calculate loss and accuracy without tracking gradients."""
    model.eval()

    with torch.no_grad():
        logits = model(X)
        loss = criterion(logits, y).item()
        predictions = (logits >= 0).float()
        accuracy = (predictions == y).float().mean().item()

    return loss, accuracy


def train_model(
    name,
    initial_state,
    train_loader,
    val_X,
    val_y,
    epochs=100,
    learning_rate=0.01,
):
    """Train a model with one optimizer."""
    model = BinaryClassifier()
    model.load_state_dict(copy.deepcopy(initial_state))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = create_optimizer(
        name,
        model.parameters(),
        learning_rate,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / total_samples
        val_loss, val_accuracy = evaluate(
            model,
            val_X,
            val_y,
            criterion,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

    return model, history


def plot_results(results):
    """Plot training loss, validation loss, and validation accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for name, history in results.items():
        axes[0].plot(history["train_loss"], label=name)
        axes[1].plot(history["val_loss"], label=name)
        axes[2].plot(history["val_accuracy"], label=name)

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")
    axes[2].set_title("Validation Accuracy")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[1].set_ylabel("Binary Cross-Entropy")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.5, 1.0)

    fig.suptitle("PyTorch Optimizer Comparison")
    fig.tight_layout()
    plt.show()


def main():
    torch.manual_seed(42)

    train_loader, val_X, val_y = prepare_data()

    # Initialize once so every optimizer starts from identical weights.
    reference_model = BinaryClassifier()
    initial_state = copy.deepcopy(reference_model.state_dict())

    optimizer_configs = {
        "SGD": 0.05,
        "RMSprop": 0.001,
        "Adam": 0.001,
        "AdamW": 0.001,
    }

    results = {}

    print("PyTorch Optimizer Comparison")
    print("-" * 76)
    print(
        f"{'Optimizer':<14}"
        f"{'Train Loss':>14}"
        f"{'Val Loss':>14}"
        f"{'Val Accuracy':>16}"
    )

    for name, learning_rate in optimizer_configs.items():
        _, history = train_model(
            name=name,
            initial_state=initial_state,
            train_loader=train_loader,
            val_X=val_X,
            val_y=val_y,
            epochs=100,
            learning_rate=learning_rate,
        )

        results[name] = history

        print(
            f"{name:<14}"
            f"{history['train_loss'][-1]:>14.4f}"
            f"{history['val_loss'][-1]:>14.4f}"
            f"{history['val_accuracy'][-1]:>15.2%}"
        )

    plot_results(results)


if __name__ == "__main__":
    main()
