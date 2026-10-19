"""
L1, L2 Regularization, and Weight Decay
======================================

Experiments:
1. Compare an unregularized model with L1 and L2 regularization.
2. Inspect parameter magnitudes and sparsity after training.
3. Compare Adam with AdamW under weight decay.
4. Analyze training loss, validation loss, and generalization.
5. Examine the effect of regularization strength.
"""

import copy
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
INPUT_DIM = 20
HIDDEN_DIM = 64
OUTPUT_DIM = 1
TRAIN_SIZE = 600
VALIDATION_SIZE = 200
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.01
REGULARIZATION_STRENGTH = 0.001


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)


def create_dataset() -> tuple[DataLoader, DataLoader]:
    """Create a synthetic regression dataset with informative features."""
    set_seed()

    total_size = TRAIN_SIZE + VALIDATION_SIZE
    features = torch.randn(total_size, INPUT_DIM)

    true_weights = torch.zeros(INPUT_DIM)
    true_weights[:5] = torch.tensor([3.0, -2.0, 1.5, 0.5, -1.0])

    noise = 0.3 * torch.randn(total_size)
    targets = (features @ true_weights + noise).unsqueeze(1)

    train_features = features[:TRAIN_SIZE]
    train_targets = targets[:TRAIN_SIZE]
    validation_features = features[TRAIN_SIZE:]
    validation_targets = targets[TRAIN_SIZE:]

    train_dataset = TensorDataset(train_features, train_targets)
    validation_dataset = TensorDataset(
        validation_features,
        validation_targets,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, validation_loader


class RegressionMLP(nn.Module):
    """A small multilayer perceptron for regression."""

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def regularization_penalty(
    model: nn.Module,
    method: str,
    strength: float,
) -> torch.Tensor:
    """Calculate an L1 or L2 penalty over weight matrices only."""
    weights = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("weight")
    ]

    if method == "none":
        return sum(
            (parameter.sum() * 0.0 for parameter in weights),
            torch.tensor(0.0),
        )

    if method == "l1":
        penalty = sum(parameter.abs().sum() for parameter in weights)

    elif method == "l2":
        penalty = sum(parameter.square().sum() for parameter in weights)

    else:
        raise ValueError(f"Unsupported regularization method: {method}")

    return strength * penalty


def calculate_epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Calculate mean data loss without adding regularization penalties."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for features, targets in loader:
            predictions = model(features)
            loss = criterion(predictions, targets)

            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

    return total_loss / total_samples


def train_model(
    method: str,
    strength: float,
    optimizer_name: str = "adam",
    epochs: int = EPOCHS,
) -> dict:
    """Train a model and return loss curves and parameter statistics."""
    set_seed()

    train_loader, validation_loader = create_dataset()
    model = RegressionMLP()
    criterion = nn.MSELoss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=strength if method == "weight_decay" else 0.0,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    train_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.train()

        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)

            predictions = model(features)
            data_loss = criterion(predictions, targets)

            if method in {"l1", "l2"}:
                penalty = regularization_penalty(
                    model,
                    method,
                    strength,
                )
            else:
                penalty = data_loss.new_zeros(())

            loss = data_loss + penalty
            loss.backward()
            optimizer.step()

        train_loss = calculate_epoch_loss(
            model,
            train_loader,
            criterion,
        )
        validation_loss = calculate_epoch_loss(
            model,
            validation_loader,
            criterion,
        )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if (epoch + 1) % 20 == 0:
            print(
                f"{method:12s} | {optimizer_name:5s} | "
                f"Epoch {epoch + 1:3d} | "
                f"Train MSE: {train_loss:.4f} | "
                f"Validation MSE: {validation_loss:.4f}"
            )

    weights = torch.cat(
        [
            parameter.detach().flatten()
            for name, parameter in model.named_parameters()
            if name.endswith("weight")
        ]
    )

    return {
        "model": copy.deepcopy(model),
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "weight_l1_norm": weights.abs().sum().item(),
        "weight_l2_norm": weights.norm().item(),
        "near_zero_fraction": (
            (weights.abs() < 1e-3).float().mean().item()
        ),
    }


def experiment_l1_l2_comparison() -> dict:
    """Compare no regularization, L1, and L2."""
    configurations = [
        ("none", 0.0),
        ("l1", REGULARIZATION_STRENGTH),
        ("l2", REGULARIZATION_STRENGTH),
    ]

    results = {}

    for method, strength in configurations:
        print(f"\nTraining with {method.upper()} regularization")

        results[method] = train_model(
            method=method,
            strength=strength,
        )

        result = results[method]

        print(f"Weight L1 norm: {result['weight_l1_norm']:.4f}")
        print(f"Weight L2 norm: {result['weight_l2_norm']:.4f}")
        print(
            "Fraction of near-zero weights: "
            f"{result['near_zero_fraction']:.2%}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for method, result in results.items():
        axes[0].plot(
            result["train_losses"],
            label=method.upper(),
        )
        axes[1].plot(
            result["validation_losses"],
            label=method.upper(),
        )

    axes[0].set_title("Training MSE")
    axes[1].set_title("Validation MSE")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean Squared Error")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    plt.show()

    return results


def experiment_adam_vs_adamw() -> dict:
    """Compare L2 penalties in Adam with decoupled weight decay in AdamW."""
    strength = REGULARIZATION_STRENGTH

    configurations = [
        ("none", "adam", 0.0),
        ("l2", "adam", strength),
        ("weight_decay", "adamw", strength),
    ]

    results = {}

    for method, optimizer_name, coefficient in configurations:
        label = f"{optimizer_name.upper()} + {method}"

        print(f"\nTraining: {label}")

        results[label] = train_model(
            method=method,
            strength=coefficient,
            optimizer_name=optimizer_name,
        )

        result = results[label]

        print(f"Weight L1 norm: {result['weight_l1_norm']:.4f}")
        print(f"Weight L2 norm: {result['weight_l2_norm']:.4f}")
        print(
            f"Final validation MSE: "
            f"{result['validation_losses'][-1]:.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, result in results.items():
        axes[0].plot(
            result["train_losses"],
            label=label,
        )
        axes[1].plot(
            result["validation_losses"],
            label=label,
        )

    axes[0].set_title("Training MSE")
    axes[1].set_title("Validation MSE")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean Squared Error")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    plt.show()

    return results


def experiment_regularization_strength() -> None:
    """Observe how L2 regularization strength affects generalization."""
    strengths = [0.0, 0.0001, 0.001, 0.01, 0.1]

    validation_losses = []
    weight_norms = []

    for strength in strengths:
        method = "none" if strength == 0.0 else "l2"

        result = train_model(
            method=method,
            strength=strength,
            epochs=EPOCHS,
        )

        validation_losses.append(result["validation_losses"][-1])
        weight_norms.append(result["weight_l2_norm"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        strengths,
        validation_losses,
        marker="o",
    )
    axes[0].set_xscale("symlog", linthresh=1e-4)
    axes[0].set_title("Validation MSE vs. Regularization Strength")
    axes[0].set_xlabel("L2 Strength")
    axes[0].set_ylabel("Validation MSE")

    axes[1].plot(
        strengths,
        weight_norms,
        marker="o",
    )
    axes[1].set_xscale("symlog", linthresh=1e-4)
    axes[1].set_title("Weight Norm vs. Regularization Strength")
    axes[1].set_xlabel("L2 Strength")
    axes[1].set_ylabel("L2 Weight Norm")

    for axis in axes:
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    experiment_l1_l2_comparison()
    experiment_adam_vs_adamw()
    experiment_regularization_strength()
