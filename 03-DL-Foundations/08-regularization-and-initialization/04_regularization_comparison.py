"""
Regularization Comparison
=========================

Experiments:
1. Train a baseline model without regularization.
2. Compare dropout, weight decay, and their combination.
3. Track training and validation learning curves.
4. Use early stopping to select the best validation checkpoint.
5. Evaluate the selected models on an independent test set.
6. Compare generalization gaps and parameter magnitudes.
"""

import copy
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
INPUT_DIM = 20
HIDDEN_DIM = 128
TRAIN_SIZE = 160
VALIDATION_SIZE = 400
TEST_SIZE = 400
BATCH_SIZE = 32
MAX_EPOCHS = 250
PATIENCE = 25
LEARNING_RATE = 0.003
DROPOUT_PROBABILITY = 0.3
WEIGHT_DECAY = 0.01


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)


def generate_dataset(
    size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate nonlinear regression data with noisy targets."""
    generator = torch.Generator().manual_seed(seed)

    features = torch.randn(
        size,
        INPUT_DIM,
        generator=generator,
    )

    noise = 0.4 * torch.randn(
        size,
        generator=generator,
    )

    targets = (
        2.0 * torch.sin(features[:, 0])
        + 0.5 * features[:, 1].square()
        + features[:, 2] * features[:, 3]
        + noise
    ).unsqueeze(1)

    return features, targets


def create_datasets() -> tuple[
    TensorDataset,
    TensorDataset,
    TensorDataset,
]:
    """Create independent training, validation, and test datasets."""
    train_features, train_targets = generate_dataset(
        TRAIN_SIZE,
        seed=SEED,
    )
    validation_features, validation_targets = generate_dataset(
        VALIDATION_SIZE,
        seed=SEED + 1,
    )
    test_features, test_targets = generate_dataset(
        TEST_SIZE,
        seed=SEED + 2,
    )

    train_dataset = TensorDataset(
        train_features,
        train_targets,
    )
    validation_dataset = TensorDataset(
        validation_features,
        validation_targets,
    )
    test_dataset = TensorDataset(
        test_features,
        test_targets,
    )

    return train_dataset, validation_dataset, test_dataset


class RegularizationMLP(nn.Module):
    """High-capacity MLP with optional dropout."""

    def __init__(self, use_dropout: bool = False) -> None:
        super().__init__()

        layers = [
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
        ]

        if use_dropout:
            layers.append(nn.Dropout(DROPOUT_PROBABILITY))

        layers.extend([
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        ])

        if use_dropout:
            layers.append(nn.Dropout(DROPOUT_PROBABILITY))

        layers.extend([
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        ])

        if use_dropout:
            layers.append(nn.Dropout(DROPOUT_PROBABILITY))

        layers.append(nn.Linear(HIDDEN_DIM, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Calculate mean squared error in evaluation mode."""
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


def train_experiment(
    name: str,
    use_dropout: bool,
    weight_decay: float,
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
) -> dict:
    """Train a configuration and retain its best validation checkpoint."""
    set_seed()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = RegularizationMLP(use_dropout=use_dropout)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    train_losses = []
    validation_losses = []

    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()

        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)

            predictions = model(features)
            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

        # Evaluate with dropout disabled so the training and validation
        # losses are measured under consistent inference behavior.
        train_loss = evaluate(
            model,
            train_eval_loader,
            criterion,
        )
        validation_loss = evaluate(
            model,
            validation_loader,
            criterion,
        )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 25 == 0:
            print(
                f"{name:20s} | Epoch {epoch + 1:3d} | "
                f"Train MSE: {train_loss:.4f} | "
                f"Validation MSE: {validation_loss:.4f}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"{name}: early stopping at epoch {epoch + 1}; "
                f"best checkpoint was epoch {best_epoch}."
            )
            break

    model.load_state_dict(best_state)

    return {
        "name": name,
        "model": model,
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }


def calculate_parameter_norms(model: nn.Module) -> dict:
    """Calculate L1 and L2 norms of weight matrices."""
    weights = [
        parameter.detach().flatten()
        for name, parameter in model.named_parameters()
        if name.endswith("weight")
    ]

    all_weights = torch.cat(weights)

    return {
        "l1_norm": all_weights.abs().sum().item(),
        "l2_norm": all_weights.norm().item(),
    }


def run_comparison() -> dict:
    """Compare regularization strategies using the same datasets."""
    train_dataset, validation_dataset, test_dataset = create_datasets()

    configurations = [
        {
            "name": "Baseline",
            "use_dropout": False,
            "weight_decay": 0.0,
        },
        {
            "name": "Dropout",
            "use_dropout": True,
            "weight_decay": 0.0,
        },
        {
            "name": "Weight Decay",
            "use_dropout": False,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "name": "Dropout + Weight Decay",
            "use_dropout": True,
            "weight_decay": WEIGHT_DECAY,
        },
    ]

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    criterion = nn.MSELoss()
    results = {}

    for configuration in configurations:
        name = configuration["name"]

        print(f"\nTraining configuration: {name}")

        result = train_experiment(
            name=name,
            use_dropout=configuration["use_dropout"],
            weight_decay=configuration["weight_decay"],
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
        )

        test_loss = evaluate(
            result["model"],
            test_loader,
            criterion,
        )

        parameter_norms = calculate_parameter_norms(
            result["model"]
        )

        result["test_loss"] = test_loss
        result["parameter_norms"] = parameter_norms

        results[name] = result

        best_epoch = result["best_epoch"]
        best_train_loss = result["train_losses"][best_epoch - 1]
        best_validation_loss = result["best_validation_loss"]
        generalization_gap = best_validation_loss - best_train_loss

        print(f"Best epoch: {best_epoch}")
        print(f"Training MSE at best epoch: {best_train_loss:.4f}")
        print(f"Best validation MSE: {best_validation_loss:.4f}")
        print(f"Test MSE: {test_loss:.4f}")
        print(f"Generalization gap: {generalization_gap:.4f}")
        print(f"Weight L1 norm: {parameter_norms['l1_norm']:.4f}")
        print(f"Weight L2 norm: {parameter_norms['l2_norm']:.4f}")

    return results


def plot_learning_curves(results: dict) -> None:
    """Plot training and validation curves for each configuration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, result in results.items():
        axes[0].plot(
            result["train_losses"],
            label=name,
        )
        axes[1].plot(
            result["validation_losses"],
            label=name,
        )

        best_epoch = result["best_epoch"]
        best_loss = result["best_validation_loss"]

        axes[1].scatter(
            [best_epoch - 1],
            [best_loss],
            s=35,
        )

    axes[0].set_title("Training MSE")
    axes[1].set_title("Validation MSE and Selected Checkpoints")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean Squared Error")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    plt.show()


def plot_test_performance(results: dict) -> None:
    """Compare held-out test loss across regularization strategies."""
    names = list(results.keys())
    test_losses = [
        results[name]["test_loss"]
        for name in names
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(names, test_losses)
    plt.title("Held-Out Test Performance")
    plt.xlabel("Regularization Strategy")
    plt.ylabel("Test MSE")
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    experiment_results = run_comparison()
    plot_learning_curves(experiment_results)
    plot_test_performance(experiment_results)
