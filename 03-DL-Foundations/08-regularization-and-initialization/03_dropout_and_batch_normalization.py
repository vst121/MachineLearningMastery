"""
Dropout and Batch Normalization
===============================

Experiments:
1. Compare models with no regularization, dropout, batch normalization,
   and both techniques.
2. Track training and validation loss.
3. Inspect how dropout changes activations during training and evaluation.
4. Examine batch normalization statistics and train/eval behavior.
5. Demonstrate why batch size matters for BatchNorm1d.
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
DROPOUT_PROBABILITY = 0.3


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)


def create_dataset() -> tuple[DataLoader, DataLoader]:
    """Create a synthetic regression dataset."""
    set_seed()

    total_size = TRAIN_SIZE + VALIDATION_SIZE
    features = torch.randn(total_size, INPUT_DIM)

    true_weights = torch.zeros(INPUT_DIM)
    true_weights[:5] = torch.tensor([3.0, -2.0, 1.5, 0.5, -1.0])

    targets = (
        features @ true_weights
        + 0.5 * torch.randn(total_size)
    ).unsqueeze(1)

    train_dataset = TensorDataset(
        features[:TRAIN_SIZE],
        targets[:TRAIN_SIZE],
    )
    validation_dataset = TensorDataset(
        features[TRAIN_SIZE:],
        targets[TRAIN_SIZE:],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, validation_loader


class RegularizedMLP(nn.Module):
    """MLP with configurable dropout and batch normalization."""

    def __init__(
        self,
        use_dropout: bool = False,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        layers = [
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(HIDDEN_DIM))

        layers.append(nn.ReLU())

        if use_dropout:
            layers.append(
                nn.Dropout(p=DROPOUT_PROBABILITY)
            )

        layers.append(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        )

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(HIDDEN_DIM))

        layers.append(nn.ReLU())

        if use_dropout:
            layers.append(
                nn.Dropout(p=DROPOUT_PROBABILITY)
            )

        layers.append(nn.Linear(HIDDEN_DIM, OUTPUT_DIM))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Calculate mean squared error without gradient tracking."""
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
    use_dropout: bool,
    use_batch_norm: bool,
    epochs: int = EPOCHS,
) -> dict:
    """Train one model configuration and record learning curves."""
    set_seed()

    train_loader, validation_loader = create_dataset()

    model = RegularizedMLP(
        use_dropout=use_dropout,
        use_batch_norm=use_batch_norm,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.MSELoss()

    train_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.train()

        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)

            predictions = model(features)
            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

        train_losses.append(
            evaluate(model, train_loader, criterion)
        )
        validation_losses.append(
            evaluate(model, validation_loader, criterion)
        )

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3d} | "
                f"Train MSE: {train_losses[-1]:.4f} | "
                f"Validation MSE: {validation_losses[-1]:.4f}"
            )

    return {
        "model": copy.deepcopy(model),
        "train_losses": train_losses,
        "validation_losses": validation_losses,
    }


def experiment_regularization_comparison() -> dict:
    """Compare dropout and batch normalization combinations."""
    configurations = {
        "Baseline": (False, False),
        "Dropout": (True, False),
        "BatchNorm": (False, True),
        "Dropout + BatchNorm": (True, True),
    }

    results = {}

    for name, (use_dropout, use_batch_norm) in configurations.items():
        print(f"\nTraining configuration: {name}")

        results[name] = train_model(
            use_dropout=use_dropout,
            use_batch_norm=use_batch_norm,
        )

        result = results[name]

        print(
            f"Final training MSE: "
            f"{result['train_losses'][-1]:.4f}"
        )
        print(
            f"Final validation MSE: "
            f"{result['validation_losses'][-1]:.4f}"
        )

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


def experiment_dropout_train_eval_behavior() -> None:
    """Demonstrate that dropout behaves differently in train and eval modes."""
    set_seed()

    dropout = nn.Dropout(p=0.5)
    inputs = torch.ones(12, 8)

    dropout.train()
    training_output = dropout(inputs)

    dropout.eval()
    evaluation_output = dropout(inputs)

    print("\nDropout train/eval behavior")
    print("Input:")
    print(inputs[0])

    print("\nTraining output:")
    print(training_output[0])

    print("\nEvaluation output:")
    print(evaluation_output[0])

    print(
        "\nTraining output contains zeros:",
        bool((training_output == 0).any()),
    )
    print(
        "Evaluation output equals input:",
        torch.equal(evaluation_output, inputs),
    )

    print(
        "Mean training activation:",
        training_output.mean().item(),
    )
    print(
        "Mean evaluation activation:",
        evaluation_output.mean().item(),
    )


def experiment_batch_norm_train_eval_behavior() -> None:
    """Inspect BatchNorm running statistics and output behavior."""
    set_seed()

    batch_norm = nn.BatchNorm1d(4)

    inputs = torch.randn(16, 4) * 3.0 + 5.0

    print("\nBatch normalization behavior")

    print(
        "Initial running mean:",
        batch_norm.running_mean.detach().clone(),
    )
    print(
        "Initial running variance:",
        batch_norm.running_var.detach().clone(),
    )

    batch_norm.train()
    training_output = batch_norm(inputs)

    print(
        "\nTraining output mean:",
        training_output.mean(dim=0),
    )
    print(
        "Training output variance:",
        training_output.var(dim=0, unbiased=False),
    )
    print(
        "Updated running mean:",
        batch_norm.running_mean.detach().clone(),
    )

    batch_norm.eval()
    evaluation_output = batch_norm(inputs)

    print(
        "\nEvaluation output mean:",
        evaluation_output.mean(dim=0),
    )
    print(
        "Evaluation output variance:",
        evaluation_output.var(dim=0, unbiased=False),
    )


def experiment_batch_size_constraint() -> None:
    """Show why BatchNorm1d needs multiple values per channel in training."""
    batch_norm = nn.BatchNorm1d(4)
    batch_norm.train()

    single_sample = torch.randn(1, 4)

    print("\nBatchNorm1d batch-size constraint")

    try:
        batch_norm(single_sample)
    except ValueError as error:
        print("Expected error for a single training sample:")
        print(error)

    batch_norm.eval()
    output = batch_norm(single_sample)

    print("\nSingle-sample inference succeeds:")
    print(output)


if __name__ == "__main__":
    experiment_regularization_comparison()
    experiment_dropout_train_eval_behavior()
    experiment_batch_norm_train_eval_behavior()
    experiment_batch_size_constraint()
