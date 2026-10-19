"""
Convergence and Learning Dynamics
=================================

Experiments:
1. Compare low, moderate, and high learning rates.
2. Visualize loss curves during gradient-based optimization.
3. Observe slow convergence, stable convergence, oscillation, and divergence.
4. Track gradient norms and parameter updates.
5. Compare convergence speed and final loss.
6. Demonstrate why a decreasing training loss does not guarantee generalization.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def create_data():
    """Create a reproducible linear regression dataset."""
    generator = torch.Generator().manual_seed(42)

    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    noise = 0.15 * torch.randn(
        x.shape,
        generator=generator,
    )

    y = 3.0 * x + 1.0 + noise

    return x, y


def train_with_learning_rate(
    x,
    y,
    learning_rate: float,
    epochs: int = 100,
):
    """Train the same model architecture under a chosen learning rate."""
    torch.manual_seed(7)

    model = RegressionModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    history = {
        "loss": [],
        "gradient_norm": [],
        "parameter_update_norm": [],
    }

    previous_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        predictions = model(x)
        loss = loss_function(predictions, y)

        if not torch.isfinite(loss):
            print(
                f"Learning rate {learning_rate}: "
                f"non-finite loss at epoch {epoch + 1}."
            )
            break

        loss.backward()

        gradient_norm_squared = sum(
            parameter.grad.detach().norm().item() ** 2
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        gradient_norm = gradient_norm_squared ** 0.5

        optimizer.step()

        update_norm_squared = sum(
            (
                parameter.detach() - previous
            ).norm().item() ** 2
            for parameter, previous in zip(
                model.parameters(),
                previous_parameters,
            )
        )
        update_norm = update_norm_squared ** 0.5

        previous_parameters = [
            parameter.detach().clone()
            for parameter in model.parameters()
        ]

        history["loss"].append(loss.item())
        history["gradient_norm"].append(gradient_norm)
        history["parameter_update_norm"].append(update_norm)

        if not all(
            torch.isfinite(parameter).all()
            for parameter in model.parameters()
        ):
            print(
                f"Learning rate {learning_rate}: "
                f"non-finite parameters after epoch {epoch + 1}."
            )
            break

    return model, history


def plot_loss_curves(histories):
    section("2. Loss Curves")

    figure, axis = plt.subplots(figsize=(10, 5))

    for learning_rate, history in histories.items():
        if history["loss"]:
            axis.plot(
                range(1, len(history["loss"]) + 1),
                history["loss"],
                label=f"Learning rate={learning_rate}",
            )

    axis.set_title("Convergence Under Different Learning Rates")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean Squared Error")
    axis.set_yscale("log")
    axis.legend()
    axis.grid(True)
    figure.tight_layout()
    plt.show()


def plot_gradient_and_update_norms(histories):
    section("3. Gradient and Parameter Update Dynamics")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for learning_rate, history in histories.items():
        epochs = range(1, len(history["gradient_norm"]) + 1)

        axes[0].plot(
            epochs,
            history["gradient_norm"],
            label=f"LR={learning_rate}",
        )
        axes[1].plot(
            epochs,
            history["parameter_update_norm"],
            label=f"LR={learning_rate}",
        )

    axes[0].set_title("Gradient Norm")
    axes[1].set_title("Parameter Update Norm")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Norm")
        axis.legend()
        axis.grid(True)

    figure.tight_layout()
    plt.show()


def compare_results(histories):
    section("4. Final Comparison")

    for learning_rate, history in histories.items():
        if not history["loss"]:
            print(f"LR={learning_rate}: no valid training steps")
            continue

        initial_loss = history["loss"][0]
        final_loss = history["loss"][-1]
        minimum_loss = min(history["loss"])

        print(
            f"LR={learning_rate:<7} | "
            f"Steps={len(history['loss']):3d} | "
            f"First loss={initial_loss:.6f} | "
            f"Final loss={final_loss:.6f} | "
            f"Minimum loss={minimum_loss:.6f}"
        )


def demonstrate_plateau():
    section("5. Learning Rate and a Plateau")

    x, y = create_data()

    # A very small learning rate makes progress slow over a fixed budget.
    _, small_lr_history = train_with_learning_rate(
        x,
        y,
        learning_rate=0.001,
        epochs=100,
    )

    # A moderate learning rate typically converges much faster here.
    _, moderate_lr_history = train_with_learning_rate(
        x,
        y,
        learning_rate=0.05,
        epochs=100,
    )

    print(
        "Small LR final loss:",
        f"{small_lr_history['loss'][-1]:.6f}",
    )
    print(
        "Moderate LR final loss:",
        f"{moderate_lr_history['loss'][-1]:.6f}",
    )

    print(
        "\nA plateau means progress has slowed. It does not always mean "
        "the model has reached a good solution."
    )


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    section("1. Learning Rate Experiments")

    x, y = create_data()

    learning_rates = [
        0.001,  # Slow progress
        0.05,   # Moderate learning rate
        0.5,    # Aggressive learning rate
        1.0,    # May oscillate or diverge
    ]

    histories = {}

    for learning_rate in learning_rates:
        print(f"\nTraining with learning rate {learning_rate}")

        _, history = train_with_learning_rate(
            x,
            y,
            learning_rate=learning_rate,
            epochs=100,
        )

        histories[learning_rate] = history

    plot_loss_curves(histories)
    plot_gradient_and_update_norms(histories)
    compare_results(histories)
    demonstrate_plateau()


if __name__ == "__main__":
    main()
