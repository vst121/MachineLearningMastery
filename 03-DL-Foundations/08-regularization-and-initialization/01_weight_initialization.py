"""
Weight Initialization in Deep Neural Networks
==============================================

Experiments:
1. Compare default, Xavier, and He initialization.
2. Measure activation variance across network depth.
3. Compare initialization strategies with tanh and ReLU.
4. Measure gradient norms to investigate gradient propagation.
5. Observe how initialization affects early training behavior.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


SEED = 42
INPUT_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 10
DEPTHS = [2, 5, 10, 20]


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducible experiments."""
    torch.manual_seed(seed)


class DeepMLP(nn.Module):
    """A configurable multilayer perceptron for initialization experiments."""

    def __init__(
        self,
        depth: int,
        activation_name: str,
        initialization: str,
    ) -> None:
        super().__init__()

        layers = []
        in_features = INPUT_DIM

        for _ in range(depth):
            linear = nn.Linear(in_features, HIDDEN_DIM)
            layers.append(linear)

            if activation_name == "relu":
                layers.append(nn.ReLU())
            elif activation_name == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(
                    f"Unsupported activation: {activation_name}"
                )

            in_features = HIDDEN_DIM

        layers.append(nn.Linear(in_features, OUTPUT_DIM))
        self.network = nn.Sequential(*layers)

        self.initialize_weights(initialization, activation_name)

    def initialize_weights(
        self,
        initialization: str,
        activation_name: str,
    ) -> None:
        """Apply the selected initialization to linear layers."""
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue

            if initialization == "default":
                # Preserve PyTorch's default initialization.
                continue

            if initialization == "xavier":
                nn.init.xavier_uniform_(module.weight)

            elif initialization == "he":
                if activation_name == "relu":
                    nn.init.kaiming_normal_(
                        module.weight,
                        nonlinearity="relu",
                    )
                else:
                    # Kaiming initialization is primarily designed
                    # for ReLU-family activations. This option allows
                    # us to observe its behavior with tanh as well.
                    nn.init.kaiming_normal_(
                        module.weight,
                        nonlinearity="relu",
                    )
            else:
                raise ValueError(
                    f"Unsupported initialization: {initialization}"
                )

            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def measure_activation_statistics(
    model: DeepMLP,
    inputs: torch.Tensor,
) -> list[float]:
    """Measure activation variance after each hidden linear layer."""
    variances = []
    activations = inputs

    for layer in model.network:
        activations = layer(activations)

        if isinstance(layer, (nn.ReLU, nn.Tanh)):
            variances.append(
                activations.detach().var(unbiased=False).item()
            )

    return variances


def measure_gradient_norms(
    model: DeepMLP,
    inputs: torch.Tensor,
) -> list[float]:
    """Measure gradient norms for each hidden linear layer."""
    targets = torch.randint(
        0,
        OUTPUT_DIM,
        (inputs.shape[0],),
    )

    criterion = nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)

    predictions = model(inputs)
    loss = criterion(predictions, targets)
    loss.backward()

    gradient_norms = []

    for layer in model.network:
        if isinstance(layer, nn.Linear):
            if layer.weight.grad is not None:
                gradient_norms.append(
                    layer.weight.grad.detach().norm().item()
                )

    return gradient_norms


def experiment_initialization_comparison() -> None:
    """Compare activation variance and gradient norms."""
    set_seed()

    inputs = torch.randn(256, INPUT_DIM)
    initializations = ["default", "xavier", "he"]
    activations = ["relu", "tanh"]

    results = defaultdict(dict)

    for activation_name in activations:
        for initialization in initializations:
            model = DeepMLP(
                depth=10,
                activation_name=activation_name,
                initialization=initialization,
            )

            activation_variances = measure_activation_statistics(
                model,
                inputs,
            )
            gradient_norms = measure_gradient_norms(
                model,
                inputs,
            )

            results[(activation_name, initialization)] = {
                "activation_variances": activation_variances,
                "gradient_norms": gradient_norms,
            }

            print(
                f"\nActivation: {activation_name:5s} | "
                f"Initialization: {initialization}"
            )
            print(
                "Activation variances:",
                [round(value, 5) for value in activation_variances],
            )
            print(
                "Gradient norms:",
                [round(value, 5) for value in gradient_norms],
            )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for initialization in initializations:
        key = ("relu", initialization)
        axes[0, 0].plot(
            range(1, 11),
            results[key]["activation_variances"],
            marker="o",
            label=initialization,
        )

        axes[1, 0].plot(
            range(1, len(results[key]["gradient_norms"]) + 1),
            results[key]["gradient_norms"],
            marker="o",
            label=initialization,
        )

        key = ("tanh", initialization)
        axes[0, 1].plot(
            range(1, 11),
            results[key]["activation_variances"],
            marker="o",
            label=initialization,
        )

        axes[1, 1].plot(
            range(1, len(results[key]["gradient_norms"]) + 1),
            results[key]["gradient_norms"],
            marker="o",
            label=initialization,
        )

    axes[0, 0].set_title("ReLU: Activation Variance")
    axes[0, 1].set_title("Tanh: Activation Variance")
    axes[1, 0].set_title("ReLU: Weight Gradient Norms")
    axes[1, 1].set_title("Tanh: Weight Gradient Norms")

    for axis in axes.flat:
        axis.set_xlabel("Layer")
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    plt.show()


def experiment_depth_comparison() -> None:
    """Observe how depth affects activation variance."""
    set_seed()

    inputs = torch.randn(256, INPUT_DIM)
    initialization = "he"
    activation_name = "relu"

    mean_variances = []

    for depth in DEPTHS:
        model = DeepMLP(
            depth=depth,
            activation_name=activation_name,
            initialization=initialization,
        )

        variances = measure_activation_statistics(model, inputs)
        mean_variances.append(variances)

        print(
            f"Depth {depth:2d}: "
            f"first variance={variances[0]:.5f}, "
            f"last variance={variances[-1]:.5f}"
        )

    plt.figure(figsize=(9, 5))

    for depth, variances in zip(DEPTHS, mean_variances):
        plt.plot(
            range(1, depth + 1),
            variances,
            marker="o",
            label=f"{depth} hidden layers",
        )

    plt.title("Activation Variance Across Different Network Depths")
    plt.xlabel("Hidden Layer")
    plt.ylabel("Activation Variance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def experiment_initial_training_behavior() -> None:
    """Compare early loss behavior under different initializations."""
    set_seed()

    inputs = torch.randn(512, INPUT_DIM)
    targets = torch.randint(0, OUTPUT_DIM, (512,))

    initializations = ["default", "xavier", "he"]
    learning_rate = 0.01
    steps = 100

    plt.figure(figsize=(9, 5))

    for initialization in initializations:
        set_seed()

        model = DeepMLP(
            depth=10,
            activation_name="relu",
            initialization=initialization,
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
        )
        criterion = nn.CrossEntropyLoss()
        losses = []

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)

            predictions = model(inputs)
            loss = criterion(predictions, targets)

            if not torch.isfinite(loss):
                print(
                    f"{initialization}: non-finite loss encountered."
                )
                break

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        plt.plot(
            range(1, len(losses) + 1),
            losses,
            label=initialization,
        )

        if losses:
            print(
                f"{initialization:7s}: "
                f"initial loss={losses[0]:.4f}, "
                f"final loss={losses[-1]:.4f}"
            )

    plt.title("Early Training Behavior by Initialization")
    plt.xlabel("Optimization Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    experiment_initialization_comparison()
    experiment_depth_comparison()
    experiment_initial_training_behavior()
