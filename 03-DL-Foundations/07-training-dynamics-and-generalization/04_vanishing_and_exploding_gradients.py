"""
Vanishing and Exploding Gradients
=================================

Experiments:
1. Measure gradient norms across layers in deep neural networks.
2. Compare sigmoid, tanh, and ReLU activation functions.
3. Investigate how network depth affects gradient flow.
4. Detect very small and excessively large gradients.
5. Demonstrate exploding gradients in a recurrent-style computation.
6. Apply gradient clipping and compare gradient norms.
7. Observe how gradient behavior affects parameter updates.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


class DeepMLP(nn.Module):
    def __init__(
        self,
        depth: int,
        width: int = 32,
        activation: str = "relu",
    ):
        super().__init__()

        activation_layers = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }

        if activation not in activation_layers:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        input_features = 16

        for _ in range(depth):
            layers.append(nn.Linear(input_features, width))
            layers.append(activation_layers[activation]())
            input_features = width

        layers.append(nn.Linear(input_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def measure_layer_gradients(model, x, y):
    """Return each trainable layer's gradient norm after backpropagation."""
    model.zero_grad(set_to_none=True)

    loss_function = nn.MSELoss()
    predictions = model(x)
    loss = loss_function(predictions, y)
    loss.backward()

    layer_names = []
    gradient_norms = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameter_gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.grad is not None
            ]

            if parameter_gradients:
                norm_squared = sum(
                    gradient.detach().norm().item() ** 2
                    for gradient in parameter_gradients
                )
                layer_names.append(name)
                gradient_norms.append(norm_squared ** 0.5)

    return loss.item(), layer_names, gradient_norms


def compare_depth_and_activation():
    section("1. Gradient Flow Across Depth and Activation Functions")

    torch.manual_seed(42)

    x = torch.randn(64, 16)
    y = torch.randn(64, 1)

    depths = [2, 5, 10, 20]
    activations = ["sigmoid", "tanh", "relu"]

    results = {}

    for activation in activations:
        for depth in depths:
            # Reset the seed so each experiment starts reproducibly.
            torch.manual_seed(42)

            model = DeepMLP(
                depth=depth,
                width=32,
                activation=activation,
            )

            loss, layer_names, gradient_norms = measure_layer_gradients(
                model, x, y
            )

            results[(activation, depth)] = {
                "loss": loss,
                "layer_names": layer_names,
                "gradient_norms": gradient_norms,
            }

            print(
                f"Activation={activation:7s} | "
                f"Depth={depth:2d} | "
                f"Loss={loss:.4f} | "
                f"First layer grad={gradient_norms[0]:.3e} | "
                f"Last layer grad={gradient_norms[-1]:.3e}"
            )

    return results


def plot_gradient_flow(results):
    section("2. Visualizing Gradient Flow")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for axis, activation in zip(axes, ["sigmoid", "tanh", "relu"]):
        for depth in [2, 5, 10, 20]:
            result = results[(activation, depth)]

            axis.plot(
                range(1, len(result["gradient_norms"]) + 1),
                result["gradient_norms"],
                marker="o",
                label=f"Depth={depth}",
            )

        axis.set_title(f"Activation: {activation}")
        axis.set_xlabel("Linear layer index")
        axis.set_ylabel("Gradient norm")
        axis.set_yscale("log")
        axis.legend()
        axis.grid(True)

    figure.tight_layout()
    plt.show()


def demonstrate_exploding_gradients():
    section("3. Exploding Gradients in a Recurrent-Style Computation")

    # Repeated multiplication illustrates how gradients can grow or shrink.
    # This is a controlled mathematical demonstration, not a complete RNN.
    initial_value = torch.tensor(1.0, requires_grad=True)

    for multiplier in [0.5, 1.0, 1.5, 2.0]:
        value = initial_value

        for _ in range(20):
            value = value * multiplier

        initial_value.grad = None
        value.backward()

        print(
            f"Multiplier={multiplier:.1f} | "
            f"Output={value.item():.3e} | "
            f"Gradient={initial_value.grad.item():.3e}"
        )


def train_with_clipping(clip_norm):
    """Compare gradient norms with and without clipping."""
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_function = nn.MSELoss()

    x = torch.randn(128, 10) * 10
    y = torch.randn(128, 1) * 100

    history = {
        "loss": [],
        "gradient_norm_before": [],
        "gradient_norm_after": [],
    }

    for _ in range(50):
        optimizer.zero_grad()

        predictions = model(x)
        loss = loss_function(predictions, y)

        if not torch.isfinite(loss):
            history["loss"].append(float("nan"))
            break

        loss.backward()

        parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.grad is not None
        ]

        norm_before = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=float("inf"),
        ).item()

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=clip_norm,
            )

        norm_after = torch.sqrt(
            sum(
                parameter.grad.detach().norm() ** 2
                for parameter in parameters
            )
        ).item()

        optimizer.step()

        history["loss"].append(loss.item())
        history["gradient_norm_before"].append(norm_before)
        history["gradient_norm_after"].append(norm_after)

    return history


def compare_gradient_clipping():
    section("4. Gradient Clipping")

    unclipped = train_with_clipping(clip_norm=None)
    clipped = train_with_clipping(clip_norm=1.0)

    print("Unclipped final loss:", unclipped["loss"][-1])
    print("Clipped final loss:", clipped["loss"][-1])

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(unclipped["gradient_norm_before"], label="No clipping")
    axes[0].plot(clipped["gradient_norm_before"], label="Before clipping")
    axes[0].plot(clipped["gradient_norm_after"], label="After clipping")
    axes[0].set_title("Gradient Norms")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Global gradient norm")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(unclipped["loss"], label="No clipping")
    axes[1].plot(clipped["loss"], label="Clipping at 1.0")
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend()
    axes[1].grid(True)

    figure.tight_layout()
    plt.show()


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    results = compare_depth_and_activation()
    plot_gradient_flow(results)

    demonstrate_exploding_gradients()
    compare_gradient_clipping()


if __name__ == "__main__":
    main()
