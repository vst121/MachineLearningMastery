"""
ReLU and Its Variants
=====================

Experiments:
1. Implement ReLU, Leaky ReLU, PReLU, and GELU.
2. Compare their activation curves.
3. Examine how each function handles negative inputs.
4. Demonstrate the dying ReLU phenomenon conceptually.
"""

import numpy as np
import matplotlib.pyplot as plt


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit."""
    return np.maximum(0.0, x)


def leaky_relu(
    x: np.ndarray, alpha: float = 0.01
) -> np.ndarray:
    """Leaky ReLU with a small negative slope."""
    return np.where(x >= 0, x, alpha * x)


def prelu(
    x: np.ndarray, alpha: float = 0.25
) -> np.ndarray:
    """PReLU with a configurable negative slope.

    In a neural network, alpha is typically a learnable parameter.
    Here it is supplied explicitly for experimentation.
    """
    return np.where(x >= 0, x, alpha * x)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU using its tanh-based approximation."""
    coefficient = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (
        1.0 + np.tanh(coefficient * (x + 0.044715 * x**3))
    )


def main() -> None:
    # Experiment 1: Compare activation curves.
    x = np.linspace(-5, 5, 500)

    activations = {
        "ReLU": relu(x),
        "Leaky ReLU (alpha=0.01)": leaky_relu(x),
        "PReLU (alpha=0.25)": prelu(x),
        "GELU": gelu(x),
    }

    plt.figure(figsize=(10, 6))

    for name, output in activations.items():
        plt.plot(x, output, label=name)

    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.axvline(0, linestyle="--", linewidth=0.8)
    plt.title("ReLU and Its Variants")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Experiment 2: Compare outputs for representative inputs.
    test_inputs = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])

    print(
        f"{'Input':>8} | {'ReLU':>10} | {'Leaky ReLU':>12} | "
        f"{'PReLU':>10} | {'GELU':>10}"
    )
    print("-" * 64)

    for value in test_inputs:
        scalar = np.array(value)

        print(
            f"{value:8.2f} | "
            f"{float(relu(scalar)):10.4f} | "
            f"{float(leaky_relu(scalar)):12.4f} | "
            f"{float(prelu(scalar)):10.4f} | "
            f"{float(gelu(scalar)):10.4f}"
        )

    # Experiment 3: Illustrate inactive ReLU neurons.
    negative_inputs = np.array([-4.0, -3.0, -2.0, -1.0])

    print("\nDying ReLU concept")
    print("Negative inputs:", negative_inputs)
    print("ReLU outputs:   ", relu(negative_inputs))
    print("Leaky ReLU:     ", leaky_relu(negative_inputs))

    print(
        "\nA ReLU neuron that receives negative pre-activation values "
        "outputs zero. If its parameters keep it inactive across "
        "training examples, it may stop learning through that path."
    )


if __name__ == "__main__":
    main()
