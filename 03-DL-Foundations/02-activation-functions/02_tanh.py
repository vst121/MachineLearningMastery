"""
Hyperbolic Tangent (tanh) Activation
====================================

Experiments:
1. Implement tanh using NumPy.
2. Visualize its activation curve.
3. Compare tanh with sigmoid.
4. Examine output symmetry and saturation.
"""

import numpy as np
import matplotlib.pyplot as plt


def tanh(x: np.ndarray) -> np.ndarray:
    """Apply the hyperbolic tangent activation function."""
    return np.tanh(x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the sigmoid activation function."""
    # Clipping prevents overflow for extreme negative inputs.
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    # Experiment 1: Visualize tanh and sigmoid.
    x = np.linspace(-10, 10, 500)

    tanh_output = tanh(x)
    sigmoid_output = sigmoid(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x, tanh_output, label="tanh")
    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].axvline(0, linestyle="--", linewidth=1)
    axes[0].set_title("Hyperbolic Tangent")
    axes[0].set_xlabel("Input")
    axes[0].set_ylabel("Output")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x, sigmoid_output, label="Sigmoid")
    axes[1].plot(x, tanh_output, label="tanh")
    axes[1].set_title("Sigmoid vs. tanh")
    axes[1].set_xlabel("Input")
    axes[1].set_ylabel("Output")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Experiment 2: Compare representative outputs.
    test_inputs = np.array([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])

    print(f"{'Input':>8} | {'tanh':>12} | {'Sigmoid':>12}")
    print("-" * 38)

    for value in test_inputs:
        tanh_value = tanh(np.array(value))
        sigmoid_value = sigmoid(np.array(value))

        print(
            f"{value:8.2f} | "
            f"{float(tanh_value):12.6f} | "
            f"{float(sigmoid_value):12.6f}"
        )

    # Experiment 3: Verify symmetry around the origin.
    positive_inputs = np.array([0.5, 1.0, 2.0, 5.0])

    positive_outputs = tanh(positive_inputs)
    negative_outputs = tanh(-positive_inputs)

    print("\nSymmetry check")
    print("tanh(-x) = -tanh(x)")

    for value, positive, negative in zip(
        positive_inputs, positive_outputs, negative_outputs
    ):
        print(
            f"x = {value:4.1f} | "
            f"tanh(x) = {positive: .6f} | "
            f"tanh(-x) = {negative: .6f}"
        )


if __name__ == "__main__":
    main()
