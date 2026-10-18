
"""
Step and Sigmoid Activation Functions
=====================================

Explore two fundamental activation functions used in neural networks.

Experiments:
1. Implement step and sigmoid functions.
2. Visualize their output curves.
3. Compare their behavior around the decision threshold.
4. Examine sigmoid saturation for extreme inputs.
"""

import numpy as np
import matplotlib.pyplot as plt


def step(x: np.ndarray) -> np.ndarray:
    """Return 1 where x >= 0, otherwise return 0."""
    return np.where(x >= 0, 1.0, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the sigmoid activation function."""
    # Clipping avoids overflow in exp for extreme input values.
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    # Experiment 1: Compare activation curves.
    x = np.linspace(-10, 10, 500)

    step_output = step(x)
    sigmoid_output = sigmoid(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x, step_output, label="Step")
    axes[0].set_title("Step Activation")
    axes[0].set_xlabel("Input")
    axes[0].set_ylabel("Output")
    axes[0].set_yticks([0, 1])
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x, sigmoid_output, label="Sigmoid")
    axes[1].axhline(0.5, linestyle="--", label="Output = 0.5")
    axes[1].set_title("Sigmoid Activation")
    axes[1].set_xlabel("Input")
    axes[1].set_ylabel("Output")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Experiment 2: Inspect representative inputs.
    test_inputs = np.array([-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0])

    print("Input | Step | Sigmoid")
    print("-" *  thirty if False else "-" * 25)

    for value in test_inputs:
        step_value = step(np.array(value))
        sigmoid_value = sigmoid(np.array(value))

        print(
            f"{value:5.1f} | "
            f"{float(step_value):4.1f} | "
            f"{float(sigmoid_value):.4f}"
        )

    # Experiment 3: Examine sigmoid saturation.
    extreme_inputs = np.array([-100.0, -20.0, 0.0, 20.0, 100.0])
    extreme_outputs = sigmoid(extreme_inputs)

    print("\nSigmoid saturation")
    for value, output in zip(extreme_inputs, extreme_outputs):
        print(f"Input: {value:6.1f} -> Output: {output:.8f}")


if __name__ == "__main__":
    main()
