"""
Activation Function Comparison
==============================

Experiments:
1. Compare common activation functions visually.
2. Inspect outputs for representative inputs.
3. Compare output ranges and zero-centered behavior.
4. Summarize typical use cases.
"""

import numpy as np
import matplotlib.pyplot as plt


def step(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # Stable sigmoid evaluation for positive and negative inputs.
    result = np.empty_like(x)
    positive = x >= 0

    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))

    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)

    return result


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def leaky_relu(
    x: np.ndarray, alpha: float = 0.01
) -> np.ndarray:
    return np.where(x >= 0, x, alpha * x)


def prelu(
    x: np.ndarray, alpha: float = 0.25
) -> np.ndarray:
    # Fixed alpha for this comparison, not a trained parameter.
    return np.where(x >= 0, x, alpha * x)


def gelu(x: np.ndarray) -> np.ndarray:
    coefficient = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (
        1.0 + np.tanh(coefficient * (x + 0.044715 * x**3))
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)

    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("Expected a non-empty one-dimensional array.")

    if not np.all(np.isfinite(logits)):
        raise ValueError("Logits must contain only finite values.")

    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)

    return exponentials / np.sum(exponentials)


def main() -> None:
    # Experiment 1: Compare activation curves.
    x = np.linspace(-5, 5, 500)

    activations = {
        "Step": step(x),
        "Sigmoid": sigmoid(x),
        "Tanh": tanh(x),
        "ReLU": relu(x),
        "Leaky ReLU": leaky_relu(x),
        "PReLU (fixed alpha)": prelu(x),
        "GELU": gelu(x),
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    for name in ["Step", "Sigmoid", "Tanh"]:
        axes[0].plot(x, activations[name], label=name)

    axes[0].set_title("Bounded Activation Functions")
    axes[0].set_xlabel("Input")
    axes[0].set_ylabel("Output")
    axes[0].grid(True)
    axes[0].legend()

    for name in ["ReLU", "Leaky ReLU", "PReLU (fixed alpha)", "GELU"]:
        axes[1].plot(x, activations[name], label=name)

    axes[1].set_title("ReLU and Its Variants")
    axes[1].set_xlabel("Input")
    axes[1].set_ylabel("Output")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Experiment 2: Compare representative outputs.
    test_inputs = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

    print("Activation outputs")
    print("-" * 78)
    print(
        f"{'Input':>7} | {'Step':>7} | {'Sigmoid':>9} | "
        f"{'Tanh':>9} | {'ReLU':>7} | {'Leaky ReLU':>11} | {'GELU':>9}"
    )
    print("-" * 78)

    for value in test_inputs:
        scalar = np.array([value])

        print(
            f"{value:7.2f} | "
            f"{float(step(scalar)[0]):7.3f} | "
            f"{float(sigmoid(scalar)[0]):9.4f} | "
            f"{float(tanh(scalar)[0]):9.4f} | "
            f"{float(relu(scalar)[0]):7.3f} | "
            f"{float(leaky_relu(scalar)[0]):11.4f} | "
            f"{float(gelu(scalar)[0]):9.4f}"
        )

    # Experiment 3: Demonstrate Softmax on a multiclass output.
    logits = np.array([2.0, 1.0, 0.1])
    probabilities = softmax(logits)

    print("\nSoftmax example")
    for index, probability in enumerate(probabilities):
        print(f"Class {index}: {probability:.4f}")

    print(f"Probability sum: {probabilities.sum():.6f}")

    # Experiment 4: Compare practical use cases.
    use_cases = {
        "Step": "Perceptron fundamentals",
        "Sigmoid": "Binary classification output",
        "Tanh": "Selected hidden layers and recurrent networks",
        "ReLU": "Common hidden-layer default",
        "Leaky ReLU": "Hidden layers with a negative slope",
        "PReLU": "Models that learn the negative slope",
        "GELU": "Transformers and modern neural networks",
        "Softmax": "Mutually exclusive multiclass output",
    }

    print("\nTypical use cases")
    for function, use_case in use_cases.items():
        print(f"{function:12} -> {use_case}")


if __name__ == "__main__":
    main()
