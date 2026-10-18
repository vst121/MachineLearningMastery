"""
Softmax Activation Function
===========================

Experiments:
1. Implement numerically stable Softmax.
2. Convert logits into probability distributions.
3. Verify that probabilities sum to one.
4. Examine how logits affect predicted probabilities.
5. Compare prediction confidence at different logit scales.
"""

import numpy as np
import matplotlib.pyplot as plt


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits into a numerically stable probability distribution."""
    logits = np.asarray(logits, dtype=np.float64)

    if logits.ndim != 1:
        raise ValueError("Expected a one-dimensional array of logits.")

    if logits.size == 0:
        raise ValueError("Logits must not be empty.")

    if not np.all(np.isfinite(logits)):
        raise ValueError("Logits must contain only finite values.")

    # Subtracting the maximum prevents exponential overflow.
    shifted_logits = logits - np.max(logits)
    exponentials = np.exp(shifted_logits)

    return exponentials / np.sum(exponentials)


def main() -> None:
    # Experiment 1: Convert logits into probabilities.
    logits = np.array([2.0, 1.0, 0.1])
    probabilities = softmax(logits)

    print("Logits:", logits)
    print("Softmax probabilities:", probabilities)
    print(f"Probability sum: {np.sum(probabilities):.6f}")

    predicted_class = int(np.argmax(probabilities))
    print("Predicted class index:", predicted_class)

    # Experiment 2: Compare different logit distributions.
    examples = {
        "Similar logits": np.array([1.0, 1.1, 0.9]),
        "Moderate difference": np.array([2.0, 1.0, 0.0]),
        "Large difference": np.array([5.0, 1.0, 0.0]),
    }

    labels = list(examples.keys())
    probability_matrix = np.array(
        [softmax(values) for values in examples.values()]
    )

    print("\nProbability distributions")
    for label, values, probs in zip(
        labels, examples.values(), probability_matrix
    ):
        print(f"{label}:")
        print("  Logits:", values)
        print("  Probabilities:", np.round(probs, 4))

    # Experiment 3: Visualize class probabilities.
    class_indices = np.arange(len(logits))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for index, (label, probs) in enumerate(
        zip(labels, probability_matrix)
    ):
        ax.bar(
            class_indices + (index - 1) * width,
            probs,
            width=width,
            label=label,
        )

    ax.set_title("Softmax Probabilities for Different Logits")
    ax.set_xlabel("Class Index")
    ax.set_ylabel("Probability")
    ax.set_xticks(class_indices)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.show()

    # Experiment 4: Verify numerical stability for large logits.
    large_logits = np.array([1000.0, 1001.0, 999.0])
    stable_probabilities = softmax(large_logits)

    print("\nNumerical stability")
    print("Large logits:", large_logits)
    print("Probabilities:", stable_probabilities)
    print("All finite:", np.all(np.isfinite(stable_probabilities)))
    print(f"Probability sum: {np.sum(stable_probabilities):.6f}")


if __name__ == "__main__":
    main()
