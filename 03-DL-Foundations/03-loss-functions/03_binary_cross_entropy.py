"""
Binary Cross-Entropy (BCE)
==========================

Experiments:
1. Implement BCE using NumPy with numerical stability.
2. Calculate binary cross-entropy for individual predictions.
3. Investigate how predicted probabilities affect the loss.
4. Compare correct and incorrect confident predictions.
5. Visualize BCE across predicted probabilities for both classes.
"""

import numpy as np
import matplotlib.pyplot as plt


def binary_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Calculate mean binary cross-entropy from predicted probabilities."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Inputs must not be empty.")

    if not np.all(np.isfinite(y_true)) or not np.all(
        np.isfinite(y_pred)
    ):
        raise ValueError("Inputs must contain only finite values.")

    if not np.all(np.isin(y_true, [0.0, 1.0])):
        raise ValueError("y_true must contain only binary labels: 0 or 1.")

    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5.")

    if np.any((y_pred < 0.0) | (y_pred > 1.0)):
        raise ValueError("Predicted probabilities must be in [0, 1].")

    # Clip probabilities to avoid log(0).
    probabilities = np.clip(y_pred, epsilon, 1.0 - epsilon)

    losses = -(
        y_true * np.log(probabilities)
        + (1.0 - y_true) * np.log(1.0 - probabilities)
    )

    return float(np.mean(losses))


def binary_cross_entropy_per_sample(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return BCE for each prediction."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Inputs must not be empty.")

    if not np.all(np.isfinite(y_true)) or not np.all(
        np.isfinite(y_pred)
    ):
        raise ValueError("Inputs must contain only finite values.")

    if not np.all(np.isin(y_true, [0.0, 1.0])):
        raise ValueError("y_true must contain only binary labels: 0 or 1.")

    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5.")

    if np.any((y_pred < 0.0) | (y_pred > 1.0)):
        raise ValueError("Predicted probabilities must be in [0, 1].")

    probabilities = np.clip(y_pred, epsilon, 1.0 - epsilon)

    return -(
        y_true * np.log(probabilities)
        + (1.0 - y_true) * np.log(1.0 - probabilities)
    )


def main() -> None:
    # Experiment 1: Basic BCE calculation
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.2, 0.8, 0.1])

    losses = binary_cross_entropy_per_sample(y_true, y_pred)

    print("Experiment 1: Basic BCE")
    for index, (label, probability, loss) in enumerate(
        zip(y_true, y_pred, losses), start=1
    ):
        print(
            f"Sample {index}: true={label}, "
            f"predicted_probability={probability:.2f}, "
            f"BCE={loss:.4f}"
        )

    print(f"Mean BCE: {binary_cross_entropy(y_true, y_pred):.4f}")
    print()

    # Experiment 2: Predicted probability and loss
    print("Experiment 2: Probability and loss")

    for true_label in [0, 1]:
        probabilities = np.array([0.1, 0.5, 0.9])
        labels = np.full(probabilities.shape, true_label)

        sample_losses = binary_cross_entropy_per_sample(
            labels, probabilities
        )

        print(f"True label = {true_label}")
        for probability, loss in zip(probabilities, sample_losses):
            print(f"  p={probability:.1f} -> loss={loss:.4f}")

    print()

    # Experiment 3: Correct versus incorrect confident predictions
    y_true_confident = np.array([1, 0, 1, 0])

    predictions_correct = np.array([0.99, 0.01, 0.95, 0.05])
    predictions_uncertain = np.array([0.55, 0.45, 0.60, 0.40])
    predictions_wrong = np.array([0.01, 0.99, 0.05, 0.95])

    print("Experiment 3: Prediction confidence")

    for name, predictions in [
        ("Correct and confident", predictions_correct),
        ("Uncertain", predictions_uncertain),
        ("Confident but wrong", predictions_wrong),
    ]:
        loss = binary_cross_entropy(y_true_confident, predictions)
        print(f"{name}: BCE = {loss:.4f}")

    print()

    # Experiment 4: Visualize BCE for each possible true label
    epsilon = 1e-6
    probabilities = np.linspace(epsilon, 1.0 - epsilon, 500)

    loss_for_zero = -np.log(1.0 - probabilities)
    loss_for_one = -np.log(probabilities)

    plt.figure(figsize=(9, 5))
    plt.plot(probabilities, loss_for_zero, label="True label = 0")
    plt.plot(probabilities, loss_for_one, label="True label = 1")
    plt.xlabel("Predicted probability of class 1")
    plt.ylabel("Binary cross-entropy")
    plt.title("Binary Cross-Entropy versus Predicted Probability")
    plt.ylim(0, 8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
