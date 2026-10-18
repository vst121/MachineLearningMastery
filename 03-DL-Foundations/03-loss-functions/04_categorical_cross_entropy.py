"""
Categorical Cross-Entropy (CCE)
================================

Experiments:
1. Implement categorical cross-entropy using NumPy.
2. Calculate loss for individual multi-class predictions.
3. Compare correct and incorrect confident predictions.
4. Compare one-hot labels with integer class labels.
5. Visualize how the loss changes with the probability assigned
   to the true class.
"""

import numpy as np
import matplotlib.pyplot as plt


def categorical_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Calculate mean categorical cross-entropy.

    Args:
        y_true: One-hot encoded labels with shape (n_samples, n_classes).
        y_pred: Predicted class probabilities with the same shape.
        epsilon: Small value used to clip probabilities.

    Returns:
        Mean categorical cross-entropy across samples.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("Inputs must be 2D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.shape[0] == 0 or y_true.shape[1] < 2:
        raise ValueError("Inputs must contain samples and at least two classes.")

    if not np.all(np.isfinite(y_true)) or not np.all(
        np.isfinite(y_pred)
    ):
        raise ValueError("Inputs must contain only finite values.")

    if not np.all(np.isin(y_true, [0.0, 1.0])):
        raise ValueError("y_true must contain one-hot encoded labels.")

    if not np.all(np.sum(y_true, axis=1) == 1.0):
        raise ValueError("Each target row must contain exactly one true class.")

    if np.any((y_pred < 0.0) | (y_pred > 1.0)):
        raise ValueError("Predicted probabilities must be in [0, 1].")

    if not np.allclose(np.sum(y_pred, axis=1), 1.0, atol=1e-7):
        raise ValueError("Each predicted probability row must sum to 1.")

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be between 0 and 1.")

    # Clip probabilities to prevent log(0).
    probabilities = np.clip(y_pred, epsilon, 1.0)

    # Sum class-wise losses, then average across samples.
    sample_losses = -np.sum(y_true * np.log(probabilities), axis=1)

    return float(np.mean(sample_losses))


def categorical_cross_entropy_from_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Calculate CCE using integer class labels.

    Args:
        y_true: Integer class labels with shape (n_samples,).
        y_pred: Predicted probabilities with shape (n_samples, n_classes).
        epsilon: Small value used to clip probabilities.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 1 or y_pred.ndim != 2:
        raise ValueError("Expected 1D labels and 2D probability predictions.")

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("Number of labels and predictions must match.")

    if y_true.size == 0 or y_pred.shape[1] < 2:
        raise ValueError("Inputs must contain samples and at least two classes.")

    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("Class labels must be integers.")

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("Predictions must contain only finite values.")

    if np.any((y_pred < 0.0) | (y_pred > 1.0)):
        raise ValueError("Predicted probabilities must be in [0, 1].")

    if not np.allclose(np.sum(y_pred, axis=1), 1.0, atol=1e-7):
        raise ValueError("Each predicted probability row must sum to 1.")

    if np.any((y_true < 0) | (y_true >= y_pred.shape[1])):
        raise ValueError("Class labels are outside the valid class range.")

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be between 0 and 1.")

    probabilities = np.clip(y_pred, epsilon, 1.0)

    # Select the predicted probability assigned to each true class.
    true_class_probabilities = probabilities[
        np.arange(y_true.size), y_true
    ]

    return float(-np.mean(np.log(true_class_probabilities)))


def main() -> None:
    # Experiment 1: Basic multi-class classification
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    y_pred = np.array([
        [0.80, 0.10, 0.10],
        [0.10, 0.70, 0.20],
        [0.05, 0.15, 0.80],
    ])

    print("Experiment 1: Basic CCE")

    sample_losses = -np.sum(y_true * np.log(y_pred), axis=1)

    for index, loss in enumerate(sample_losses, start=1):
        print(f"Sample {index}: CCE = {loss:.4f}")

    print(f"Mean CCE: {categorical_cross_entropy(y_true, y_pred):.4f}")
    print()

    # Experiment 2: Correct versus incorrect confident predictions
    predictions_correct = np.array([
        [0.98, 0.01, 0.01],
        [0.01, 0.98, 0.01],
        [0.01, 0.01, 0.98],
    ])

    predictions_uncertain = np.array([
        [0.34, 0.33, 0.33],
        [0.33, 0.34, 0.33],
        [0.33, 0.33, 0.34],
    ])

    predictions_wrong = np.array([
        [0.01, 0.98, 0.01],
        [0.98, 0.01, 0.01],
        [0.01, 0.98, 0.01],
    ])

    print("Experiment 2: Prediction confidence")

    for name, predictions in [
        ("Correct and confident", predictions_correct),
        ("Uncertain", predictions_uncertain),
        ("Confident but wrong", predictions_wrong),
    ]:
        loss = categorical_cross_entropy(y_true, predictions)
        print(f"{name}: CCE = {loss:.4f}")

    print()

    # Experiment 3: One-hot labels versus integer labels
    integer_labels = np.array([0, 1, 2])

    one_hot_loss = categorical_cross_entropy(y_true, y_pred)
    integer_loss = categorical_cross_entropy_from_labels(
        integer_labels, y_pred
    )

    print("Experiment 3: Label representations")
    print(f"One-hot labels:     CCE = {one_hot_loss:.4f}")
    print(f"Integer class labels: CCE = {integer_loss:.4f}")
    print()

    # Experiment 4: Loss versus probability assigned to true class
    true_class_probabilities = np.linspace(0.01, 1.0, 200)
    losses = -np.log(true_class_probabilities)

    plt.figure(figsize=(9, 5))
    plt.plot(true_class_probabilities, losses)
    plt.xlabel("Probability assigned to the true class")
    plt.ylabel("Categorical cross-entropy")
    plt.title("CCE versus True-Class Probability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
