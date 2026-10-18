"""
Loss Function Comparison
========================

Experiments:
1. Compare MSE, MAE, and Huber Loss on regression data.
2. Investigate how outliers affect regression losses.
3. Compare Binary Cross-Entropy for different predictions.
4. Compare Categorical Cross-Entropy for multi-class predictions.
5. Visualize regression loss functions across a range of residuals.
6. Summarize the characteristics and use cases of each loss.
"""

import numpy as np
import matplotlib.pyplot as plt


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def huber_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.0,
) -> float:
    """Calculate Huber Loss."""
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)

    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a positive finite number.")

    residuals = np.abs(y_true - y_pred)

    losses = np.where(
        residuals <= delta,
        0.5 * residuals**2,
        delta * (residuals - 0.5 * delta),
    )

    return float(np.mean(losses))


def binary_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Calculate Binary Cross-Entropy from predicted probabilities."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Inputs must not be empty.")

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Inputs must contain only finite values.")

    if not np.all(np.isin(y_true, [0.0, 1.0])):
        raise ValueError("y_true must contain only binary labels.")

    if np.any((y_pred < 0.0) | (y_pred > 1.0)):
        raise ValueError("Predicted probabilities must be in [0, 1].")

    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5.")

    probabilities = np.clip(y_pred, epsilon, 1.0 - epsilon)

    losses = -(
        y_true * np.log(probabilities)
        + (1.0 - y_true) * np.log(1.0 - probabilities)
    )

    return float(np.mean(losses))


def categorical_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Calculate Categorical Cross-Entropy for one-hot labels."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("Inputs must be 2D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.shape[0] == 0 or y_true.shape[1] < 2:
        raise ValueError("Inputs need samples and at least two classes.")

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Inputs must contain only finite values.")

    if not np.all(np.isin(y_true, [0.0, 1.0])):
        raise ValueError("Targets must be one-hot encoded.")

    if not np.all(np.sum(y_true, axis=1) == 1.0):
        raise ValueError("Each target row must contain exactly one true class.")

    if np.any((y_pred < 0.0) | (y_pred > 1.0)):
        raise ValueError("Predicted probabilities must be in [0, 1].")

    if not np.allclose(np.sum(y_pred, axis=1), 1.0, atol=1e-7):
        raise ValueError("Each probability row must sum to 1.")

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be between 0 and 1.")

    probabilities = np.clip(y_pred, epsilon, 1.0)
    losses = -np.sum(y_true * np.log(probabilities), axis=1)

    return float(np.mean(losses))


def _validate_regression_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert regression targets and predictions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Inputs must not be empty.")

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Inputs must contain only finite values.")

    return y_true, y_pred


def compare_regression_losses(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate multiple regression losses for the same predictions."""
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Huber (delta=1)": huber_loss(y_true, y_pred, delta=1.0),
    }


def main() -> None:
    # Experiment 1: Compare regression losses
    y_true = np.array([10.0, 12.0, 13.0, 15.0])
    y_pred = np.array([11.0, 11.0, 14.0, 14.0])

    print("Experiment 1: Regression loss comparison")
    for name, loss in compare_regression_losses(y_true, y_pred).items():
        print(f"{name}: {loss:.4f}")

    print()

    # Experiment 2: Investigate the effect of an outlier
    y_pred_outlier = np.array([11.0, 11.0, 14.0, 30.0])

    print("Experiment 2: Effect of an outlier")

    for label, predictions in [
        ("Normal predictions", y_pred),
        ("With outlier", y_pred_outlier),
    ]:
        print(f"{label}:")
        for name, loss in compare_regression_losses(
            y_true, predictions
        ).items():
            print(f"  {name}: {loss:.4f}")

    print()

    # Experiment 3: Binary Cross-Entropy
    binary_targets = np.array([1, 0, 1, 0])

    binary_predictions = {
        "Good predictions": np.array([0.95, 0.05, 0.90, 0.10]),
        "Uncertain predictions": np.array([0.55, 0.45, 0.60, 0.40]),
        "Confident but wrong": np.array([0.05, 0.95, 0.10, 0.90]),
    }

    print("Experiment 3: Binary classification")

    for label, predictions in binary_predictions.items():
        loss = binary_cross_entropy(binary_targets, predictions)
        print(f"{label}: BCE = {loss:.4f}")

    print()

    # Experiment 4: Categorical Cross-Entropy
    class_targets = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    class_predictions = {
        "Good predictions": np.array([
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.05, 0.90],
        ]),
        "Uncertain predictions": np.array([
            [0.34, 0.33, 0.33],
            [0.33, 0.34, 0.33],
            [0.33, 0.33, 0.34],
        ]),
        "Confident but wrong": np.array([
            [0.05, 0.90, 0.05],
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
        ]),
    }

    print("Experiment 4: Multi-class classification")

    for label, predictions in class_predictions.items():
        loss = categorical_cross_entropy(class_targets, predictions)
        print(f"{label}: CCE = {loss:.4f}")

    print()

    # Experiment 5: Visualize regression losses across residuals
    residuals = np.linspace(-5, 5, 400)

    mse_values = residuals**2
    mae_values = np.abs(residuals)

    delta = 1.0
    huber_values = np.where(
        np.abs(residuals) <= delta,
        0.5 * residuals**2,
        delta * (np.abs(residuals) - 0.5 * delta),
    )

    plt.figure(figsize=(9, 5))
    plt.plot(residuals, mse_values, label="MSE (squared error)")
    plt.plot(residuals, mae_values, label="MAE (absolute error)")
    plt.plot(residuals, huber_values, label="Huber Loss (delta=1)")
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Loss for one prediction")
    plt.title("Comparison of Regression Loss Functions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Experiment 6: Summary
    print("\nLoss Function Summary")
    print("-" * 72)
    print(f"{'Loss':<28}{'Typical task':<25}{'Key characteristic'}")
    print("-" * 72)
    print(f"{'MSE':<28}{'Regression':<25}{'Strongly penalizes large errors'}")
    print(f"{'MAE':<28}{'Regression':<25}{'Linear penalty for errors'}")
    print(f"{'Huber Loss':<28}{'Regression':<25}{'Quadratic then linear'}")
    print(f"{'Binary Cross-Entropy':<28}{'Binary classification':<25}{'Probability-based loss'}")
    print(f"{'Categorical Cross-Entropy':<28}{'Multi-class classification':<25}{'True-class probability'}")


if __name__ == "__main__":
    main()
