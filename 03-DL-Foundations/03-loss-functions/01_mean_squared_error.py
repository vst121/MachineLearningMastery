"""
Mean Squared Error (MSE)
========================

Experiments:
1. Implement MSE using NumPy.
2. Calculate squared errors for individual predictions.
3. Compare MSE with Mean Absolute Error (MAE).
4. Investigate the impact of an outlier.
5. Visualize how MSE penalizes prediction errors.
"""

import numpy as np
import matplotlib.pyplot as plt


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate the mean squared error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Inputs must not be empty.")

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Inputs must contain only finite values.")

    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate the mean absolute error for comparison."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Inputs must not be empty.")

    return float(np.mean(np.abs(y_true - y_pred)))


def main() -> None:
    # Experiment 1: Calculate MSE for a simple regression example.
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    errors = y_true - y_pred
    squared_errors = errors**2

    print("Experiment 1: Individual squared errors")
    print("True values:    ", y_true)
    print("Predictions:    ", y_pred)
    print("Errors:         ", errors)
    print("Squared errors: ", squared_errors)
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")

    # Experiment 2: Compare MSE and MAE.
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print("\nExperiment 2: MSE vs. MAE")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Experiment 3: Investigate the effect of an outlier.
    true_values = np.array([10.0, 12.0, 13.0, 15.0])

    ordinary_predictions = np.array([11.0, 11.0, 14.0, 14.0])
    outlier_predictions = np.array([11.0, 11.0, 14.0, 30.0])

    print("\nExperiment 3: Outlier impact")

    for name, predictions in [
        ("Ordinary predictions", ordinary_predictions),
        ("Prediction with outlier", outlier_predictions),
    ]:
        print(f"\n{name}")
        print(f"MSE: {mean_squared_error(true_values, predictions):.4f}")
        print(f"MAE: {mean_absolute_error(true_values, predictions):.4f}")

    # Experiment 4: Visualize squared error as a function of residual.
    residuals = np.linspace(-5, 5, 300)
    squared_error = residuals**2
    absolute_error = np.abs(residuals)

    plt.figure(figsize=(9, 5))
    plt.plot(residuals, squared_error, label="Squared error")
    plt.plot(residuals, absolute_error, label="Absolute error")
    plt.xlabel("Residual (true value - prediction)")
    plt.ylabel("Loss for one sample")
    plt.title("Squared Error vs. Absolute Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
