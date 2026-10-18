"""
Mean Absolute Error (MAE)
=========================

Experiments:
1. Implement MAE using NumPy.
2. Calculate absolute errors for individual predictions.
3. Compare MAE with Mean Squared Error (MSE).
4. Investigate the impact of an outlier.
5. Visualize how MAE penalizes prediction errors.
"""

import numpy as np
import matplotlib.pyplot as plt


def mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate Mean Absolute Error (MAE)."""
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

    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate Mean Squared Error (MSE) for comparison."""
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

    residuals = y_true - y_pred
    return float(np.mean(residuals**2))


def main() -> None:
    # Experiment 1: Basic MAE calculation
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    residuals = y_true - y_pred
    absolute_errors = np.abs(residuals)

    print("Experiment 1: Basic MAE")
    print(f"True values:      {y_true}")
    print(f"Predicted values: {y_pred}")
    print(f"Residuals:        {residuals}")
    print(f"Absolute errors:  {absolute_errors}")
    print(f"MAE:              {mean_absolute_error(y_true, y_pred):.4f}")
    print()

    # Experiment 2: MAE versus MSE
    print("Experiment 2: MAE versus MSE")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print()

    # Experiment 3: Effect of an outlier
    y_true_outlier = np.array([10.0, 12.0, 13.0, 15.0])

    y_pred_normal = np.array([11.0, 11.0, 14.0, 14.0])
    y_pred_outlier = np.array([11.0, 11.0, 14.0, 30.0])

    print("Experiment 3: Effect of an outlier")

    for label, predictions in [
        ("Normal predictions", y_pred_normal),
        ("With outlier", y_pred_outlier),
    ]:
        mae = mean_absolute_error(y_true_outlier, predictions)
        mse = mean_squared_error(y_true_outlier, predictions)

        print(f"{label}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")

    print()

    # Experiment 4: Compare loss functions across residuals
    residuals = np.linspace(-5, 5, 201)
    mae_values = np.abs(residuals)
    mse_values = residuals**2

    plt.figure(figsize=(9, 5))
    plt.plot(residuals, mae_values, label="Absolute error (MAE)")
    plt.plot(residuals, mse_values, label="Squared error (MSE)")
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Loss for one prediction")
    plt.title("Absolute Error versus Squared Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
