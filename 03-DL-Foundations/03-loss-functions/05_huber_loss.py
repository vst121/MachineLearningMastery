"""
Huber Loss
==========

Experiments:
1. Implement Huber Loss using NumPy.
2. Calculate losses for individual prediction errors.
3. Compare Huber Loss with MSE and MAE.
4. Investigate the impact of outliers.
5. Visualize how the delta parameter changes the loss function.
"""

import numpy as np
import matplotlib.pyplot as plt


def huber_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.0,
) -> float:
    """Calculate mean Huber Loss."""
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

    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a positive finite number.")

    residuals = y_true - y_pred
    absolute_errors = np.abs(residuals)

    # Quadratic for small errors, linear for large errors.
    losses = np.where(
        absolute_errors <= delta,
        0.5 * residuals**2,
        delta * (absolute_errors - 0.5 * delta),
    )

    return float(np.mean(losses))


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate mean squared error for comparison."""
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

    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate mean absolute error for comparison."""
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


def huber_loss_per_sample(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.0,
) -> np.ndarray:
    """Return Huber Loss for each prediction."""
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

    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a positive finite number.")

    residuals = y_true - y_pred
    absolute_errors = np.abs(residuals)

    return np.where(
        absolute_errors <= delta,
        0.5 * residuals**2,
        delta * (absolute_errors - 0.5 * delta),
    )


def main() -> None:
    # Experiment 1: Basic Huber Loss calculation
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    losses = huber_loss_per_sample(y_true, y_pred, delta=1.0)

    print("Experiment 1: Basic Huber Loss")
    for index, loss in enumerate(losses, start=1):
        print(f"Sample {index}: Huber Loss = {loss:.4f}")

    print(f"Mean Huber Loss: {huber_loss(y_true, y_pred):.4f}")
    print()

    # Experiment 2: Compare loss functions
    print("Experiment 2: Loss function comparison")

    for name, loss_function in [
        ("MSE", mean_squared_error),
        ("MAE", mean_absolute_error),
        ("Huber (delta=1)", lambda yt, yp: huber_loss(yt, yp, delta=1.0)),
    ]:
        print(f"{name}: {loss_function(y_true, y_pred):.4f}")

    print()

    # Experiment 3: Investigate the effect of an outlier
    y_true_outlier = np.array([10.0, 12.0, 13.0, 15.0])

    predictions_normal = np.array([11.0, 11.0, 14.0, 14.0])
    predictions_outlier = np.array([11.0, 11.0, 14.0, 30.0])

    print("Experiment 3: Impact of an outlier")

    for label, predictions in [
        ("Normal predictions", predictions_normal),
        ("With outlier", predictions_outlier),
    ]:
        print(f"{label}:")
        print(f"  MSE:   {mean_squared_error(y_true_outlier, predictions):.4f}")
        print(f"  MAE:   {mean_absolute_error(y_true_outlier, predictions):.4f}")
        print(f"  Huber: {huber_loss(y_true_outlier, predictions):.4f}")

    print()

    # Experiment 4: Visualize the effect of delta
    residuals = np.linspace(-5, 5, 400)

    plt.figure(figsize=(9, 5))

    for delta in [0.5, 1.0, 2.0]:
        absolute_errors = np.abs(residuals)

        losses = np.where(
            absolute_errors <= delta,
            0.5 * residuals**2,
            delta * (absolute_errors - 0.5 * delta),
        )

        plt.plot(residuals, losses, label=f"Huber (delta={delta})")

    plt.plot(residuals, residuals**2, linestyle="--", label="Squared error")
    plt.plot(residuals, np.abs(residuals), linestyle=":", label="Absolute error")

    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Loss")
    plt.title("Huber Loss versus MSE and MAE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
