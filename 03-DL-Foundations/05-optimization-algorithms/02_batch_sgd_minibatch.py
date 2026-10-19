
"""
Batch, Stochastic, and Mini-Batch Gradient Descent
==================================================

Experiments:
1. Implement batch gradient descent.
2. Implement stochastic gradient descent.
3. Implement mini-batch gradient descent.
4. Compare loss curves and parameter updates.
5. Examine the effect of batch size on optimization.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(samples=200, seed=42):
    """Generate synthetic regression data."""
    rng = np.random.default_rng(seed)

    X = rng.uniform(-2, 2, size=(samples, 1))
    noise = rng.normal(0, 0.5, size=(samples, 1))
    y = 3 * X + 2 + noise

    return X, y


def predict(X, weights, bias):
    """Calculate model predictions."""
    return X @ weights + bias


def compute_loss(X, y, weights, bias):
    """Calculate mean squared error."""
    errors = predict(X, weights, bias) - y
    return np.mean(errors**2)


def compute_gradients(X, y, weights, bias):
    """Calculate gradients of mean squared error."""
    sample_count = X.shape[0]
    errors = predict(X, weights, bias) - y

    dW = (2 / sample_count) * X.T @ errors
    db = 2 * np.mean(errors)

    return dW, db


def train(
    X,
    y,
    method,
    learning_rate=0.05,
    epochs=100,
    batch_size=16,
    seed=42,
):
    """Train a linear regression model using a selected update strategy."""
    rng = np.random.default_rng(seed)

    weights = np.zeros((X.shape[1], 1))
    bias = 0.0

    losses = []
    update_counts = []
    updates = 0
    sample_count = X.shape[0]

    for _ in range(epochs):
        if method == "batch":
            batches = [(X, y)]

        else:
            indices = rng.permutation(sample_count)

            if method == "stochastic":
                current_batch_size = 1
            elif method == "mini_batch":
                current_batch_size = batch_size
            else:
                raise ValueError(f"Unknown method: {method}")

            batches = [
                (X[indices[i:i + current_batch_size]],
                 y[indices[i:i + current_batch_size]])
                for i in range(0, sample_count, current_batch_size)
            ]

        for X_batch, y_batch in batches:
            dW, db = compute_gradients(
                X_batch, y_batch, weights, bias
            )

            weights -= learning_rate * dW
            bias -= learning_rate * db
            updates += 1

        losses.append(compute_loss(X, y, weights, bias))
        update_counts.append(updates)

    return weights, bias, losses, update_counts


def main():
    X, y = generate_data()

    methods = {
        "batch": "Batch Gradient Descent",
        "stochastic": "Stochastic Gradient Descent",
        "mini_batch": "Mini-Batch Gradient Descent",
    }

    results = {}

    print("Gradient Descent Strategy Comparison")
    print("-" * 78)
    print(
        f"{'Method':<30}{'Weight':>12}"
        f"{'Bias':>12}{'Final Loss':>14}{'Updates':>10}"
    )

    for method, label in methods.items():
        weights, bias, losses, updates = train(
            X,
            y,
            method=method,
            learning_rate=0.05,
            epochs=100,
            batch_size=16,
        )

        results[label] = {
            "losses": losses,
            "updates": updates,
        }

        print(
            f"{label:<30}"
            f"{weights[0, 0]:>12.4f}"
            f"{bias:>12.4f}"
            f"{losses[-1]:>14.4f}"
            f"{updates[-1]:>10}"
        )

    plt.figure(figsize=(10, 6))

    for label, result in results.items():
        plt.plot(result["losses"], label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Full-Dataset MSE")
    plt.title("Convergence by Gradient Descent Strategy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))

    for label, result in results.items():
        plt.plot(result["updates"], result["losses"], label=label)

    plt.xlabel("Parameter Updates")
    plt.ylabel("Full-Dataset MSE")
    plt.title("Convergence by Number of Parameter Updates")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
