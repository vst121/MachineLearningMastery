"""
Learning Rate Experiments
=========================

Experiments:
1. Compare small, appropriate, and large learning rates.
2. Observe convergence and divergence.
3. Track loss and parameter updates.
4. Visualize optimization trajectories.
5. Identify learning-rate sensitivity.
"""

import numpy as np
import matplotlib.pyplot as plt


def objective(x):
    """Quadratic objective with a minimum at x = 3."""
    return (x - 3.0) ** 2 + 2.0


def gradient(x):
    """Analytical derivative of the objective."""
    return 2.0 * (x - 3.0)


def optimize(initial_x, learning_rate, epochs=50):
    """Run gradient descent and record optimization history."""
    x = float(initial_x)
    history = {
        "x": [x],
        "loss": [objective(x)],
        "gradient": [],
    }

    for _ in range(epochs):
        grad = gradient(x)
        history["gradient"].append(grad)

        x -= learning_rate * grad

        history["x"].append(x)
        history["loss"].append(objective(x))

        if not np.isfinite(x) or not np.isfinite(history["loss"][-1]):
            break

    return history


def compare_learning_rates():
    """Compare convergence for different learning rates."""
    learning_rates = [0.01, 0.1, 0.5, 0.9, 1.01]
    results = {}

    print("Learning Rate Comparison")
    print("-" * 72)
    print(
        f"{'Learning Rate':<18}"
        f"{'Final x':<18}"
        f"{'Final Loss':<18}"
        f"{'Status':<18}"
    )

    for learning_rate in learning_rates:
        history = optimize(
            initial_x=-5.0,
            learning_rate=learning_rate,
            epochs=50,
        )

        results[learning_rate] = history

        final_x = history["x"][-1]
        final_loss = history["loss"][-1]

        if not np.isfinite(final_loss):
            status = "Diverged"
        elif abs(final_x - 3.0) < 1e-4:
            status = "Converged"
        elif abs(final_x) > 1e6:
            status = "Diverging"
        else:
            status = "Slow / Unstable"

        print(
            f"{learning_rate:<18.2f}"
            f"{final_x:<18.6f}"
            f"{final_loss:<18.6f}"
            f"{status:<18}"
        )

    return results


def plot_loss(results):
    """Plot loss against optimization steps."""
    plt.figure(figsize=(10, 6))

    for learning_rate, history in results.items():
        plt.plot(
            history["loss"],
            label=f"Learning rate = {learning_rate}",
        )

    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title("Learning Rate vs. Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_parameter_trajectories(results):
    """Plot parameter values during optimization."""
    plt.figure(figsize=(10, 6))

    for learning_rate, history in results.items():
        plt.plot(
            history["x"],
            label=f"Learning rate = {learning_rate}",
        )

    plt.axhline(
        y=3.0,
        linestyle="--",
        label="Optimal parameter",
    )

    plt.xlabel("Optimization Step")
    plt.ylabel("Parameter x")
    plt.title("Parameter Updates for Different Learning Rates")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    results = compare_learning_rates()

    plot_loss(results)
    plot_parameter_trajectories(results)


if __name__ == "__main__":
    main()
