
"""
Momentum Optimization
=====================

Experiments:
1. Implement standard gradient descent.
2. Implement momentum-based gradient descent.
3. Compare optimization trajectories on an ill-conditioned function.
4. Examine the effect of the momentum coefficient.
5. Visualize convergence and parameter updates.
"""

import numpy as np
import matplotlib.pyplot as plt


def objective(parameters):
    """An ill-conditioned quadratic function."""
    x, y = parameters
    return 0.5 * (x**2 + 25.0 * y**2)


def gradient(parameters):
    """Analytical gradient of the objective."""
    x, y = parameters
    return np.array([x, 25.0 * y])


def optimize(
    initial_parameters,
    learning_rate,
    epochs,
    momentum=0.0,
):
    """Optimize using gradient descent with optional momentum."""
    parameters = np.array(initial_parameters, dtype=float)
    velocity = np.zeros_like(parameters)

    history = {
        "parameters": [parameters.copy()],
        "loss": [objective(parameters)],
    }

    for _ in range(epochs):
        grad = gradient(parameters)

        velocity = momentum * velocity + grad
        parameters -= learning_rate * velocity

        history["parameters"].append(parameters.copy())
        history["loss"].append(objective(parameters))

        if (
            not np.all(np.isfinite(parameters))
            or not np.isfinite(history["loss"][-1])
        ):
            break

    return history


def compare_optimizers():
    """Compare standard gradient descent and momentum."""
    initial_parameters = np.array([5.0, 5.0])
    learning_rate = 0.04
    epochs = 100

    configurations = {
        "Gradient Descent": 0.0,
        "Momentum (0.5)": 0.5,
        "Momentum (0.9)": 0.9,
    }

    results = {}

    print("Optimizer Comparison")
    print("-" * 70)
    print(
        f"{'Optimizer':<24}"
        f"{'Final Loss':<18}"
        f"{'Final x':<14}"
        f"{'Final y':<14}"
    )

    for name, momentum in configurations.items():
        history = optimize(
            initial_parameters=initial_parameters,
            learning_rate=learning_rate,
            epochs=epochs,
            momentum=momentum,
        )

        results[name] = history
        final_parameters = history["parameters"][-1]
        final_loss = history["loss"][-1]

        print(
            f"{name:<24}"
            f"{final_loss:<18.6e}"
            f"{final_parameters[0]:<14.6f}"
            f"{final_parameters[1]:<14.6f}"
        )

    return results


def plot_loss(results):
    """Compare loss convergence."""
    plt.figure(figsize=(10, 6))

    for name, history in results.items():
        plt.plot(history["loss"], label=name)

    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title("Gradient Descent vs. Momentum")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_trajectories(results):
    """Visualize parameter trajectories across the objective surface."""
    x_values = np.linspace(-6, 6, 200)
    y_values = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x_values, y_values)

    Z = 0.5 * (X**2 + 25.0 * Y**2)

    plt.figure(figsize=(10, 7))
    plt.contour(X, Y, Z, levels=30)

    for name, history in results.items():
        parameters = np.array(history["parameters"])

        plt.plot(
            parameters[:, 0],
            parameters[:, 1],
            marker="o",
            markersize=2,
            label=name,
        )

    plt.scatter(0, 0, marker="*", s=150, label="Minimum")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Optimization Trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    results = compare_optimizers()

    plot_loss(results)
    plot_trajectories(results)


if __name__ == "__main__":
    main()
