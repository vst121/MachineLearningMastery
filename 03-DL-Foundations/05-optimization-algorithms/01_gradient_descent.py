"""
Gradient Descent
================

Experiments:
1. Implement gradient descent from scratch.
2. Minimize a quadratic function.
3. Compare analytical and numerical gradients.
4. Experiment with different learning rates.
5. Visualize the optimization trajectory and convergence.
"""

import numpy as np
import matplotlib.pyplot as plt


def objective(x):
    """Quadratic function with a minimum at x = 3."""
    return (x - 3.0) ** 2 + 2.0


def analytical_gradient(x):
    """Derivative of the objective function."""
    return 2.0 * (x - 3.0)


def numerical_gradient(x, h=1e-5):
    """Approximate the derivative using central differences."""
    return (objective(x + h) - objective(x - h)) / (2.0 * h)


def gradient_descent(initial_x, learning_rate, epochs):
    """Minimize the objective function using gradient descent."""
    x = float(initial_x)

    history = {
        "x": [],
        "loss": [],
        "gradient": [],
    }

    for epoch in range(epochs):
        loss = objective(x)
        gradient = analytical_gradient(x)

        history["x"].append(x)
        history["loss"].append(loss)
        history["gradient"].append(gradient)

        x -= learning_rate * gradient

    history["x"].append(x)
    history["loss"].append(objective(x))

    return x, history


def compare_gradients():
    """Compare analytical and numerical derivatives."""
    x = 1.5

    analytical = analytical_gradient(x)
    numerical = numerical_gradient(x)

    print("Gradient Verification")
    print("-" * 40)
    print(f"x:                  {x:.4f}")
    print(f"Analytical gradient: {analytical:.8f}")
    print(f"Numerical gradient:  {numerical:.8f}")
    print(f"Absolute error:      {abs(analytical - numerical):.2e}")


def compare_learning_rates():
    """Observe how learning rates affect convergence."""
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    initial_x = -5.0
    epochs = 50

    print("\nLearning Rate Comparison")
    print("-" * 60)
    print(f"{'Learning Rate':<18}{'Final x':<18}{'Final Loss':<18}")

    histories = {}

    for learning_rate in learning_rates:
        final_x, history = gradient_descent(
            initial_x,
            learning_rate,
            epochs,
        )

        histories[learning_rate] = history

        print(
            f"{learning_rate:<18.2f}"
            f"{final_x:<18.6f}"
            f"{objective(final_x):<18.6f}"
        )

    return histories


def plot_convergence(histories):
    """Plot loss values across optimization steps."""
    plt.figure(figsize=(9, 5))

    for learning_rate, history in histories.items():
        plt.plot(
            range(len(history["loss"])),
            history["loss"],
            label=f"Learning rate = {learning_rate}",
        )

    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title("Gradient Descent: Learning Rate Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_objective_and_trajectory(history):
    """Visualize the objective function and optimization trajectory."""
    x_values = np.linspace(-6, 10, 300)
    y_values = objective(x_values)

    trajectory_x = np.array(history["x"])
    trajectory_y = objective(trajectory_x)

    plt.figure(figsize=(9, 5))
    plt.plot(x_values, y_values, label="Objective function")
    plt.plot(
        trajectory_x,
        trajectory_y,
        "o-",
        markersize=3,
        label="Optimization trajectory",
    )

    plt.xlabel("x")
    plt.ylabel("Loss")
    plt.title("Gradient Descent Optimization Trajectory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    compare_gradients()

    initial_x = -5.0
    learning_rate = 0.1
    epochs = 50

    final_x, history = gradient_descent(
        initial_x,
        learning_rate,
        epochs,
    )

    print("\nSingle Gradient Descent Run")
    print("-" * 40)
    print(f"Initial x: {initial_x:.4f}")
    print(f"Learning rate: {learning_rate:.2f}")
    print(f"Final x: {final_x:.6f}")
    print(f"Final loss: {objective(final_x):.6f}")
    print("Expected minimum: x = 3.0, loss = 2.0")

    histories = compare_learning_rates()

    plot_convergence(histories)
    plot_objective_and_trajectory(history)


if __name__ == "__main__":
    main()
