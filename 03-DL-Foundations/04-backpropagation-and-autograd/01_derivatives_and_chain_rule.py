"""
Derivatives and the Chain Rule
==============================

Experiments:
1. Calculate derivatives analytically for common functions.
2. Approximate derivatives using finite differences.
3. Explore partial derivatives of functions with multiple variables.
4. Apply the chain rule to composed functions.
5. Compare analytical derivatives with numerical approximations.
6. Visualize how a derivative represents the local slope of a function.
"""

import numpy as np
import matplotlib.pyplot as plt


def numerical_derivative(
    function,
    x: float,
    h: float = 1e-5,
) -> float:
    """Approximate a derivative using central finite differences."""
    if not np.isfinite(x):
        raise ValueError("x must be finite.")

    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite number.")

    return float((function(x + h) - function(x - h)) / (2.0 * h))


def square(x: float) -> float:
    """Return x squared."""
    return x**2


def square_derivative(x: float) -> float:
    """Analytical derivative of x squared."""
    return 2.0 * x


def sigmoid(x: float) -> float:
    """Calculate the sigmoid activation."""
    return float(1.0 / (1.0 + np.exp(-x)))


def sigmoid_derivative(x: float) -> float:
    """Analytical derivative of sigmoid."""
    activation = sigmoid(x)
    return activation * (1.0 - activation)


def partial_derivatives(
    x: float,
    y: float,
) -> tuple[float, float]:
    """Calculate partial derivatives of f(x, y) = x^2 + 3xy + y^2."""
    df_dx = 2.0 * x + 3.0 * y
    df_dy = 3.0 * x + 2.0 * y
    return df_dx, df_dy


def main() -> None:
    # Experiment 1: Analytical versus numerical derivatives
    x = 3.0

    analytical = square_derivative(x)
    numerical = numerical_derivative(square, x)

    print("Experiment 1: Derivative of f(x) = x^2")
    print(f"x:                    {x:.4f}")
    print(f"Analytical derivative: {analytical:.6f}")
    print(f"Numerical derivative:  {numerical:.6f}")
    print(f"Absolute difference:   {abs(analytical - numerical):.2e}")
    print()

    # Experiment 2: Derivative of sigmoid
    x = 1.5

    analytical = sigmoid_derivative(x)
    numerical = numerical_derivative(sigmoid, x)

    print("Experiment 2: Derivative of sigmoid")
    print(f"x:                    {x:.4f}")
    print(f"Analytical derivative: {analytical:.6f}")
    print(f"Numerical derivative:  {numerical:.6f}")
    print()

    # Experiment 3: Partial derivatives
    # f(x, y) = x^2 + 3xy + y^2
    x, y = 2.0, 4.0

    df_dx, df_dy = partial_derivatives(x, y)

    print("Experiment 3: Partial derivatives")
    print("f(x, y) = x^2 + 3xy + y^2")
    print(f"At x={x}, y={y}:")
    print(f"Partial derivative with respect to x: {df_dx:.4f}")
    print(f"Partial derivative with respect to y: {df_dy:.4f}")
    print()

    # Experiment 4: Chain rule
    # g(x) = x^2, f(u) = 3u + 1, so f(g(x)) = 3x^2 + 1.
    x = 2.0
    u = square(x)

    dg_dx = 2.0 * x
    df_du = 3.0
    chain_rule_derivative = df_du * dg_dx

    def composed_function(value: float) -> float:
        return 3.0 * square(value) + 1.0

    numerical_chain_rule = numerical_derivative(composed_function, x)

    print("Experiment 4: Chain rule")
    print("g(x) = x^2")
    print("f(u) = 3u + 1")
    print(f"dg/dx:                    {dg_dx:.6f}")
    print(f"df/du:                    {df_du:.6f}")
    print(f"Chain rule result:        {chain_rule_derivative:.6f}")
    print(f"Numerical derivative:     {numerical_chain_rule:.6f}")
    print()

    # Experiment 5: Compare derivatives across multiple points
    points = np.linspace(-3.0, 3.0, 7)

    print("Experiment 5: Derivative comparison")
    print(f"{'x':>8} {'Analytical':>14} {'Numerical':>14}")
    print("-" * 38)

    for point in points:
        analytical = square_derivative(point)
        numerical = numerical_derivative(square, point)
        print(f"{point:8.3f} {analytical:14.6f} {numerical:14.6f}")

    # Experiment 6: Visualize function and tangent at x = 2
    x_values = np.linspace(-3.0, 3.0, 300)
    y_values = x_values**2

    x0 = 2.0
    y0 = square(x0)
    slope = square_derivative(x0)

    tangent = y0 + slope * (x_values - x0)

    plt.figure(figsize=(9, 5))
    plt.plot(x_values, y_values, label="f(x) = x²")
    plt.plot(
        x_values,
        tangent,
        linestyle="--",
        label=f"Tangent at x = {x0}",
    )
    plt.scatter([x0], [y0], zorder=3, label=f"Point ({x0}, {y0})")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("A Derivative Represents the Local Slope")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
