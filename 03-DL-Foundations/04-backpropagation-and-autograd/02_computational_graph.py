"""
Computational Graphs
====================

Experiments:
1. Represent a mathematical expression as a computational graph.
2. Execute a forward pass to calculate intermediate values and loss.
3. Calculate gradients by propagating derivatives backward.
4. Apply the chain rule to a multi-operation graph.
5. Compare analytical gradients with numerical approximations.
6. Visualize the computational graph and its dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt


class Value:
    """A scalar value in a computational graph."""

    def __init__(self, data: float, parents=(), operation: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents
        self.operation = operation
        self._backward = lambda: None

    def __add__(self, other):
        other = as_value(other)
        result = Value(
            self.data + other.data,
            parents=(self, other),
            operation="+",
        )

        def backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = backward
        return result

    def __mul__(self, other):
        other = as_value(other)
        result = Value(
            self.data * other.data,
            parents=(self, other),
            operation="*",
        )

        def backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = backward
        return result

    def __pow__(self, exponent: float):
        result = Value(
            self.data**exponent,
            parents=(self,),
            operation=f"**{exponent}",
        )

        def backward():
            self.grad += (
                exponent * self.data ** (exponent - 1) * result.grad
            )

        result._backward = backward
        return result

    def backward(self):
        """Propagate gradients from this output through the graph."""
        topological_order = []
        visited = set()

        def build_topology(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    build_topology(parent)
                topological_order.append(node)

        build_topology(self)

        # Reset gradients so repeated calls don't accumulate old values.
        for node in topological_order:
            node.grad = 0.0

        self.grad = 1.0

        for node in reversed(topological_order):
            node._backward()


def as_value(value) -> Value:
    """Convert a scalar to a graph node if necessary."""
    return value if isinstance(value, Value) else Value(value)


def numerical_derivative(function, x: float, h: float = 1e-5) -> float:
    """Approximate a scalar derivative using central differences."""
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite number.")

    return float((function(x + h) - function(x - h)) / (2.0 * h))


def main() -> None:
    # Experiment 1: Build a computational graph.
    # f = (a * b) + c
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)

    product = a * b
    output = product + c

    print("Experiment 1: Forward pass")
    print(f"a = {a.data}")
    print(f"b = {b.data}")
    print(f"c = {c.data}")
    print(f"a * b = {product.data}")
    print(f"output = (a * b) + c = {output.data}")
    print()

    # Experiment 2: Backward pass.
    output.backward()

    print("Experiment 2: Backward pass")
    print(f"d(output)/da = {a.grad:.4f}")
    print(f"d(output)/db = {b.grad:.4f}")
    print(f"d(output)/dc = {c.grad:.4f}")
    print()

    # Experiment 3: A graph with a squared error loss.
    # prediction = w*x + b
    # loss = (prediction - target)^2
    w = Value(2.0)
    x = Value(3.0)
    bias = Value(0.5)
    target = 7.0

    prediction = w * x + bias
    error = prediction + (-target)
    loss = error**2

    loss.backward()

    print("Experiment 3: Linear prediction and squared error")
    print(f"Prediction: {prediction.data:.4f}")
    print(f"Target:     {target:.4f}")
    print(f"Loss:       {loss.data:.4f}")
    print(f"dLoss/dw:   {w.grad:.4f}")
    print(f"dLoss/db:   {bias.grad:.4f}")
    print()

    # Experiment 4: Verify the gradient numerically.
    def loss_for_weight(weight: float) -> float:
        prediction_value = weight * 3.0 + 0.5
        return (prediction_value - target) ** 2

    numerical_grad = numerical_derivative(loss_for_weight, w.data)

    print("Experiment 4: Gradient verification")
    print(f"Analytical dLoss/dw: {w.grad:.6f}")
    print(f"Numerical dLoss/dw:  {numerical_grad:.6f}")
    print(f"Absolute difference: {abs(w.grad - numerical_grad):.2e}")
    print()

    # Experiment 5: Visualize the computational graph.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    positions = {
        "a": (0.1, 0.7),
        "b": (0.1, 0.3),
        "multiply": (0.4, 0.5),
        "c": (0.4, 0.1),
        "output": (0.75, 0.4),
    }

    labels = {
        "a": f"a = {a.data}",
        "b": f"b = {b.data}",
        "multiply": f"a × b = {product.data}",
        "c": f"c = {c.data}",
        "output": f"output = {output.data}",
    }

    for name, (px, py) in positions.items():
        ax.text(
            px,
            py,
            labels[name],
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white"},
        )

    edges = [
        ("a", "multiply"),
        ("b", "multiply"),
        ("multiply", "output"),
        ("c", "output"),
    ]

    for source, destination in edges:
        x1, y1 = positions[source]
        x2, y2 = positions[destination]

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->"},
        )

    ax.set_title("Computational Graph: output = (a × b) + c")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
