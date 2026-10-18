"""
Gradient Checking
=================

Experiments:
1. Approximate derivatives using central finite differences.
2. Compare analytical and numerical gradients for a scalar function.
3. Check gradients of a small neural network.
4. Calculate relative gradient errors.
5. Demonstrate how gradient checking detects an incorrect gradient.
"""

import numpy as np


def central_difference(function, parameters, h=1e-5):
    """Approximate the gradient using central finite differences."""
    parameters = np.asarray(parameters, dtype=float)
    gradient = np.zeros_like(parameters)

    for i in range(parameters.size):
        parameters_plus = parameters.copy()
        parameters_minus = parameters.copy()

        parameters_plus[i] += h
        parameters_minus[i] -= h

        gradient[i] = (
            function(parameters_plus) - function(parameters_minus)
        ) / (2 * h)

    return gradient


def relative_error(analytical, numerical, epsilon=1e-12):
    """Calculate the relative error between two gradient vectors."""
    analytical = np.asarray(analytical, dtype=float)
    numerical = np.asarray(numerical, dtype=float)

    numerator = np.linalg.norm(analytical - numerical)
    denominator = max(
        np.linalg.norm(analytical) + np.linalg.norm(numerical),
        epsilon,
    )

    return numerator / denominator


def sigmoid(x):
    """Apply the sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def scalar_function(parameters):
    """A differentiable function for a basic gradient check."""
    x, y = parameters
    return x**2 + 3 * x * y + y**2


def scalar_function_gradient(parameters):
    """Analytical gradient of scalar_function."""
    x, y = parameters

    return np.array([
        2 * x + 3 * y,
        3 * x + 2 * y,
    ])


def unpack_parameters(parameters, input_size, hidden_size):
    """Convert a flat parameter vector into network parameters."""
    index = 0

    w1_size = input_size * hidden_size
    W1 = parameters[index:index + w1_size].reshape(
        input_size, hidden_size
    )
    index += w1_size

    b1 = parameters[index:index + hidden_size]
    index += hidden_size

    W2 = parameters[index:index + hidden_size].reshape(
        hidden_size, 1
    )
    index += hidden_size

    b2 = parameters[index:index + 1]

    return W1, b1, W2, b2


def pack_gradients(dW1, db1, dW2, db2):
    """Flatten network gradients into a single vector."""
    return np.concatenate([
        dW1.ravel(),
        db1.ravel(),
        dW2.ravel(),
        db2.ravel(),
    ])


def network_loss(parameters, X, y, input_size, hidden_size):
    """Calculate binary cross-entropy loss for a small neural network."""
    W1, b1, W2, b2 = unpack_parameters(
        parameters, input_size, hidden_size
    )

    hidden = sigmoid(X @ W1 + b1)
    predictions = sigmoid(hidden @ W2 + b2)

    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    return -np.mean(
        y * np.log(predictions)
        + (1 - y) * np.log(1 - predictions)
    )


def network_loss_and_gradients(
    parameters, X, y, input_size, hidden_size
):
    """Calculate the loss and analytical gradients."""
    W1, b1, W2, b2 = unpack_parameters(
        parameters, input_size, hidden_size
    )

    sample_count = X.shape[0]

    # Forward pass
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    epsilon = 1e-12
    a2_clipped = np.clip(a2, epsilon, 1 - epsilon)

    loss = -np.mean(
        y * np.log(a2_clipped)
        + (1 - y) * np.log(1 - a2_clipped)
    )

    # Backward pass: sigmoid output with binary cross-entropy
    dz2 = (a2 - y) / sample_count
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * a1 * (1 - a1)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)

    gradients = pack_gradients(dW1, db1, dW2, db2)

    return loss, gradients


def check_scalar_function():
    """Compare analytical and numerical derivatives."""
    parameters = np.array([1.5, -0.5])

    analytical = scalar_function_gradient(parameters)
    numerical = central_difference(scalar_function, parameters)

    print("Scalar Function Gradient Check")
    print("-" * 40)
    print("Analytical gradient:", analytical)
    print("Numerical gradient: ", numerical)
    print(f"Relative error:     {relative_error(analytical, numerical):.3e}")


def check_neural_network():
    """Compare all network gradients with finite differences."""
    np.random.seed(42)

    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    y = np.array([
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ])

    input_size = X.shape[1]
    hidden_size = 3

    parameter_count = (
        input_size * hidden_size
        + hidden_size
        + hidden_size
        + 1
    )

    parameters = np.random.randn(parameter_count) * 0.3

    loss, analytical = network_loss_and_gradients(
        parameters, X, y, input_size, hidden_size
    )

    numerical = central_difference(
        lambda p: network_loss(
            p, X, y, input_size, hidden_size
        ),
        parameters,
    )

    error = relative_error(analytical, numerical)

    print("\nNeural Network Gradient Check")
    print("-" * 40)
    print(f"Loss:               {loss:.6f}")
    print(f"Parameter count:    {parameter_count}")
    print(f"Relative error:     {error:.3e}")
    print("\nGradient comparison (first five parameters):")

    for index, (analytic, numeric) in enumerate(
        zip(analytical[:5], numerical[:5])
    ):
        print(
            f"{index:2d}: analytical={analytic: .8f}, "
            f"numerical={numeric: .8f}"
        )

    if error < 1e-7:
        print("\nResult: Gradients match.")
    else:
        print("\nResult: Gradient check failed.")


def demonstrate_incorrect_gradient():
    """Show how gradient checking identifies an incorrect gradient."""
    parameters = np.array([1.5, -0.5])

    analytical = scalar_function_gradient(parameters)
    numerical = central_difference(scalar_function, parameters)

    incorrect = analytical.copy()
    incorrect[0] += 0.5

    correct_error = relative_error(analytical, numerical)
    incorrect_error = relative_error(incorrect, numerical)

    print("\nIncorrect Gradient Demonstration")
    print("-" * 40)
    print(f"Correct gradient error:   {correct_error:.3e}")
    print(f"Incorrect gradient error:  {incorrect_error:.3e}")


if __name__ == "__main__":
    check_scalar_function()
    check_neural_network()
    demonstrate_incorrect_gradient()
