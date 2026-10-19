"""
Explore Backpropagation Through Time (BPTT) in a Recurrent Neural Network.

This experiment covers:
1. Forward propagation across a sequence.
2. Loss calculation over multiple time steps.
3. Backward propagation through recurrent connections.
4. Gradient accumulation for shared parameters.
5. Numerical gradient verification using finite differences.
"""

import numpy as np


def forward(
    sequence: np.ndarray,
    targets: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
    W_hy: np.ndarray,
    b_y: np.ndarray,
) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
    """Run the RNN forward pass and calculate mean squared error."""
    h_t = np.zeros(W_hh.shape[0])
    hidden_states = [h_t.copy()]
    outputs = []

    for x_t in sequence:
        h_t = np.tanh(W_xh @ x_t + W_hh @ h_t + b_h)
        y_t = W_hy @ h_t + b_y

        hidden_states.append(h_t.copy())
        outputs.append(y_t.copy())

    outputs_array = np.array(outputs)
    loss = np.mean((outputs_array - targets) ** 2)

    return loss, hidden_states, outputs


def backward(
    sequence: np.ndarray,
    targets: np.ndarray,
    hidden_states: list[np.ndarray],
    outputs: list[np.ndarray],
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    W_hy: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate gradients using Backpropagation Through Time."""
    sequence_length = len(sequence)
    output_size = targets.shape[1]

    # Derivative of mean squared error with respect to outputs.
    d_outputs = (
        2.0 * (np.array(outputs) - targets)
        / (sequence_length * output_size)
    )

    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db_h = np.zeros(W_hh.shape[0])
    dW_hy = np.zeros_like(W_hy)
    db_y = np.zeros(W_hy.shape[0])

    # Gradient arriving from the next time step.
    dh_next = np.zeros(W_hh.shape[0])

    for t in reversed(range(sequence_length)):
        h_t = hidden_states[t + 1]
        h_prev = hidden_states[t]

        # Output-layer gradients.
        dy = d_outputs[t]
        dW_hy += np.outer(dy, h_t)
        db_y += dy

        # Combine gradients from the output and future time steps.
        dh = W_hy.T @ dy + dh_next

        # Derivative of tanh activation.
        dz = dh * (1.0 - h_t**2)

        # Accumulate gradients for shared recurrent parameters.
        dW_xh += np.outer(dz, sequence[t])
        dW_hh += np.outer(dz, h_prev)
        db_h += dz

        # Propagate the gradient to the previous time step.
        dh_next = W_hh.T @ dz

    return {
        "W_xh": dW_xh,
        "W_hh": dW_hh,
        "b_h": db_h,
        "W_hy": dW_hy,
        "b_y": db_y,
    }


def numerical_gradient(
    parameter: np.ndarray,
    parameter_name: str,
    parameters: dict[str, np.ndarray],
    sequence: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Estimate a parameter gradient using central finite differences."""
    gradient = np.zeros_like(parameter)

    for index in np.ndindex(parameter.shape):
        original = parameter[index]

        parameter[index] = original + epsilon
        loss_plus, _, _ = forward(sequence, targets, **parameters)

        parameter[index] = original - epsilon
        loss_minus, _, _ = forward(sequence, targets, **parameters)

        parameter[index] = original
        gradient[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

    return gradient


def main() -> None:
    np.random.seed(42)

    sequence_length = 4
    input_size = 2
    hidden_size = 3
    output_size = 1

    sequence = np.random.randn(sequence_length, input_size)
    targets = np.random.randn(sequence_length, output_size)

    parameters = {
        "W_xh": np.random.randn(hidden_size, input_size) * 0.1,
        "W_hh": np.random.randn(hidden_size, hidden_size) * 0.1,
        "b_h": np.zeros(hidden_size),
        "W_hy": np.random.randn(output_size, hidden_size) * 0.1,
        "b_y": np.zeros(output_size),
    }

    loss, hidden_states, outputs = forward(
        sequence, targets, **parameters
    )

    gradients = backward(
        sequence=sequence,
        targets=targets,
        hidden_states=hidden_states,
        outputs=outputs,
        W_xh=parameters["W_xh"],
        W_hh=parameters["W_hh"],
        W_hy=parameters["W_hy"],
    )

    print("Backpropagation Through Time")
    print("=" * 50)
    print(f"Sequence shape:       {sequence.shape}")
    print(f"Hidden states shape:  {np.array(hidden_states).shape}")
    print(f"Output shape:         {np.array(outputs).shape}")
    print(f"Mean squared error:   {loss:.6f}")
    print()

    print("Gradient norms")
    print("-" * 50)

    for name, gradient in gradients.items():
        print(f"{name:8s}: {np.linalg.norm(gradient):.8f}")

    # Verify analytical gradients against numerical gradients.
    print("\nNumerical Gradient Verification")
    print("-" * 50)

    for name in ("W_xh", "W_hh", "W_hy", "b_h", "b_y"):
        numerical = numerical_gradient(
            parameter=parameters[name],
            parameter_name=name,
            parameters=parameters,
            sequence=sequence,
            targets=targets,
        )

        np.testing.assert_allclose(
            gradients[name],
            numerical,
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"Gradient verification failed for {name}",
        )

        max_error = np.max(np.abs(gradients[name] - numerical))
        print(f"{name:8s}: maximum absolute error = {max_error:.2e}")

    print("\nAll gradient checks passed.")


if __name__ == "__main__":
    main()