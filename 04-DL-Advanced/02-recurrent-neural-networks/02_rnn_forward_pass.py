"""
Implement the forward pass of a Recurrent Neural Network (RNN).

This experiment covers:
1. Hidden-state updates across time steps.
2. Output computation from hidden states.
3. Shared parameters across a sequence.
4. Tensor dimensions for inputs, hidden states, and outputs.
5. Numerical verification of the forward pass.
"""

import numpy as np


def rnn_forward(
    sequence: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
    W_hy: np.ndarray,
    b_y: np.ndarray,
    h0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute hidden states and outputs for an entire sequence."""
    hidden_size = W_hh.shape[0]
    output_size = W_hy.shape[0]

    h_t = h0.copy()
    hidden_states = []
    outputs = []

    for x_t in sequence:
        h_t = np.tanh(W_xh @ x_t + W_hh @ h_t + b_h)
        y_t = W_hy @ h_t + b_y

        hidden_states.append(h_t.copy())
        outputs.append(y_t.copy())

    return np.array(hidden_states).reshape(-1, hidden_size), np.array(
        outputs
    ).reshape(-1, output_size)


def main() -> None:
    np.random.seed(42)

    sequence_length = 5
    input_size = 3
    hidden_size = 4
    output_size = 2

    # Input sequence: (sequence_length, input_size)
    sequence = np.random.randn(sequence_length, input_size)

    # Initialize shared model parameters.
    W_xh = np.random.randn(hidden_size, input_size) * 0.1
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
    b_h = np.zeros(hidden_size)

    W_hy = np.random.randn(output_size, hidden_size) * 0.1
    b_y = np.zeros(output_size)

    # Initial hidden state.
    h0 = np.zeros(hidden_size)

    hidden_states, outputs = rnn_forward(
        sequence=sequence,
        W_xh=W_xh,
        W_hh=W_hh,
        b_h=b_h,
        W_hy=W_hy,
        b_y=b_y,
        h0=h0,
    )

    print("RNN Forward Pass")
    print("=" * 50)

    print(f"Input sequence shape: {sequence.shape}")
    print(f"Hidden states shape:  {hidden_states.shape}")
    print(f"Output sequence shape: {outputs.shape}")
    print()

    for t in range(sequence_length):
        print(f"Time step {t + 1}")
        print(f"  Input:  {sequence[t]}")
        print(f"  Hidden: {hidden_states[t]}")
        print(f"  Output: {outputs[t]}")
        print()

    # Verify tensor dimensions.
    assert hidden_states.shape == (sequence_length, hidden_size)
    assert outputs.shape == (sequence_length, output_size)

    # Verify the output calculation at each time step.
    expected_outputs = hidden_states @ W_hy.T + b_y
    np.testing.assert_allclose(outputs, expected_outputs, rtol=1e-10, atol=1e-10)

    # Verify that all results are finite.
    assert np.isfinite(hidden_states).all()
    assert np.isfinite(outputs).all()

    print("All dimension and numerical checks passed.")


if __name__ == "__main__":
    main()