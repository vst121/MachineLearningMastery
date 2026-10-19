"""
Explore the fundamentals of Recurrent Neural Networks (RNNs).

This experiment covers:
1. Sequential input processing.
2. Hidden-state updates across time steps.
3. Parameter sharing in recurrent connections.
4. Tensor dimensions and sequence representation.
5. Verification of hidden-state dimensions.
"""

import numpy as np


def rnn_step(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    W_xh: np.ndarray,
    W_hh: np.ndarray,
    b_h: np.ndarray,
) -> np.ndarray:
    """Calculate the hidden state for one time step."""
    return np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)


def main() -> None:
    np.random.seed(42)

    sequence_length = 5
    input_size = 3
    hidden_size = 4

    # Generate a sequence: (sequence_length, input_size)
    sequence = np.random.randn(sequence_length, input_size)

    # Initialize shared recurrent parameters.
    # W_xh: (hidden_size, input_size)
    # W_hh: (hidden_size, hidden_size)
    W_xh = np.random.randn(hidden_size, input_size) * 0.1
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
    b_h = np.zeros(hidden_size)

    # Initialize the hidden state with zeros.
    h_t = np.zeros(hidden_size)

    hidden_states = []

    parameter_count = W_xh.size + W_hh.size + b_h.size

    print("RNN Fundamentals")
    print("=" * 50)
    print(f"Input sequence shape: {sequence.shape}")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Recurrent parameter count: {parameter_count}")
    print()

    # Process the sequence one time step at a time.
    for t, x_t in enumerate(sequence):
        h_prev = h_t.copy()

        h_t = rnn_step(
            x_t=x_t,
            h_prev=h_prev,
            W_xh=W_xh,
            W_hh=W_hh,
            b_h=b_h,
        )

        hidden_states.append(h_t.copy())

        print(f"Time step {t + 1}")
        print(f"  Input:          {x_t}")
        print(f"  Previous state: {h_prev}")
        print(f"  Current state:  {h_t}")
        print()

    # Stack the hidden states into a sequence.
    hidden_states = np.stack(hidden_states)

    print("Final Results")
    print("=" * 50)
    print(f"Input sequence shape: {sequence.shape}")
    print(f"Hidden states shape:  {hidden_states.shape}")
    print(f"Final hidden state:   {hidden_states[-1]}")
    print(f"Final state shape:    {h_t.shape}")

    # Verify the expected dimensions.
    assert sequence.shape == (sequence_length, input_size)
    assert hidden_states.shape == (sequence_length, hidden_size)
    assert h_t.shape == (hidden_size,)

    # Verify that every hidden state is within the tanh range.
    assert np.all(np.abs(hidden_states) <= 1.0)

    print("\nAll dimension and value checks passed.")


if __name__ == "__main__":
    main()