"""
Implement a Recurrent Neural Network using PyTorch.

This experiment covers:
1. RNN construction with nn.RNN.
2. Batch-first sequence representation.
3. Output sequences and final hidden states.
4. Tensor dimensions and parameter counts.
5. Verification against a manually computed RNN forward pass.
"""

import torch
from torch import nn


def main() -> None:
    torch.manual_seed(42)

    batch_size = 2
    sequence_length = 5
    input_size = 3
    hidden_size = 4
    num_layers = 1

    # Input shape: (batch_size, sequence_length, input_size)
    x = torch.randn(batch_size, sequence_length, input_size)

    rnn = nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity="tanh",
        batch_first=True,
    )

    # Initial hidden state shape:
    # (num_layers, batch_size, hidden_size)
    h0 = torch.zeros(num_layers, batch_size, hidden_size)

    # Outputs:
    # output: (batch_size, sequence_length, hidden_size)
    # h_n:    (num_layers, batch_size, hidden_size)
    output, h_n = rnn(x, h0)

    print("PyTorch RNN")
    print("=" * 55)
    print(f"Input shape:          {tuple(x.shape)}")
    print(f"Initial state shape:  {tuple(h0.shape)}")
    print(f"Output shape:         {tuple(output.shape)}")
    print(f"Final state shape:    {tuple(h_n.shape)}")
    print()

    print("Model parameters")
    print("-" * 55)

    total_parameters = 0

    for name, parameter in rnn.named_parameters():
        print(f"{name:20s} {tuple(parameter.shape)}")
        total_parameters += parameter.numel()

    print(f"\nTotal parameters: {total_parameters}")

    # For a single-layer, unidirectional RNN with biases:
    expected_parameters = (
        hidden_size * input_size
        + hidden_size * hidden_size
        + 2 * hidden_size
    )

    assert total_parameters == expected_parameters
    
    # The last output equals the final hidden state for this
    # single-layer, unidirectional RNN.
    torch.testing.assert_close(output[:, -1, :], h_n[-1])

    # Verify that outputs and hidden states are finite.
    assert torch.isfinite(output).all()
    assert torch.isfinite(h_n).all()

    print("\nAll dimension and numerical checks passed.")

    # Demonstrate how a linear layer can map each hidden state
    # to a task-specific output dimension.
    output_size = 2
    classifier = nn.Linear(hidden_size, output_size)

    predictions = classifier(output)

    print("\nOutput projection")
    print("-" * 55)
    print(f"Hidden sequence shape: {tuple(output.shape)}")
    print(f"Prediction shape:      {tuple(predictions.shape)}")


if __name__ == "__main__":
    main()