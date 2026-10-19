"""
Explore vanishing and exploding gradients in Recurrent Neural Networks.

This experiment covers:
1. Gradient propagation through recurrent connections.
2. The effect of recurrent weight scales on gradient magnitudes.
3. Vanishing and exploding gradients over long sequences.
4. Gradient clipping as a stabilization technique.
5. Gradient-norm comparisons across sequence lengths.
"""

import numpy as np
import torch
from torch import nn


def simulate_gradient_propagation(
    sequence_length: int,
    recurrent_scale: float,
    hidden_size: int = 8,
) -> list[float]:
    """Illustrate gradient propagation through a simplified linear RNN."""
    recurrent_matrix = np.eye(hidden_size) * recurrent_scale
    gradient = np.ones(hidden_size)

    gradient_norms = [np.linalg.norm(gradient)]

    for _ in range(sequence_length):
        gradient = recurrent_matrix.T @ gradient
        gradient_norms.append(np.linalg.norm(gradient))

    return gradient_norms


def demonstrate_gradient_clipping() -> None:
    """Compare gradients before and after global-norm clipping."""
    model = nn.Linear(4, 1)

    # Create deliberately large gradients for demonstration.
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)

    norm_before = torch.sqrt(
        sum(parameter.grad.norm() ** 2 for parameter in model.parameters())
    ).item()

    max_norm = 1.0
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

    norm_after = torch.sqrt(
        sum(parameter.grad.norm() ** 2 for parameter in model.parameters())
    ).item()

    print("\nGradient Clipping")
    print("-" * 60)
    print(f"Gradient norm before clipping: {norm_before:.6f}")
    print(f"Gradient norm after clipping:  {norm_after:.6f}")
    print(f"Maximum allowed norm:          {max_norm:.6f}")

    assert norm_after <= max_norm + 1e-6


def main() -> None:
    sequence_length = 30

    print("Gradient Propagation in Recurrent Networks")
    print("=" * 60)
    print(f"Sequence length: {sequence_length}")
    print()

    scenarios = {
        "Vanishing": 0.5,
        "Stable": 1.0,
        "Exploding": 1.5,
    }

    final_norms = {}

    for name, scale in scenarios.items():
        norms = simulate_gradient_propagation(
            sequence_length=sequence_length,
            recurrent_scale=scale,
        )

        final_norms[name] = norms[-1]

        print(f"{name} gradients (scale={scale}):")
        print(f"  Initial norm: {norms[0]:.6e}")
        print(f"  Final norm:   {norms[-1]:.6e}")
        print()

    # Verify the expected qualitative behavior.
    assert final_norms["Vanishing"] < final_norms["Stable"]
    assert final_norms["Exploding"] > final_norms["Stable"]

    print("Sequence-Length Experiment")
    print("-" * 60)

    for length in (5, 10, 20, 30):
        vanishing = simulate_gradient_propagation(length, 0.5)[-1]
        exploding = simulate_gradient_propagation(length, 1.5)[-1]

        print(
            f"Length {length:2d}: "
            f"vanishing={vanishing:.3e}, "
            f"exploding={exploding:.3e}"
        )

    demonstrate_gradient_clipping()

    print("\nAll qualitative checks passed.")


if __name__ == "__main__":
    main()