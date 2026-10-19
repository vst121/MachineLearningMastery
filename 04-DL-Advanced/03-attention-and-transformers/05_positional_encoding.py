"""
Explore sinusoidal positional encoding for Transformer models.

This experiment covers:
1. Generating sinusoidal positional encodings.
2. Understanding sine and cosine frequencies across dimensions.
3. Combining positional encodings with token embeddings.
4. Inspecting positional patterns across sequence positions.
5. Verifying dimensions and numerical properties.
"""

import numpy as np


def sinusoidal_positional_encoding(
    sequence_length: int,
    model_dim: int,
) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.

    Returns:
        Array with shape (sequence_length, model_dim).
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    if model_dim <= 0:
        raise ValueError("model_dim must be positive.")

    positions = np.arange(sequence_length, dtype=np.float64)[:, np.newaxis]
    dimensions = np.arange(model_dim, dtype=np.float64)[np.newaxis, :]

    angle_rates = np.power(
        10000.0,
        -(2.0 * np.floor(dimensions / 2.0)) / model_dim,
    )
    angles = positions * angle_rates

    encoding = np.empty((sequence_length, model_dim), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])

    return encoding


def main() -> None:
    sequence_length = 6
    model_dim = 8

    positions = sinusoidal_positional_encoding(
        sequence_length,
        model_dim,
    )

    print("=== Sinusoidal Positional Encoding ===")
    print(f"Sequence length: {sequence_length}")
    print(f"Model dimension: {model_dim}")
    print(f"Encoding shape:  {positions.shape}")

    print("\nPositional encoding matrix:")
    print(np.round(positions, 4))

    # Each row represents one position in the sequence.
    assert positions.shape == (sequence_length, model_dim)

    # Sinusoidal values must remain between -1 and 1.
    assert np.all(positions >= -1.0)
    assert np.all(positions <= 1.0)

    # Position zero has sine values of zero and cosine values of one.
    expected_first_position = np.zeros(model_dim)
    expected_first_position[1::2] = 1.0

    assert np.allclose(positions[0], expected_first_position)

    # Different positions should have different encodings.
    assert not np.allclose(positions[1], positions[2])

    # Demonstrate how positional information is added to embeddings.
    rng = np.random.default_rng(42)
    token_embeddings = rng.normal(size=(sequence_length, model_dim))

    combined_embeddings = token_embeddings + positions

    print("\n=== Adding Positional Information ===")
    print(f"Token embeddings shape: {token_embeddings.shape}")
    print(f"Combined shape:         {combined_embeddings.shape}")

    print("\nFirst token embedding:")
    print(np.round(token_embeddings[0], 4))

    print("\nPosition encoding for the first token:")
    print(np.round(positions[0], 4))

    print("\nCombined representation of the first token:")
    print(np.round(combined_embeddings[0], 4))

    # Positional information changes the token representations.
    assert combined_embeddings.shape == token_embeddings.shape
    assert np.allclose(combined_embeddings, token_embeddings + positions)
    assert not np.allclose(combined_embeddings, token_embeddings)

    # Odd model dimensions are supported.
    odd_dim_encoding = sinusoidal_positional_encoding(
        sequence_length=4,
        model_dim=5,
    )
    assert odd_dim_encoding.shape == (4, 5)

    print("\nOdd-dimensional encoding shape:", odd_dim_encoding.shape)
    print("\nAll positional encoding checks passed.")


if __name__ == "__main__":
    main()
