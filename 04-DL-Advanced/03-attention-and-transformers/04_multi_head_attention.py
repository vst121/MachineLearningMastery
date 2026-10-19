"""
Implement multi-head attention using NumPy.

This experiment covers:
1. Projecting token embeddings into queries, keys, and values.
2. Splitting representations into multiple attention heads.
3. Computing scaled dot-product attention independently per head.
4. Concatenating head outputs and applying an output projection.
5. Verifying tensor dimensions and attention properties.
"""

import numpy as np


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable softmax along an axis."""
    shifted_scores = scores - np.max(scores, axis=axis, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    exp_scores = np.exp(scores - np.max(scores, axis=axis, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)


def split_heads(
    tensor: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """Reshape (sequence_length, model_dim) into (num_heads, sequence_length, head_dim)."""
    sequence_length, model_dim = tensor.shape

    if model_dim % num_heads != 0:
        raise ValueError("model_dim must be divisible by num_heads.")

    head_dim = model_dim // num_heads
    tensor = tensor.reshape(sequence_length, num_heads, head_dim)

    return np.transpose(tensor, (1, 0, 2))


def combine_heads(tensor: np.ndarray) -> np.ndarray:
    """Combine (num_heads, sequence_length, head_dim) into (sequence_length, model_dim)."""
    num_heads, sequence_length, head_dim = tensor.shape
    tensor = np.transpose(tensor, (1, 0, 2))

    return tensor.reshape(sequence_length, num_heads * head_dim)


def multi_head_attention(
    inputs: np.ndarray,
    w_query: np.ndarray,
    w_key: np.ndarray,
    w_value: np.ndarray,
    w_output: np.ndarray,
    num_heads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply multi-head self-attention.

    Args:
        inputs:   (sequence_length, model_dim)
        w_query:  (model_dim, model_dim)
        w_key:    (model_dim, model_dim)
        w_value:  (model_dim, model_dim)
        w_output: (model_dim, model_dim)
        num_heads: Number of attention heads.

    Returns:
        output:  (sequence_length, model_dim)
        weights: (num_heads, sequence_length, sequence_length)
    """
    if inputs.ndim != 2:
        raise ValueError("Inputs must be a two-dimensional array.")

    if num_heads <= 0:
        raise ValueError("num_heads must be positive.")

    model_dim = inputs.shape[-1]

    if model_dim % num_heads != 0:
        raise ValueError("model_dim must be divisible by num_heads.")

    expected_shape = (model_dim, model_dim)

    for name, matrix in (
        ("w_query", w_query),
        ("w_key", w_key),
        ("w_value", w_value),
        ("w_output", w_output),
    ):
        if matrix.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}.")

    query = split_heads(inputs @ w_query, num_heads)
    key = split_heads(inputs @ w_key, num_heads)
    value = split_heads(inputs @ w_value, num_heads)

    head_dim = query.shape[-1]
    scores = query @ np.swapaxes(key, -2, -1) / np.sqrt(head_dim)
    weights = softmax(scores, axis=-1)

    head_outputs = weights @ value
    concatenated = combine_heads(head_outputs)
    output = concatenated @ w_output

    return output, weights


def main() -> None:
    rng = np.random.default_rng(42)

    sequence_length = 4
    model_dim = 8
    num_heads = 2

    inputs = rng.normal(size=(sequence_length, model_dim))

    # In a trained model, these matrices are learned parameters.
    w_query = rng.normal(size=(model_dim, model_dim)) * 0.2
    w_key = rng.normal(size=(model_dim, model_dim)) * 0.2
    w_value = rng.normal(size=(model_dim, model_dim)) * 0.2
    w_output = rng.normal(size=(model_dim, model_dim)) * 0.2

    output, weights = multi_head_attention(
        inputs,
        w_query,
        w_key,
        w_value,
        w_output,
        num_heads,
    )

    head_dim = model_dim // num_heads

    print("=== Multi-Head Attention ===")
    print(f"Input shape:       {inputs.shape}")
    print(f"Model dimension:   {model_dim}")
    print(f"Number of heads:   {num_heads}")
    print(f"Head dimension:    {head_dim}")
    print(f"Attention weights: {weights.shape}")
    print(f"Output shape:      {output.shape}")

    for head_index, head_weights in enumerate(weights):
        print(f"\nHead {head_index + 1} attention weights:")
        print(np.round(head_weights, 3))

    print("\nOutput representations:")
    print(np.round(output, 3))

    # Verify shape and probability invariants.
    assert weights.shape == (num_heads, sequence_length, sequence_length)
    assert output.shape == (sequence_length, model_dim)
    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=-1), 1.0)

    # Each head has its own attention distribution.
    assert not np.allclose(weights[0], weights[1])

    # Invalid head dimensions must be rejected.
    try:
        split_heads(np.zeros((sequence_length, 7)), num_heads)
    except ValueError:
        print("\nInvalid head dimension correctly rejected.")
    else:
        raise AssertionError("Expected an error for incompatible dimensions.")

    print("\nAll multi-head attention checks passed.")


if __name__ == "__main__":
    main()
