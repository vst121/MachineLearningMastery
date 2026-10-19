"""
Implement self-attention with learnable query, key, and value projections.

This experiment covers:
1. Projecting token embeddings into queries, keys, and values.
2. Computing scaled dot-product self-attention.
3. Understanding attention and context tensor dimensions.
4. Inspecting token-to-token attention weights.
5. Verifying output shapes and attention properties.
"""

import numpy as np


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable softmax along an axis."""
    shifted_scores = scores - np.max(scores, axis=axis, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)


def self_attention(
    inputs: np.ndarray,
    w_query: np.ndarray,
    w_key: np.ndarray,
    w_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply self-attention to a sequence of token embeddings.

    Shapes:
        inputs:  (sequence_length, input_dim)
        w_query: (input_dim, key_dim)
        w_key:   (input_dim, key_dim)
        w_value: (input_dim, value_dim)

    Returns:
        context: (sequence_length, value_dim)
        weights: (sequence_length, sequence_length)
    """
    if inputs.ndim != 2:
        raise ValueError("Inputs must have shape (sequence_length, input_dim).")

    if inputs.shape[0] == 0:
        raise ValueError("The input sequence must not be empty.")

    if w_query.shape[0] != inputs.shape[1]:
        raise ValueError("Query projection input dimension does not match.")

    if w_key.shape[0] != inputs.shape[1]:
        raise ValueError("Key projection input dimension does not match.")

    if w_value.shape[0] != inputs.shape[1]:
        raise ValueError("Value projection input dimension does not match.")

    if w_query.shape[1] != w_key.shape[1]:
        raise ValueError("Query and key projection dimensions must match.")

    query = inputs @ w_query
    key = inputs @ w_key
    value = inputs @ w_value

    key_dim = key.shape[-1]
    scores = query @ key.T / np.sqrt(key_dim)
    weights = softmax(scores, axis=-1)
    context = weights @ value

    return context, weights


def main() -> None:
    # Three tokens, each represented by four embedding features.
    inputs = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 2.0],
        ]
    )

    # Fixed projection matrices make the experiment reproducible.
    # In a neural network, these matrices are trainable parameters.
    w_query = np.array(
        [
            [0.2, 0.1],
            [0.0, 0.3],
            [0.4, 0.2],
            [0.1, 0.5],
        ]
    )

    w_key = np.array(
        [
            [0.3, 0.2],
            [0.1, 0.4],
            [0.5, 0.1],
            [0.2, 0.3],
        ]
    )

    w_value = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.5, 0.5, 1.0],
            [1.0, 0.5, 0.0],
        ]
    )

    query = inputs @ w_query
    key = inputs @ w_key
    value = inputs @ w_value

    context, weights = self_attention(
        inputs,
        w_query,
        w_key,
        w_value,
    )

    print("=== Self-Attention ===")
    print(f"Input shape:   {inputs.shape}")
    print(f"Query shape:   {query.shape}")
    print(f"Key shape:     {key.shape}")
    print(f"Value shape:   {value.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Context shape: {context.shape}")

    print("\nAttention weights:")
    print(np.round(weights, 4))

    print("\nContextual token representations:")
    print(np.round(context, 4))

    # Each row describes how one query token attends to all key tokens.
    assert weights.shape == (3, 3)
    assert context.shape == (3, 3)
    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=-1), 1.0)

    # Verify that the context is the weighted sum of projected values.
    assert np.allclose(context, weights @ value)

    # Demonstrate that changing token content can change attention.
    changed_inputs = inputs.copy()
    changed_inputs[0] = np.array([0.0, 0.0, 0.0, 3.0])

    changed_context, changed_weights = self_attention(
        changed_inputs,
        w_query,
        w_key,
        w_value,
    )

    print("\n=== After Changing the First Token ===")
    print("Updated attention weights:")
    print(np.round(changed_weights, 4))

    print("\nUpdated contextual representations:")
    print(np.round(changed_context, 4))

    assert not np.allclose(weights, changed_weights)

    print("\nAll self-attention checks passed.")


if __name__ == "__main__":
    main()
