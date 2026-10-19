"""
Implement scaled dot-product attention from first principles.

This experiment covers:
1. Query-key similarity using matrix multiplication.
2. Scaling scores by the square root of the key dimension.
3. Numerically stable softmax normalization.
4. Attention masks for excluding selected positions.
5. Numerical verification of attention properties.
"""

import numpy as np


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable softmax along the specified axis."""
    shifted_scores = scores - np.max(scores, axis=axis, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

    Shapes:
        query: (..., query_length, key_dim)
        key:   (..., key_length, key_dim)
        value: (..., key_length, value_dim)
        mask:  Broadcastable to (..., query_length, key_length)
               True means the position is allowed to attend.

    Returns:
        context: (..., query_length, value_dim)
        weights: (..., query_length, key_length)
    """
    key_dim = query.shape[-1]

    if key_dim == 0:
        raise ValueError("The key dimension must be positive.")

    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Query and key dimensions must match.")

    if key.shape[-2] != value.shape[-2]:
        raise ValueError("Key and value sequence lengths must match.")

    scores = query @ np.swapaxes(key, -2, -1)
    scores = scores / np.sqrt(key_dim)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

        try:
            mask = np.broadcast_to(mask, scores.shape)
        except ValueError as error:
            raise ValueError(
                "Mask must be broadcastable to the attention score shape."
            ) from error

        if np.any(~np.any(mask, axis=-1)):
            raise ValueError(
                "Every query must have at least one allowed key position."
            )

        scores = np.where(mask, scores, -np.inf)

    weights = softmax(scores, axis=-1)
    context = weights @ value

    return context, weights


def main() -> None:
    # Two queries, three keys, and three-dimensional key vectors.
    query = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )

    key = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    value = np.array(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [5.0, 5.0],
        ]
    )

    context, weights = scaled_dot_product_attention(query, key, value)

    print("=== Scaled Dot-Product Attention ===")
    print(f"Query shape:   {query.shape}")
    print(f"Key shape:     {key.shape}")
    print(f"Value shape:   {value.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Context shape: {context.shape}")

    print("\nAttention weights:")
    print(np.round(weights, 4))

    print("\nContext vectors:")
    print(np.round(context, 4))

    # Verify output dimensions and probability distributions.
    assert weights.shape == (2, 3)
    assert context.shape == (2, 2)
    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=-1), 1.0)

    # Allow each query to attend only to the first two keys.
    mask = np.array([True, True, False])

    masked_context, masked_weights = scaled_dot_product_attention(
        query,
        key,
        value,
        mask=mask,
    )

    print("\n=== Attention with a Mask ===")
    print("Mask:", mask)
    print("Masked attention weights:")
    print(np.round(masked_weights, 4))

    print("\nMasked context vectors:")
    print(np.round(masked_context, 4))

    # The third key must receive zero attention weight.
    assert np.allclose(masked_weights[:, 2], 0.0)
    assert np.allclose(masked_weights.sum(axis=-1), 1.0)

    # Compare scaled and unscaled attention for the first query.
    raw_scores = query[0] @ key.T
    unscaled_weights = softmax(raw_scores)

    print("\n=== Effect of Scaling ===")
    print("Unscaled weights:")
    print(np.round(unscaled_weights, 4))
    print("Scaled weights:")
    print(np.round(weights[0], 4))

    assert not np.allclose(unscaled_weights, weights[0])

    print("\nAll scaled dot-product attention checks passed.")


if __name__ == "__main__":
    main()
