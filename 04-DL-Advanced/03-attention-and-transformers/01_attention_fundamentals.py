"""
Explore the fundamental concepts behind attention mechanisms.

This experiment covers:
1. Queries, keys, and values using a simple numerical example.
2. Similarity scores between a query and candidate keys.
3. Softmax normalization into attention weights.
4. Context vector computation as a weighted sum of values.
5. The effect of changing attention scores on the output.
"""

import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    """Convert scores into probabilities using numerically stable softmax."""
    shifted_scores = scores - np.max(scores)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores)


def compute_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute similarity scores, attention weights, and context vector."""
    scores = keys @ query
    weights = softmax(scores)
    context = weights @ values

    return scores, weights, context


def main() -> None:
    # Three input tokens represented by two-dimensional key vectors.
    keys = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    # Each token carries a value vector containing information to aggregate.
    values = np.array(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [5.0, 5.0],
        ]
    )

    # The query describes the information we want to retrieve.
    query = np.array([1.0, 0.2])

    scores, weights, context = compute_attention(query, keys, values)

    print("=== Attention Fundamentals ===")
    print(f"Query shape:  {query.shape}")
    print(f"Keys shape:   {keys.shape}")
    print(f"Values shape: {values.shape}")

    print("\nSimilarity scores:")
    print(scores)

    print("\nAttention weights:")
    for index, weight in enumerate(weights):
        print(f"Token {index + 1}: {weight:.4f}")

    print(f"\nSum of attention weights: {weights.sum():.4f}")
    print(f"Context vector: {context}")
    print(f"Context shape:  {context.shape}")

    # Verify basic attention properties.
    assert scores.shape == (3,)
    assert weights.shape == (3,)
    assert context.shape == (2,)
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(context, weights @ values)

    # Change the query to observe how attention shifts.
    second_query = np.array([0.2, 1.0])
    _, second_weights, second_context = compute_attention(
        second_query,
        keys,
        values,
    )

    print("\n=== Changing the Query ===")
    print(f"Original query: {query}")
    print(f"Original weights: {weights}")
    print(f"Original context: {context}")

    print(f"\nNew query: {second_query}")
    print(f"New weights: {second_weights}")
    print(f"New context: {second_context}")

    assert not np.allclose(weights, second_weights)
    assert not np.allclose(context, second_context)

    print("\nAll attention fundamentals checks passed.")


if __name__ == "__main__":
    main()
