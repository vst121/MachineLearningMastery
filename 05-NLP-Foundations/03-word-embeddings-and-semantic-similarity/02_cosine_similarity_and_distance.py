"""
Explore similarity and distance metrics for word embeddings.

This experiment covers:
1. Manual calculation of cosine similarity using NumPy.
2. Comparison with scikit-learn's cosine similarity implementation.
3. Calculation of Euclidean distance between word vectors.
4. Comparison of similarity rankings with distance rankings.
5. Analysis of the effect of vector magnitude on similarity metrics.
6. Numerical verification of metric properties and edge cases.
"""

import numpy as np
from gensim.downloader import load
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "glove-wiki-gigaword-50"


def manual_cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """Calculate cosine similarity and handle zero vectors safely."""
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def euclidean_distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """Calculate Euclidean distance between two vectors."""
    return float(np.linalg.norm(vector_a - vector_b))


def experiment_manual_vs_library(model) -> None:
    """Verify manual cosine similarity against scikit-learn."""
    word_pairs = [
        ("king", "queen"),
        ("cat", "dog"),
        ("computer", "keyboard"),
        ("car", "banana"),
    ]

    print("=== Manual vs. Library Cosine Similarity ===")

    for word_a, word_b in word_pairs:
        if word_a not in model or word_b not in model:
            print(f"Skipping unavailable pair: {word_a}, {word_b}")
            continue

        vector_a = model[word_a]
        vector_b = model[word_b]

        manual_score = manual_cosine_similarity(vector_a, vector_b)
        library_score = float(
            cosine_similarity(
                vector_a.reshape(1, -1),
                vector_b.reshape(1, -1),
            )[0, 0]
        )

        print(
            f"{word_a:10} vs. {word_b:10} | "
            f"manual={manual_score:.4f}, sklearn={library_score:.4f}"
        )

        assert np.isclose(manual_score, library_score, atol=1e-6)


def experiment_similarity_and_distance(model) -> None:
    """Compare cosine similarity and Euclidean distance for word pairs."""
    word_pairs = [
        ("king", "queen"),
        ("cat", "dog"),
        ("computer", "keyboard"),
        ("car", "banana"),
    ]

    print("\n=== Cosine Similarity vs. Euclidean Distance ===")

    for word_a, word_b in word_pairs:
        if word_a not in model or word_b not in model:
            continue

        vector_a = model[word_a]
        vector_b = model[word_b]

        similarity = manual_cosine_similarity(vector_a, vector_b)
        distance = euclidean_distance(vector_a, vector_b)

        print(
            f"{word_a:10} vs. {word_b:10} | "
            f"cosine={similarity:.4f}, distance={distance:.4f}"
        )

        assert -1.000001 <= similarity <= 1.000001
        assert distance >= 0


def experiment_nearest_neighbors(model) -> None:
    """Rank candidate words using cosine similarity and Euclidean distance."""
    query = "computer"
    candidates = [
        "keyboard",
        "software",
        "technology",
        "banana",
        "network",
        "screen",
        "fruit",
        "program",
    ]

    if query not in model:
        raise ValueError(f"Query word '{query}' is not in the embedding vocabulary.")

    available_candidates = [
        word for word in candidates if word in model and word != query
    ]

    if not available_candidates:
        raise ValueError("No candidate words are available in the vocabulary.")

    query_vector = model[query]
    candidate_vectors = np.vstack([model[word] for word in available_candidates])

    similarities = cosine_similarity(
        query_vector.reshape(1, -1),
        candidate_vectors,
    ).ravel()

    distances = np.linalg.norm(candidate_vectors - query_vector, axis=1)

    cosine_ranking = np.argsort(-similarities)
    distance_ranking = np.argsort(distances)

    print(f"\n=== Nearest Neighbors: {query} ===")

    print("\nCosine similarity ranking:")
    for index in cosine_ranking:
        print(
            f"{available_candidates[index]:12} "
            f"similarity={similarities[index]:.4f}"
        )

    print("\nEuclidean distance ranking:")
    for index in distance_ranking:
        print(
            f"{available_candidates[index]:12} "
            f"distance={distances[index]:.4f}"
        )


def experiment_vector_magnitude() -> None:
    """Demonstrate scale invariance of cosine similarity."""
    vector_a = np.array([1.0, 2.0, 3.0])
    vector_b = np.array([2.0, 4.0, 6.0])
    vector_c = np.array([1.0, 2.0, 4.0])

    similarity_ab = manual_cosine_similarity(vector_a, vector_b)
    similarity_ac = manual_cosine_similarity(vector_a, vector_c)

    distance_ab = euclidean_distance(vector_a, vector_b)
    distance_ac = euclidean_distance(vector_a, vector_c)

    print("\n=== Effect of Vector Magnitude ===")
    print(f"Cosine similarity A vs. B: {similarity_ab:.4f}")
    print(f"Cosine similarity A vs. C: {similarity_ac:.4f}")
    print(f"Euclidean distance A vs. B: {distance_ab:.4f}")
    print(f"Euclidean distance A vs. C: {distance_ac:.4f}")

    assert np.isclose(similarity_ab, 1.0)
    assert not np.isclose(distance_ab, 0.0)
    assert similarity_ac < similarity_ab


def experiment_edge_cases() -> None:
    """Verify identical, orthogonal, and opposite-vector behavior."""
    vector_a = np.array([1.0, 0.0])
    vector_same = np.array([2.0, 0.0])
    vector_orthogonal = np.array([0.0, 1.0])
    vector_opposite = np.array([-1.0, 0.0])

    same_score = manual_cosine_similarity(vector_a, vector_same)
    orthogonal_score = manual_cosine_similarity(vector_a, vector_orthogonal)
    opposite_score = manual_cosine_similarity(vector_a, vector_opposite)

    print("\n=== Cosine Similarity Edge Cases ===")
    print(f"Same direction: {same_score:.4f}")
    print(f"Orthogonal vectors: {orthogonal_score:.4f}")
    print(f"Opposite directions: {opposite_score:.4f}")

    assert np.isclose(same_score, 1.0)
    assert np.isclose(orthogonal_score, 0.0)
    assert np.isclose(opposite_score, -1.0)


def main() -> None:
    print(f"Loading pretrained embeddings: {MODEL_NAME}")
    model = load(MODEL_NAME)

    experiment_manual_vs_library(model)
    experiment_similarity_and_distance(model)
    experiment_nearest_neighbors(model)
    experiment_vector_magnitude()
    experiment_edge_cases()


if __name__ == "__main__":
    main()