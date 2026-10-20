"""
Explore semantic nearest-neighbor retrieval using pretrained word embeddings.

This experiment covers:
1. Retrieving the nearest words to a query using cosine similarity.
2. Comparing nearest neighbors across different semantic categories.
3. Excluding the query word from its own neighbor results.
4. Comparing similarity rankings with different candidate vocabularies.
5. Exploring word analogies as approximate vector relationships.
6. Verifying ranking order and handling out-of-vocabulary words.
"""

import numpy as np
from gensim.downloader import load
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "glove-wiki-gigaword-50"


def get_nearest_neighbors(
    model,
    query: str,
    top_k: int = 5,
    candidate_words: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Return words ranked by cosine similarity to a query vector."""
    if query not in model:
        raise ValueError(f"Query word '{query}' is not in the vocabulary.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if candidate_words is None:
        return [
            (word, float(score))
            for word, score in model.most_similar(query, topn=top_k)
        ]

    candidates = [
        word
        for word in dict.fromkeys(candidate_words)
        if word in model and word != query
    ]

    if not candidates:
        return []

    vectors = np.vstack([model[word] for word in candidates])
    query_vector = model[query].reshape(1, -1)

    scores = cosine_similarity(query_vector, vectors).ravel()
    ranking = np.argsort(-scores)[:top_k]

    return [
        (candidates[index], float(scores[index]))
        for index in ranking
    ]


def experiment_nearest_neighbors(model) -> None:
    """Retrieve nearest neighbors for words from different categories."""
    queries = [
        "computer",
        "doctor",
        "king",
        "language",
        "school",
    ]

    print("=== Semantic Nearest Neighbors ===")

    for query in queries:
        if query not in model:
            print(f"\nSkipping unavailable query: {query}")
            continue

        neighbors = get_nearest_neighbors(model, query, top_k=5)

        print(f"\nQuery: {query}")
        for word, score in neighbors:
            print(f"  {word:15} cosine similarity={score:.4f}")

        assert len(neighbors) <= 5
        assert all(word != query for word, _ in neighbors)
        assert all(
            neighbors[index][1] >= neighbors[index + 1][1]
            for index in range(len(neighbors) - 1)
        )


def experiment_candidate_vocabulary(model) -> None:
    """Rank a fixed candidate vocabulary for a given query."""
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
        "hardware",
        "database",
    ]

    neighbors = get_nearest_neighbors(
        model,
        query,
        top_k=5,
        candidate_words=candidates,
    )

    print("\n=== Candidate Vocabulary Search ===")
    print(f"Query: {query}")
    print(f"Candidate count: {len(candidates)}")

    for word, score in neighbors:
        print(f"{word:15} cosine similarity={score:.4f}")

    assert all(word in candidates for word, _ in neighbors)
    assert all(word != query for word, _ in neighbors)


def experiment_semantic_categories(model) -> None:
    """Compare similarity scores across selected semantic categories."""
    query = "doctor"

    categories = {
        "Medical": ["nurse", "hospital", "patient", "medicine"],
        "Technology": ["computer", "software", "network", "keyboard"],
        "Nature": ["forest", "river", "tree", "mountain"],
    }

    print("\n=== Semantic Category Comparison ===")
    print(f"Query: {query}")

    if query not in model:
        raise ValueError(f"Query word '{query}' is not in the vocabulary.")

    query_vector = model[query].reshape(1, -1)

    category_scores = {}

    for category, words in categories.items():
        available_words = [word for word in words if word in model]

        if not available_words:
            print(f"{category}: no available words")
            continue

        vectors = np.vstack([model[word] for word in available_words])
        scores = cosine_similarity(query_vector, vectors).ravel()

        mean_score = float(np.mean(scores))
        category_scores[category] = mean_score

        print(f"\n{category}: mean similarity={mean_score:.4f}")

        for word, score in sorted(
            zip(available_words, scores),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  {word:15} {score:.4f}")

    assert all(np.isfinite(score) for score in category_scores.values())


def experiment_word_analogies(model) -> None:
    """Explore approximate vector relationships using analogy arithmetic."""
    analogies = [
        ("king", "man", "woman"),
        ("paris", "france", "germany"),
    ]

    print("\n=== Word Analogy Experiments ===")

    for positive_a, negative, positive_b in analogies:
        words = [positive_a, negative, positive_b]

        if not all(word in model for word in words):
            print(f"Skipping unavailable analogy: {words}")
            continue

        results = model.most_similar(
            positive=[positive_a, positive_b],
            negative=[negative],
            topn=5,
        )

        print(f"\n{positive_a} - {negative} + {positive_b}")

        for word, score in results:
            print(f"  {word:15} cosine similarity={score:.4f}")

        assert all(word not in words for word, _ in results)


def experiment_out_of_vocabulary(model) -> None:
    """Handle queries that do not exist in the embedding vocabulary."""
    queries = [
        "computer",
        "word_not_in_this_embedding_vocabulary_123",
    ]

    print("\n=== Out-of-Vocabulary Queries ===")

    for query in queries:
        if query in model:
            neighbors = get_nearest_neighbors(model, query, top_k=3)
            print(f"{query}: {neighbors}")
        else:
            print(f"{query}: unavailable; no vector can be retrieved")

    assert queries[0] in model
    assert queries[1] not in model


def main() -> None:
    print(f"Loading pretrained embeddings: {MODEL_NAME}")
    model = load(MODEL_NAME)

    experiment_nearest_neighbors(model)
    experiment_candidate_vocabulary(model)
    experiment_semantic_categories(model)
    experiment_word_analogies(model)
    experiment_out_of_vocabulary(model)


if __name__ == "__main__":
    main()