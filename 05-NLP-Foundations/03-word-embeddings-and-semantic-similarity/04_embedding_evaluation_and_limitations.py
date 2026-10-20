"""
Evaluate word embeddings and investigate their limitations.

This experiment covers:
1. Comparing embedding similarity with human-rated word similarity.
2. Measuring rank correlation between model scores and human judgments.
3. Evaluating whether nearest neighbors reflect expected relationships.
4. Demonstrating out-of-vocabulary limitations of static embeddings.
5. Exploring how polysemous words receive a single static representation.
6. Summarizing limitations involving bias, context, and evaluation coverage.
"""

import numpy as np
from gensim.downloader import load
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "glove-wiki-gigaword-50"


def pair_similarity(model, word_a: str, word_b: str) -> float | None:
    """Return cosine similarity, or None if either word is unavailable."""
    if word_a not in model or word_b not in model:
        return None

    vector_a = model[word_a].reshape(1, -1)
    vector_b = model[word_b].reshape(1, -1)

    return float(cosine_similarity(vector_a, vector_b)[0, 0])


def experiment_human_similarity_correlation(model) -> None:
    """Compare embedding scores with illustrative human similarity ratings."""
    # Illustrative ratings on a 0-10 scale, not a published benchmark.
    word_pairs = [
        ("car", "automobile", 9.0),
        ("teacher", "educator", 8.5),
        ("coffee", "tea", 6.5),
        ("computer", "keyboard", 5.5),
        ("dog", "banana", 0.5),
        ("river", "mountain", 2.5),
        ("king", "queen", 8.0),
        ("doctor", "hospital", 5.0),
        ("book", "library", 5.5),
        ("music", "guitar", 5.0),
    ]

    model_scores = []
    human_scores = []
    included_pairs = []

    for word_a, word_b, human_score in word_pairs:
        score = pair_similarity(model, word_a, word_b)

        if score is None:
            print(f"Skipping unavailable pair: ({word_a}, {word_b})")
            continue

        model_scores.append(score)
        human_scores.append(human_score)
        included_pairs.append((word_a, word_b, human_score, score))

    print("\n=== Human Similarity Correlation ===")
    print("Ratings are illustrative; they are not ground-truth benchmark labels.")

    for word_a, word_b, human_score, model_score in included_pairs:
        print(
            f"{word_a:12} | {word_b:12} | "
            f"human={human_score:4.1f} | embedding={model_score:.4f}"
        )

    if len(included_pairs) < 3:
        print("Not enough available pairs to calculate a meaningful correlation.")
        return

    correlation = spearmanr(human_scores, model_scores)

    print(f"\nSpearman rank correlation: {correlation.statistic:.4f}")
    print(f"Evaluated word pairs: {len(included_pairs)}")

    assert np.isfinite(correlation.statistic)


def experiment_neighborhood_quality(model) -> None:
    """Inspect nearest neighbors and discuss their limitations."""
    queries = ["computer", "doctor", "king", "river"]

    print("\n=== Nearest-Neighbor Inspection ===")

    for query in queries:
        if query not in model:
            print(f"\nSkipping unavailable query: {query}")
            continue

        neighbors = model.most_similar(query, topn=5)

        print(f"\nQuery: {query}")
        for word, score in neighbors:
            print(f"  {word:15} similarity={score:.4f}")

        # Similarity is a ranking signal, not proof of equivalence.
        assert all(np.isfinite(score) for _, score in neighbors)
        assert all(word != query for word, _ in neighbors)


def experiment_static_representation(model) -> None:
    """Show that a static embedding assigns one vector per word."""
    word = "bank"

    print("\n=== Static Embedding Representation ===")

    if word not in model:
        print(f"'{word}' is unavailable in the vocabulary.")
        return

    vector = model[word]

    print(f"Word: {word}")
    print(f"Vector shape: {vector.shape}")
    print(f"Vector norm: {np.linalg.norm(vector):.4f}")
    print(f"First 10 dimensions: {np.round(vector[:10], 4)}")

    # A static embedding has no sentence-specific input.
    financial_context = "She deposited money at the bank."
    geographical_context = "They sat on the bank of the river."

    print("\nTwo possible contexts:")
    print(f"  1. {financial_context}")
    print(f"  2. {geographical_context}")
    print("The lookup model returns the same vector for 'bank' in both contexts.")


def experiment_out_of_vocabulary(model) -> None:
    """Demonstrate the vocabulary coverage limitation of static embeddings."""
    words = [
        "computer",
        "language",
        "Mannheim",
        "unseenwordxyz987",
    ]

    print("\n=== Vocabulary Coverage ===")

    available = 0

    for word in words:
        in_vocabulary = word in model
        available += int(in_vocabulary)
        print(f"{word:20} available={in_vocabulary}")

    coverage = available / len(words)

    print(f"\nCoverage of this small test set: {coverage:.2%}")
    print("This small sample is illustrative, not a reliable estimate of general coverage.")

    assert 0.0 <= coverage <= 1.0


def experiment_embedding_limitations(model) -> None:
    """Summarize practical limitations to consider when evaluating embeddings."""
    limitations = {
        "Context": "Static vectors do not adapt to the sentence in which a word appears.",
        "Vocabulary": "Words absent from the vocabulary have no directly retrievable vector.",
        "Bias": "Training text may encode social stereotypes and historical biases.",
        "Domain shift": "General-purpose embeddings may not represent specialist language well.",
        "Evaluation": "Similarity scores do not directly measure task performance.",
        "Coverage": "A small set of word pairs cannot establish general embedding quality.",
    }

    print("\n=== Embedding Limitations ===")

    for topic, description in limitations.items():
        print(f"{topic:14} {description}")

    print("\nEvaluation should use representative data and the intended downstream task.")


def main() -> None:
    print(f"Loading pretrained embeddings: {MODEL_NAME}")
    model = load(MODEL_NAME)

    experiment_human_similarity_correlation(model)
    experiment_neighborhood_quality(model)
    experiment_static_representation(model)
    experiment_out_of_vocabulary(model)
    experiment_embedding_limitations(model)


if __name__ == "__main__":
    main()