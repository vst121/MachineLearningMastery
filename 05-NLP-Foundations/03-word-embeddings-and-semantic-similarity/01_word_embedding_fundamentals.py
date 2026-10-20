"""
Explore the fundamentals of dense word embeddings.

This experiment covers:
1. Loading pretrained Word2Vec embeddings.
2. Inspecting embedding dimensions and vocabulary size.
3. Retrieving dense vectors for individual words.
4. Comparing dense embeddings with one-hot representations.
5. Checking vocabulary coverage and out-of-vocabulary words.
6. Verifying embedding shapes and numerical properties.
"""

import numpy as np
from gensim.downloader import load
from sklearn.preprocessing import OneHotEncoder


MODEL_NAME = "glove-wiki-gigaword-50"


def load_embedding_model():
    """Load pretrained word embeddings from gensim's model repository."""
    print(f"Loading pretrained embeddings: {MODEL_NAME}")
    print("The first run may download the model.")

    model = load(MODEL_NAME)

    print("Model loaded successfully.")
    return model


def experiment_embedding_dimensions(model) -> None:
    """Inspect vocabulary size and embedding dimensions."""
    vector_size = model.vector_size
    vocabulary_size = len(model.key_to_index)

    print("=== Embedding Dimensions ===")
    print(f"Vocabulary size: {vocabulary_size:,}")
    print(f"Embedding dimensions: {vector_size}")
    print(f"Approximate float32 vector size: {vector_size * 4} bytes")

    assert vocabulary_size > 0
    assert vector_size > 0


def experiment_word_vectors(model) -> None:
    """Retrieve and inspect dense vectors for selected words."""
    words = ["king", "queen", "computer", "language"]

    print("\n=== Word Vectors ===")

    for word in words:
        if word not in model:
            print(f"{word}: not in vocabulary")
            continue

        vector = model[word]

        print(f"\nWord: {word}")
        print(f"Shape: {vector.shape}")
        print(f"First 10 dimensions: {np.round(vector[:10], 4)}")
        print(f"L2 norm: {np.linalg.norm(vector):.4f}")

        assert vector.shape == (model.vector_size,)
        assert np.all(np.isfinite(vector))


def experiment_one_hot_vs_dense(model) -> None:
    """Compare sparse one-hot vectors with dense word embeddings."""
    words = ["king", "queen", "computer"]

    encoder = OneHotEncoder(sparse_output=True, dtype=np.float32)
    one_hot_matrix = encoder.fit_transform(np.array(words).reshape(-1, 1))

    dense_matrix = np.vstack([model[word] for word in words])

    print("\n=== One-Hot vs. Dense Embeddings ===")
    print(f"Word count: {len(words)}")
    print(f"One-hot matrix shape: {one_hot_matrix.shape}")
    print(f"One-hot nonzero entries: {one_hot_matrix.nnz}")
    print(f"Dense embedding matrix shape: {dense_matrix.shape}")
    print(f"Dense embedding nonzero entries: {np.count_nonzero(dense_matrix)}")

    print("\nOne-hot representation:")
    print(one_hot_matrix.toarray())

    print("\nDense embeddings (first 5 dimensions):")
    print(np.round(dense_matrix[:, :5], 4))

    assert one_hot_matrix.shape == (len(words), len(words))
    assert dense_matrix.shape == (len(words), model.vector_size)
    assert one_hot_matrix.nnz == len(words)


def experiment_vocabulary_coverage(model) -> None:
    """Measure vocabulary coverage for a sample text."""
    text = (
        "The intelligent system processes language and retrieves information "
        "from documents using machine learning."
    )

    tokens = [
        token.lower().strip(".,!?;:")
        for token in text.split()
    ]
    tokens = [token for token in tokens if token]

    in_vocabulary = [token for token in tokens if token in model]
    out_of_vocabulary = [token for token in tokens if token not in model]

    coverage = len(in_vocabulary) / len(tokens) if tokens else 0.0

    print("\n=== Vocabulary Coverage ===")
    print(f"Token count: {len(tokens)}")
    print(f"In-vocabulary tokens: {len(in_vocabulary)}")
    print(f"Out-of-vocabulary tokens: {len(out_of_vocabulary)}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Out-of-vocabulary examples: {out_of_vocabulary}")

    assert len(in_vocabulary) + len(out_of_vocabulary) == len(tokens)
    assert 0 <= coverage <= 1


def experiment_embedding_table(model) -> None:
    """Inspect a small section of the conceptual embedding table."""
    words = ["king", "queen", "man", "woman"]
    available_words = [word for word in words if word in model]

    embedding_matrix = np.vstack([model[word] for word in available_words])

    print("\n=== Embedding Table ===")
    print(f"Selected words: {available_words}")
    print(f"Embedding table shape: {embedding_matrix.shape}")
    print(f"Storage for selected float32 vectors: {embedding_matrix.nbytes} bytes")

    assert embedding_matrix.shape == (
        len(available_words),
        model.vector_size,
    )
    assert np.all(np.isfinite(embedding_matrix))


def main() -> None:
    model = load_embedding_model()

    experiment_embedding_dimensions(model)
    experiment_word_vectors(model)
    experiment_one_hot_vs_dense(model)
    experiment_vocabulary_coverage(model)
    experiment_embedding_table(model)


if __name__ == "__main__":
    main()