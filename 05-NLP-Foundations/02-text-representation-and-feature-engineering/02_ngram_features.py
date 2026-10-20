"""
Explore word and character n-grams for text feature engineering.

This experiment covers:
1. Extraction of word unigrams, bigrams, and trigrams.
2. Analysis of how n-gram ranges affect vocabulary size.
3. Comparison of word n-grams with character n-grams.
4. Inspection of n-gram feature counts across documents.
5. Analysis of the trade-off between contextual information and feature dimensionality.
6. Numerical verification of n-gram counts and feature matrix properties.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


DOCUMENTS = [
    "machine learning processes text",
    "deep learning processes language",
    "machine learning models understand language",
    "text processing supports machine learning",
]


def experiment_word_unigrams() -> None:
    """Extract individual word features."""
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    matrix = vectorizer.fit_transform(DOCUMENTS)

    print("=== Word Unigrams ===")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Features: {vectorizer.get_feature_names_out().tolist()}")

    assert matrix.shape == (len(DOCUMENTS), len(vectorizer.vocabulary_))


def experiment_word_bigrams() -> None:
    """Extract consecutive word pairs."""
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    matrix = vectorizer.fit_transform(DOCUMENTS)

    print("\n=== Word Bigrams ===")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Features: {vectorizer.get_feature_names_out().tolist()}")

    assert matrix.shape[0] == len(DOCUMENTS)
    assert all(len(term.split()) == 2 for term in vectorizer.get_feature_names_out())


def experiment_word_trigrams() -> None:
    """Extract consecutive sequences of three words."""
    vectorizer = CountVectorizer(ngram_range=(3, 3))
    matrix = vectorizer.fit_transform(DOCUMENTS)

    print("\n=== Word Trigrams ===")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Features: {vectorizer.get_feature_names_out().tolist()}")

    assert matrix.shape[0] == len(DOCUMENTS)
    assert all(len(term.split()) == 3 for term in vectorizer.get_feature_names_out())


def experiment_ngram_range_comparison() -> None:
    """Compare feature spaces containing different word n-gram ranges."""
    ranges = {
        "Unigrams": (1, 1),
        "Unigrams + bigrams": (1, 2),
        "Unigrams + bigrams + trigrams": (1, 3),
    }

    print("\n=== N-gram Range Comparison ===")

    vocabulary_sizes = {}

    for label, ngram_range in ranges.items():
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        matrix = vectorizer.fit_transform(DOCUMENTS)

        vocabulary_sizes[label] = len(vectorizer.vocabulary_)

        print(
            f"{label}: vocabulary={matrix.shape[1]}, "
            f"nonzero entries={matrix.nnz}, shape={matrix.shape}"
        )

    assert (
        vocabulary_sizes["Unigrams"]
        <= vocabulary_sizes["Unigrams + bigrams"]
        <= vocabulary_sizes["Unigrams + bigrams + trigrams"]
    )


def experiment_character_ngrams() -> None:
    """Extract character sequences from text."""
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
    )
    matrix = vectorizer.fit_transform(DOCUMENTS)

    features = vectorizer.get_feature_names_out()

    print("\n=== Character N-grams ===")
    print(f"Vocabulary size: {len(features)}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Sample features: {features[:20].tolist()}")

    assert matrix.shape[0] == len(DOCUMENTS)
    assert all(3 <= len(feature) <= 5 for feature in features)


def experiment_character_word_comparison() -> None:
    """Compare the dimensionality and sparsity of word and character features."""
    word_vectorizer = CountVectorizer(ngram_range=(1, 2))
    char_vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
    )

    word_matrix = word_vectorizer.fit_transform(DOCUMENTS)
    char_matrix = char_vectorizer.fit_transform(DOCUMENTS)

    def calculate_sparsity(matrix) -> float:
        total_entries = matrix.shape[0] * matrix.shape[1]
        return 1 - matrix.nnz / total_entries if total_entries else 0.0

    word_sparsity = calculate_sparsity(word_matrix)
    char_sparsity = calculate_sparsity(char_matrix)

    print("\n=== Word vs. Character Features ===")
    print(
        f"Word features: {word_matrix.shape[1]}, "
        f"sparsity={word_sparsity:.2%}"
    )
    print(
        f"Character features: {char_matrix.shape[1]}, "
        f"sparsity={char_sparsity:.2%}"
    )

    assert word_matrix.shape[0] == char_matrix.shape[0]
    assert np.all(word_matrix.data > 0)
    assert np.all(char_matrix.data > 0)


def main() -> None:
    experiment_word_unigrams()
    experiment_word_bigrams()
    experiment_word_trigrams()
    experiment_ngram_range_comparison()
    experiment_character_ngrams()
    experiment_character_word_comparison()


if __name__ == "__main__":
    main()