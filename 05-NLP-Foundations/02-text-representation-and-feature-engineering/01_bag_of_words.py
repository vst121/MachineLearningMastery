"""
Explore Bag of Words representations for numerical text feature extraction.

This experiment covers:
1. Vocabulary construction and term-to-index mapping.
2. Document-term matrix generation using term frequencies.
3. Comparison of raw term counts and binary occurrence features.
4. Analysis of repeated terms and their effect on feature values.
5. Measurement of document-term matrix sparsity.
6. Numerical verification of matrix dimensions and feature properties.
"""

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


DOCUMENTS = [
    "NLP transforms text into numerical features",
    "Text features support machine learning",
    "NLP and machine learning use text",
    "Numerical features represent documents",
]


def experiment_vocabulary_construction() -> None:
    """Build a vocabulary and inspect its term-to-index mapping."""
    vectorizer = CountVectorizer()
    vectorizer.fit(DOCUMENTS)

    vocabulary = vectorizer.vocabulary_

    print("=== Vocabulary Construction ===")
    print(f"Document count: {len(DOCUMENTS)}")
    print(f"Vocabulary size: {len(vocabulary)}")

    for term, index in sorted(vocabulary.items(), key=lambda item: item[1]):
        print(f"{index:2}: {term}")

    assert len(vocabulary) == len(set(vocabulary))


def experiment_document_term_matrix() -> None:
    """Represent documents using raw term frequencies."""
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(DOCUMENTS)
    terms = vectorizer.get_feature_names_out()

    print("\n=== Document-Term Matrix ===")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Nonzero entries: {matrix.nnz}")

    print("\nTerms:")
    print(terms.tolist())

    print("\nDense matrix:")
    print(matrix.toarray())

    assert matrix.shape == (len(DOCUMENTS), len(terms))
    assert np.all(matrix.toarray() >= 0)


def experiment_term_frequency() -> None:
    """Compare term counts across documents."""
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(DOCUMENTS)
    terms = vectorizer.get_feature_names_out()

    print("\n=== Term Frequency ===")

    selected_terms = ["text", "features", "nlp", "machine"]

    for term in selected_terms:
        if term not in vectorizer.vocabulary_:
            continue

        index = vectorizer.vocabulary_[term]
        counts = matrix[:, index].toarray().ravel()

        print(f"{term}: {counts.tolist()}")

    row_totals = np.asarray(matrix.sum(axis=1)).ravel()
    print(f"Tokens counted per document: {row_totals.tolist()}")

    assert np.all(row_totals > 0)


def experiment_binary_features() -> None:
    """Compare raw counts with binary word occurrence."""
    count_vectorizer = CountVectorizer()
    binary_vectorizer = CountVectorizer(binary=True)

    count_matrix = count_vectorizer.fit_transform(DOCUMENTS)
    binary_matrix = binary_vectorizer.fit_transform(DOCUMENTS)

    terms = count_vectorizer.get_feature_names_out()

    print("\n=== Count vs. Binary Features ===")
    print(f"Vocabulary: {terms.tolist()}")
    print(f"Count matrix:\n{count_matrix.toarray()}")
    print(f"Binary matrix:\n{binary_matrix.toarray()}")

    binary_values = binary_matrix.toarray()
    assert np.all((binary_values == 0) | (binary_values == 1))

    count_values = count_matrix.toarray()
    assert np.all(binary_values <= count_values)


def experiment_repeated_terms() -> None:
    """Demonstrate how repeated words affect count and binary features."""
    documents = [
        "model model learns",
        "model learns",
    ]

    count_vectorizer = CountVectorizer()
    binary_vectorizer = CountVectorizer(binary=True)

    count_matrix = count_vectorizer.fit_transform(documents).toarray()
    binary_matrix = binary_vectorizer.fit_transform(documents).toarray()
    terms = count_vectorizer.get_feature_names_out()

    model_index = list(terms).index("model")

    print("\n=== Repeated Terms ===")
    print(f"Vocabulary: {terms.tolist()}")
    print(f"'model' counts: {count_matrix[:, model_index].tolist()}")
    print(f"'model' binary values: {binary_matrix[:, model_index].tolist()}")

    assert count_matrix[0, model_index] == 2
    assert binary_matrix[0, model_index] == 1


def experiment_matrix_sparsity() -> None:
    """Measure how many entries in the document-term matrix are zero."""
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(DOCUMENTS)

    total_entries = matrix.shape[0] * matrix.shape[1]
    nonzero_entries = matrix.nnz
    sparsity = 1 - nonzero_entries / total_entries

    print("\n=== Matrix Sparsity ===")
    print(f"Matrix dimensions: {matrix.shape}")
    print(f"Total entries: {total_entries}")
    print(f"Nonzero entries: {nonzero_entries}")
    print(f"Sparsity: {sparsity:.2%}")

    assert 0 <= sparsity <= 1


def main() -> None:
    experiment_vocabulary_construction()
    experiment_document_term_matrix()
    experiment_term_frequency()
    experiment_binary_features()
    experiment_repeated_terms()
    experiment_matrix_sparsity()


if __name__ == "__main__":
    main()