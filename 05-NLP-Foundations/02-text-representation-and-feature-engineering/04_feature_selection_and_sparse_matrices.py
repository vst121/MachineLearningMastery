"""
Explore sparse text representations and feature selection techniques.

This experiment covers:
1. Measurement of document-term matrix sparsity.
2. Vocabulary reduction using minimum and maximum document frequency.
3. Removal of low-information features using variance thresholds.
4. Supervised feature selection using chi-squared scores.
5. Comparison of feature dimensionality and matrix storage.
6. Verification that feature selection preserves valid numerical representations.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2


DOCUMENTS = [
    "machine learning models classify text",
    "machine learning improves prediction",
    "text classification uses machine learning",
    "deep learning models process images",
    "image classification uses deep learning",
    "models learn useful representations",
    "text models classify documents",
    "images contain visual information",
]

LABELS = np.array([1, 1, 1, 0, 0, 1, 1, 0])


def calculate_sparsity(matrix: csr_matrix) -> float:
    """Return the proportion of zero-valued entries in a matrix."""
    total_entries = matrix.shape[0] * matrix.shape[1]

    if total_entries == 0:
        return 0.0

    return 1 - matrix.nnz / total_entries


def experiment_matrix_sparsity() -> None:
    """Measure sparsity in count and TF-IDF representations."""
    count_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer()

    count_matrix = count_vectorizer.fit_transform(DOCUMENTS)
    tfidf_matrix = tfidf_vectorizer.fit_transform(DOCUMENTS)

    print("=== Matrix Sparsity ===")

    for name, matrix in [
        ("Count", count_matrix),
        ("TF-IDF", tfidf_matrix),
    ]:
        print(f"{name} matrix shape: {matrix.shape}")
        print(f"{name} nonzero entries: {matrix.nnz}")
        print(f"{name} sparsity: {calculate_sparsity(matrix):.2%}")

        assert matrix.shape[0] == len(DOCUMENTS)
        assert 0 <= calculate_sparsity(matrix) <= 1


def experiment_document_frequency_filtering() -> None:
    """Filter terms according to how many documents contain them."""
    vectorizer = CountVectorizer(min_df=2)
    matrix = vectorizer.fit_transform(DOCUMENTS)

    terms = vectorizer.get_feature_names_out()

    print("\n=== Minimum Document Frequency ===")
    print(f"Original document count: {len(DOCUMENTS)}")
    print(f"Retained vocabulary size: {len(terms)}")
    print(f"Retained terms: {terms.tolist()}")
    print(f"Filtered matrix shape: {matrix.shape}")

    binary_vectorizer = CountVectorizer(binary=True)
    binary_matrix = binary_vectorizer.fit_transform(DOCUMENTS)
    document_frequencies = np.asarray(binary_matrix.sum(axis=0)).ravel()
    original_terms = binary_vectorizer.get_feature_names_out()

    frequencies = dict(zip(original_terms, document_frequencies))

    assert all(frequencies[term] >= 2 for term in terms)


def experiment_maximum_document_frequency() -> None:
    """Remove terms appearing in too many documents."""
    vectorizer = CountVectorizer(max_df=0.75)
    matrix = vectorizer.fit_transform(DOCUMENTS)

    terms = vectorizer.get_feature_names_out()

    print("\n=== Maximum Document Frequency ===")
    print(f"Retained vocabulary size: {len(terms)}")
    print(f"Retained terms: {terms.tolist()}")
    print(f"Matrix shape: {matrix.shape}")

    binary_vectorizer = CountVectorizer(binary=True)
    binary_matrix = binary_vectorizer.fit_transform(DOCUMENTS)

    document_frequencies = np.asarray(binary_matrix.sum(axis=0)).ravel()
    original_terms = binary_vectorizer.get_feature_names_out()
    frequencies = dict(zip(original_terms, document_frequencies))

    maximum_frequency = 0.75 * len(DOCUMENTS)

    assert all(frequencies[term] <= maximum_frequency for term in terms)


def experiment_variance_threshold() -> None:
    """Remove features with zero variance."""
    documents = [
        "machine learning text",
        "machine learning images",
        "machine learning models",
        "machine learning prediction",
    ]

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(documents)

    selector = VarianceThreshold(threshold=0.0)
    selected_matrix = selector.fit_transform(matrix)

    original_terms = vectorizer.get_feature_names_out()
    selected_terms = original_terms[selector.get_support()]

    print("\n=== Variance Threshold ===")
    print(f"Original vocabulary: {original_terms.tolist()}")
    print(f"Selected vocabulary: {selected_terms.tolist()}")
    print(f"Original dimensions: {matrix.shape}")
    print(f"Selected dimensions: {selected_matrix.shape}")

    assert selected_matrix.shape[0] == matrix.shape[0]
    assert selected_matrix.shape[1] <= matrix.shape[1]
    assert "machine" not in selected_terms
    assert "learning" not in selected_terms


def experiment_chi_squared_selection() -> None:
    """Select features associated with the target classes."""
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(DOCUMENTS)

    selector = SelectKBest(score_func=chi2, k=5)
    selected_matrix = selector.fit_transform(matrix, LABELS)

    all_terms = vectorizer.get_feature_names_out()
    selected_terms = all_terms[selector.get_support()]
    scores = selector.scores_[selector.get_support()]

    print("\n=== Chi-Squared Feature Selection ===")
    print(f"Original vocabulary size: {len(all_terms)}")
    print(f"Selected vocabulary size: {len(selected_terms)}")
    print(f"Selected terms: {selected_terms.tolist()}")

    for term, score in sorted(
        zip(selected_terms, scores),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{term}: chi2={score:.4f}")

    assert selected_matrix.shape == (len(DOCUMENTS), 5)
    assert np.all(scores >= 0)


def experiment_sparse_storage() -> None:
    """Compare sparse matrix storage with dense array storage."""
    vectorizer = CountVectorizer()
    sparse_matrix = vectorizer.fit_transform(DOCUMENTS)
    dense_matrix = sparse_matrix.toarray()

    dense_bytes = dense_matrix.nbytes

    # CSR storage includes data, column indices, and row pointers.
    sparse_bytes = (
        sparse_matrix.data.nbytes
        + sparse_matrix.indices.nbytes
        + sparse_matrix.indptr.nbytes
    )

    print("\n=== Sparse vs. Dense Storage ===")
    print(f"Matrix shape: {sparse_matrix.shape}")
    print(f"Nonzero entries: {sparse_matrix.nnz}")
    print(f"Dense storage: {dense_bytes} bytes")
    print(f"CSR storage: {sparse_bytes} bytes")

    assert np.array_equal(dense_matrix, sparse_matrix.toarray())
    assert sparse_bytes > 0
    assert dense_bytes > 0


def main() -> None:
    experiment_matrix_sparsity()
    experiment_document_frequency_filtering()
    experiment_maximum_document_frequency()
    experiment_variance_threshold()
    experiment_chi_squared_selection()
    experiment_sparse_storage()


if __name__ == "__main__":
    main()