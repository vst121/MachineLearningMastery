"""
Explore Term Frequency–Inverse Document Frequency (TF-IDF) for text representation.

This experiment covers:
1. Calculation of term frequency and inverse document frequency.
2. Comparison of raw term counts with TF-IDF weights.
3. Analysis of how document frequency affects term importance.
4. Comparison of scikit-learn's smoothed IDF with a manual calculation.
5. Inspection of TF-IDF document-term matrices and feature weights.
6. Numerical verification of TF-IDF values and matrix properties.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


DOCUMENTS = [
    "machine learning models process text",
    "machine learning improves text analysis",
    "text analysis uses statistical models",
    "deep learning models process language",
]


def experiment_term_frequency() -> None:
    """Calculate normalized term frequency for each document."""
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(DOCUMENTS)

    row_totals = np.asarray(count_matrix.sum(axis=1)).ravel()
    tf_matrix = count_matrix.toarray() / row_totals[:, np.newaxis]

    terms = vectorizer.get_feature_names_out()

    print("=== Term Frequency ===")
    print(f"Vocabulary: {terms.tolist()}")
    print(f"Count matrix:\n{count_matrix.toarray()}")
    print(f"Normalized TF matrix:\n{np.round(tf_matrix, 3)}")

    assert tf_matrix.shape == count_matrix.shape
    assert np.all(tf_matrix >= 0)
    assert np.allclose(tf_matrix.sum(axis=1), 1.0)


def experiment_document_frequency() -> None:
    """Inspect the number of documents containing each term."""
    vectorizer = CountVectorizer(binary=True)
    binary_matrix = vectorizer.fit_transform(DOCUMENTS)

    document_frequencies = np.asarray(binary_matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    print("\n=== Document Frequency ===")

    for term, frequency in sorted(
        zip(terms, document_frequencies),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"{term}: {frequency}/{len(DOCUMENTS)} documents")

    assert np.all(document_frequencies >= 1)
    assert np.all(document_frequencies <= len(DOCUMENTS))


def experiment_manual_idf() -> None:
    """Calculate smoothed IDF and compare it with scikit-learn."""
    vectorizer = CountVectorizer(binary=True)
    binary_matrix = vectorizer.fit_transform(DOCUMENTS)

    document_frequencies = np.asarray(binary_matrix.sum(axis=0)).ravel()
    document_count = len(DOCUMENTS)

    manual_idf = (
        np.log((1 + document_count) / (1 + document_frequencies)) + 1
    )

    tfidf_vectorizer = TfidfVectorizer(
        use_idf=True,
        smooth_idf=True,
        norm=None,
    )
    tfidf_vectorizer.fit(DOCUMENTS)

    library_idf = tfidf_vectorizer.idf_
    terms = tfidf_vectorizer.get_feature_names_out()

    print("\n=== Manual IDF vs. scikit-learn ===")

    for term, manual, library in zip(terms, manual_idf, library_idf):
        print(f"{term}: manual={manual:.4f}, sklearn={library:.4f}")

    assert np.allclose(manual_idf, library_idf)


def experiment_tf_idf_matrix() -> None:
    """Generate and inspect a normalized TF-IDF matrix."""
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(DOCUMENTS)

    terms = vectorizer.get_feature_names_out()

    print("\n=== TF-IDF Matrix ===")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Nonzero entries: {matrix.nnz}")
    print(f"Vocabulary: {terms.tolist()}")
    print(f"TF-IDF matrix:\n{np.round(matrix.toarray(), 3)}")

    row_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())

    print(f"Document vector L2 norms: {np.round(row_norms, 6)}")

    assert matrix.shape == (len(DOCUMENTS), len(terms))
    assert np.allclose(row_norms, 1.0)


def experiment_common_vs_rare_terms() -> None:
    """Compare IDF weights for common and rare terms."""
    documents = [
        "common word appears",
        "common word appears again",
        "common word appears frequently",
        "rare concept appears",
    ]

    vectorizer = TfidfVectorizer(norm=None)
    matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()
    idf_values = vectorizer.idf_

    idf_by_term = dict(zip(terms, idf_values))

    print("\n=== Common vs. Rare Terms ===")
    print(f"IDF for 'common': {idf_by_term['common']:.4f}")
    print(f"IDF for 'rare': {idf_by_term['rare']:.4f}")
    print(f"IDF for 'appears': {idf_by_term['appears']:.4f}")

    assert idf_by_term["rare"] > idf_by_term["common"]
    assert idf_by_term["common"] > idf_by_term["appears"]

    rare_index = list(terms).index("rare")
    common_index = list(terms).index("common")

    print(f"'rare' TF-IDF values: {matrix[:, rare_index].toarray().ravel()}")
    print(f"'common' TF-IDF values: {matrix[:, common_index].toarray().ravel()}")


def experiment_count_vs_tfidf() -> None:
    """Compare raw counts and TF-IDF weights for the same collection."""
    count_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer()

    count_matrix = count_vectorizer.fit_transform(DOCUMENTS)
    tfidf_matrix = tfidf_vectorizer.fit_transform(DOCUMENTS)

    terms = count_vectorizer.get_feature_names_out()

    print("\n=== Count vs. TF-IDF ===")
    print(f"Vocabulary: {terms.tolist()}")
    print(f"Raw counts:\n{count_matrix.toarray()}")
    print(f"TF-IDF weights:\n{np.round(tfidf_matrix.toarray(), 3)}")

    assert count_matrix.shape == tfidf_matrix.shape
    assert np.all(count_matrix.data >= 1)
    assert np.all(tfidf_matrix.data > 0)


def main() -> None:
    experiment_term_frequency()
    experiment_document_frequency()
    experiment_manual_idf()
    experiment_tf_idf_matrix()
    experiment_common_vs_rare_terms()
    experiment_count_vs_tfidf()


if __name__ == "__main__":
    main()