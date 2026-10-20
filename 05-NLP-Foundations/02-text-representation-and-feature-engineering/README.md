# Text Representation and Feature Engineering

## Learning Objectives

- Convert text documents into numerical feature representations.
- Understand vocabulary construction and document-term matrices.
- Extract word and character n-gram features.
- Calculate TF-IDF weights and interpret their significance.
- Understand sparse matrices, feature selection, and dimensionality reduction.
- Prepare numerical text features for classical machine learning models.

## Core Concepts

### 1. Bag of Words

Bag of Words represents documents using word frequencies from a shared vocabulary, disregarding word order.

For a dataset containing \(D\) documents and a vocabulary of \(V\) terms, the document-term matrix is:

$$
X \in \mathbb{N}^{D \times V}
$$

Each matrix entry represents the frequency of a vocabulary term in a document:

$$
X_{ij} = \operatorname{count}(t_j,d_i)
$$

Binary features record only whether a term appears:

$$
X_{ij}^{\text{binary}} =
\begin{cases}
1 & \text{if } t_j \in d_i \\
0 & \text{otherwise}
\end{cases}
$$

The number of token occurrences represented in document \(i\) is:

$$
L_i = \sum_{j=1}^{V}X_{ij}
$$

Implementation: `01_bag_of_words.py`

Experiments:
- Vocabulary construction and term-to-index mapping.
- Document-term matrix generation.
- Raw term counts versus binary features.
- Repeated terms and document lengths.
- Matrix sparsity.

### 2. N-gram Features

An n-gram is a sequence of \(n\) consecutive tokens. For a sequence of length \(L\), the number of possible contiguous n-grams is:

$$
N_n = \max(0,L-n+1)
$$

For five tokens:

$$
N_1=5,\qquad N_2=4,\qquad N_3=3
$$

- **Unigrams** represent individual words.
- **Bigrams** represent consecutive word pairs.
- **Trigrams** represent sequences of three words.
- **Character n-grams** represent character sequences and can capture partial words, spelling patterns, and morphological similarities.

If \(V_n\) is the set of unique n-grams of size \(n\), the combined vocabulary is:

$$
V_{\text{total}} =
\left|\bigcup_{n=1}^{k}V_n\right|
$$

Combining n-gram sizes preserves more local context but can substantially increase dimensionality.

Implementation: `02_ngram_features.py`

Experiments:
- Word unigrams, bigrams, and trigrams.
- Comparison of different n-gram ranges.
- Character n-grams.
- Vocabulary dimensionality and matrix sparsity.

### 3. Term Frequency–Inverse Document Frequency

TF-IDF gives higher weights to terms that are important within a document but relatively uncommon across the document collection.

Term frequency can be normalized by the number of token occurrences:

$$
\operatorname{TF}(t,d)
=
\frac{f_{t,d}}{\sum_{w \in d}f_{w,d}}
$$

Document frequency counts how many documents contain a term:

$$
\operatorname{DF}(t)
=
\left|\{d \in D:t \in d\}\right|
$$

With smoothed inverse document frequency:

$$
\operatorname{IDF}(t)
=
\log\left(\frac{1+D}{1+\operatorname{DF}(t)}\right)+1
$$

The conceptual TF-IDF formulation is:

$$
\operatorname{TFIDF}(t,d)
=
\operatorname{TF}(t,d)\operatorname{IDF}(t)
$$

The default `TfidfVectorizer` in scikit-learn uses raw term counts multiplied by IDF, followed by L2 normalization. It does not use the normalized TF formula above.

For a nonzero document vector \(x_d\), L2 normalization produces:

$$
\hat{x}_d =
\frac{x_d}{\sqrt{\sum_{j=1}^{V}x_{dj}^{2}}}
$$

Consequently:

$$
\sum_{j=1}^{V}\hat{x}_{dj}^{2}=1
$$

Implementation: `03_tf_idf.py`

Experiments:
- Normalized term-frequency calculation.
- Document-frequency analysis.
- Manual verification of smoothed IDF.
- TF-IDF matrix generation and L2 normalization.
- Common versus rare term weights.
- Raw count versus TF-IDF representations.

### 4. Feature Selection and Sparse Matrices

Text feature spaces can become large, especially when using word and character n-grams. However, each document usually contains only a small subset of the vocabulary.

Matrix sparsity is the proportion of zero-valued entries:

$$
\operatorname{Sparsity}(X)
=
1-\frac{\operatorname{nnz}(X)}{DV}
$$

where \(\operatorname{nnz}(X)\) is the number of nonzero entries.

Document-frequency filtering reduces the vocabulary by retaining terms that occur in a suitable number of documents.

Minimum document frequency:

$$
\operatorname{DF}(t)\geq m
$$

Maximum document-frequency proportion:

$$
\frac{\operatorname{DF}(t)}{D}\leq\tau
$$

Variance-based selection can remove constant features. For feature \(j\), its population variance is:

$$
\operatorname{Var}(X_j)
=
\frac{1}{D}\sum_{i=1}^{D}(x_{ij}-\mu_j)^2
$$

where:

$$
\mu_j=\frac{1}{D}\sum_{i=1}^{D}x_{ij}
$$

Supervised feature selection can use the chi-squared statistic to measure the association between features and class labels:

$$
\chi^2=\sum_i\frac{(O_i-E_i)^2}{E_i}
$$

Here, \(O_i\) and \(E_i\) represent observed and expected counts in a contingency table.

Dense matrices store every entry, while compressed sparse row (CSR) matrices store nonzero values, column indices, and row pointers. Their approximate storage complexities are:

$$
\text{Dense: }O(DV)
$$

$$
\text{CSR: }O(\operatorname{nnz}(X)+D)
$$

Implementation: `04_feature_selection_and_sparse_matrices.py`

Experiments:
- Count and TF-IDF matrix sparsity.
- Minimum and maximum document-frequency filtering.
- Variance thresholding.
- Chi-squared feature selection.
- Dense versus CSR storage.

**Data leakage prevention:** In predictive workflows, split the data before fitting vectorizers or feature selectors. Fit them on training data only, then transform validation and test data with the fitted objects.

## Implementation and Experiments

| File | Main experiment | Key outcome |
|---|---|---|
| `01_bag_of_words.py` | Bag of Words | Vocabulary and count/binary matrices |
| `02_ngram_features.py` | Word and character n-grams | Contextual features and dimensionality |
| `03_tf_idf.py` | TF-IDF | Term importance and normalized vectors |
| `04_feature_selection_and_sparse_matrices.py` | Feature selection | Reduced feature spaces and efficient storage |

The implementations use scikit-learn, NumPy, and SciPy. Run individual experiments from the repository root using `uv run python` followed by the relevant script path.

## Expected Outcome

After completing these experiments, you should be able to:

- Explain how text becomes a document-term matrix.
- Choose between raw counts, binary features, word n-grams, character n-grams, and TF-IDF.
- Calculate and interpret term frequency, document frequency, and inverse document frequency.
- Measure sparsity and understand sparse matrix storage.
- Apply vocabulary filtering and feature selection appropriately.
- Avoid data leakage when preparing features for predictive models.