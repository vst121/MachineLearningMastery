# Word Embeddings and Semantic Similarity

## 🎯 Learning Objectives

- Understand how words are represented as dense numerical vectors.
- Compare dense embeddings with sparse one-hot representations.
- Measure semantic similarity using cosine similarity and Euclidean distance.
- Retrieve nearest neighbors and explore approximate word relationships.
- Evaluate embedding quality using human judgments and rank correlation.
- Understand the limitations of static word embeddings, including vocabulary coverage, contextual ambiguity, and bias.

## 🧠 Core Concepts

### 1. Word Embeddings

A word embedding maps a vocabulary item to a dense vector.

$$
f(w)=\mathbf{v}_w\in\mathbb{R}^{d}
$$

Where:

- \(w\) is a word.
- \(\mathbf{v}\_w\) is its embedding vector.
- \(d\) is the embedding dimension.

An embedding model can be represented as a matrix:

$$
E\in\mathbb{R}^{V\times d}
$$

Here, \(V\) is the vocabulary size and \(d\) is the vector dimension.

Unlike one-hot vectors, dense embeddings can represent distributional relationships between words through their learned numerical values.

### 2. One-Hot and Dense Representations

A one-hot representation assigns one active position to each vocabulary item:

$$
\mathbf{x}_w\in\{0,1\}^{V}
$$

A dense embedding represents the same word using a smaller, generally non-binary vector:

$$
\mathbf{v}_w\in\mathbb{R}^{d}
$$

When \(d\) is much smaller than \(V\), dense vectors require fewer values to store than full-length one-hot vectors. Dense representations also encode learned relationships that one-hot representations do not directly express.

### 3. Cosine Similarity

Cosine similarity measures the angle between two nonzero vectors.

$$
\operatorname{cosine}(\mathbf{u},\mathbf{v})
=
\frac{\mathbf{u}\cdot\mathbf{v}}
{\|\mathbf{u}\|_2\|\mathbf{v}\|_2}
$$

Its value lies between -1 and 1:

- 1: vectors point in the same direction.
- 0: vectors are orthogonal.
- -1: vectors point in opposite directions.

Cosine similarity is invariant to positive scaling of either vector. It measures directional similarity, not absolute vector magnitude.

### 4. Euclidean Distance

Euclidean distance measures the straight-line distance between two vectors.

$$
d(\mathbf{u},\mathbf{v})
=
\sqrt{\sum_{i=1}^{d}(u_i-v_i)^2}
$$

Unlike cosine similarity, Euclidean distance is affected by vector magnitude.

For unit-normalized vectors, cosine similarity and Euclidean distance are related:

$$
\|\hat{\mathbf{u}}-\hat{\mathbf{v}}\|_2^2
=
2\left(1-\operatorname{cosine}(\mathbf{u},\mathbf{v})\right)
$$

Therefore, ranking unit-normalized vectors by cosine similarity produces the same ordering as ranking them by increasing Euclidean distance.

### 5. Semantic Nearest Neighbors

For a query word \(q\), nearest-neighbor retrieval ranks other words by similarity to its vector.

$$
N_k(q)=
\operatorname{TopK}_{w\ne q}
\operatorname{cosine}(\mathbf{v}_q,\mathbf{v}_w)
$$

Nearest neighbors can reveal distributional relationships, but related words are not necessarily synonyms. Results depend on the embedding model, its training corpus, and the candidate vocabulary.

### 6. Word Analogies

Embedding arithmetic can explore approximate relationships between words.

$$
\mathbf{v}_{\text{target}}
\approx
\mathbf{v}_a-\mathbf{v}_b+\mathbf{v}_c
$$

The resulting vector can be compared with vocabulary vectors to retrieve nearby words. Such analogies are exploratory patterns, not guaranteed linguistic rules or factual relationships.

### 7. Embedding Evaluation

Embedding similarity scores can be compared with human similarity judgments. Spearman rank correlation measures the association between their rankings.

$$
\rho =
1-\frac{6\sum_{i=1}^{n}d_i^2}{n(n^2-1)}
$$

Here, \(d_i\) is the difference between the two ranks for pair \(i\). This simplified formula assumes no tied ranks.

Evaluation should use reliable human annotations or established benchmarks. A small collection of illustrative ratings is useful for demonstrating the method, but it cannot establish general embedding quality.

### 8. Vocabulary Coverage

For a test set of \(N\) words:

$$
\text{Coverage}=
\frac{\text{Words present in vocabulary}}{N}
$$

Static embedding models cannot directly retrieve vectors for out-of-vocabulary words. Coverage depends on the domain and vocabulary of the target application.

### 9. Limitations of Static Embeddings

A static embedding assigns a single vector to each word:

$$
f(w)=\mathbf{v}_w
$$

The vector does not change according to sentence context. For example, the word _bank_ may refer to a financial institution or the side of a river, but a static embedding returns the same vector for both uses.

Other limitations include:

- **Bias:** training data may encode social stereotypes.
- **Domain shift:** general-purpose embeddings may perform poorly on specialist language.
- **Semantic ambiguity:** similarity does not establish equivalence or entailment.
- **Vocabulary coverage:** unseen words have no directly retrievable vector.
- **Evaluation limitations:** intrinsic similarity scores do not guarantee downstream task performance.

## 🧪 Implementation and Experiments

| File                                         | Experiment                                                                                                                                                                                         |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_word_embedding_fundamentals.py`          | Load pretrained GloVe embeddings, inspect vector dimensions, compare one-hot and dense representations, and examine vocabulary coverage and embedding storage.                                     |
| `02_cosine_similarity_and_distance.py`       | Calculate cosine similarity and Euclidean distance, compare manual calculations with scikit-learn, examine vector magnitude, and verify metric edge cases.                                         |
| `03_semantic_nearest_neighbors.py`           | Retrieve nearest neighbors, rank a fixed candidate vocabulary, compare semantic categories, explore word analogies, and handle out-of-vocabulary queries.                                          |
| `04_embedding_evaluation_and_limitations.py` | Compare embedding scores with illustrative human judgments, calculate Spearman rank correlation, inspect static representations, measure sample vocabulary coverage, and review model limitations. |

### Tools and Libraries

- **NumPy:** vector operations, norms, and numerical verification.
- **scikit-learn:** cosine similarity and one-hot encoding.
- **Gensim:** loading and querying pretrained GloVe word embeddings.
- **SciPy:** Spearman rank correlation.

The experiments use `glove-wiki-gigaword-50`, a pretrained 50-dimensional GloVe embedding model available through Gensim. The first run downloads the model if it is not already cached.

## ✅ Expected Outcome

After completing this module, you should be able to:

- Explain the purpose and structure of dense word embeddings.
- Calculate and interpret cosine similarity and Euclidean distance.
- Retrieve and analyze semantic nearest neighbors.
- Explain the assumptions behind vector-based word analogies.
- Evaluate embedding rankings against human judgments.
- Identify limitations involving context, bias, vocabulary coverage, and domain mismatch.
- Distinguish intrinsic embedding evaluation from evaluation on downstream tasks.

These concepts provide the foundation for understanding contextual embeddings and pretrained transformer models in the next NLP modules.
