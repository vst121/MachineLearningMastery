# %% [markdown]
# # Truncated SVD and Latent Semantic Analysis
#
# Truncated SVD reduces the dimensionality of a matrix without requiring
# the matrix to be centered. This makes it especially useful for sparse data.
#
# In text processing, TF-IDF converts documents into sparse numerical vectors.
# Truncated SVD then projects those vectors into a lower-dimensional space.
# This combination is commonly known as Latent Semantic Analysis (LSA).

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# %% [markdown]
# ## 1. Create a small text dataset
#
# The documents cover three broad themes: machine learning, cooking, and sports.
# Using a small built-in dataset keeps the example reproducible and avoids downloads.

# %%
documents = [
    "Machine learning models learn patterns from training data.",
    "Deep learning uses neural networks to learn complex patterns.",
    "Artificial intelligence systems use data to make predictions.",
    "A classification model predicts the class of a new observation.",
    "The neural network was trained using a large dataset.",
    "Cooking pasta requires boiling water and adding salt.",
    "Fresh vegetables and herbs improve the flavor of a meal.",
    "A chef prepares delicious meals using fresh ingredients.",
    "Baking bread requires flour, water, yeast, and time.",
    "A healthy recipe includes vegetables and balanced ingredients.",
    "Football players train regularly before competitive matches.",
    "The team won the match after scoring two goals.",
    "Athletes improve performance through practice and fitness.",
    "The coach prepared the team for the football tournament.",
    "Players need strength and endurance to compete in sports.",
]

document_labels = [
    "Machine learning",
    "Machine learning",
    "Machine learning",
    "Machine learning",
    "Machine learning",
    "Cooking",
    "Cooking",
    "Cooking",
    "Cooking",
    "Cooking",
    "Sports",
    "Sports",
    "Sports",
    "Sports",
    "Sports",
]

# %% [markdown]
# ## 2. Convert documents into sparse TF-IDF vectors
#
# Each row represents a document and each column represents a vocabulary term.
# TF-IDF assigns higher importance to terms that are informative for a document.
#
# Unlike a dense NumPy array, the resulting sparse matrix stores only nonzero values.

# %%
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)

print("Original document-term matrix:")
print(f"Documents: {X.shape[0]}")
print(f"Vocabulary size: {X.shape[1]}")
print(f"Matrix type: {type(X).__name__}")
print(f"Nonzero entries: {X.nnz}")

total_entries = X.shape[0] * X.shape[1]
sparsity = 1 - (X.nnz / total_entries)

print(f"Sparsity: {sparsity:.2%}")

# %% [markdown]
# ## 3. Reduce dimensionality with Truncated SVD
#
# We reduce each document to two numerical components for visualization.
# Truncated SVD works directly with sparse matrices and does not require centering.
#
# The components are mathematical directions, not necessarily interpretable topics.

# %%
svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(X)

print("\nReduced matrix:")
print(f"Shape: {X_reduced.shape}")
print(f"Explained variance ratio per component: {svd.explained_variance_ratio_}")
print(
    "Total explained variance ratio: "
    f"{svd.explained_variance_ratio_.sum():.2%}"
)

# %% [markdown]
# ## 4. Visualize documents in two dimensions
#
# Documents with similar vocabulary may appear closer together.
# The colors show the known document categories, not labels learned by SVD.

# %%
colors = {
    "Machine learning": "tab:blue",
    "Cooking": "tab:orange",
    "Sports": "tab:green",
}

fig, ax = plt.subplots(figsize=(10, 7))

for label in sorted(set(document_labels)):
    indices = [
        i for i, document_label in enumerate(document_labels)
        if document_label == label
    ]

    ax.scatter(
        X_reduced[indices, 0],
        X_reduced[indices, 1],
        label=label,
        color=colors[label],
        s=80,
        alpha=0.8,
    )

ax.set_title("Truncated SVD: Documents in Two Dimensions")
ax.set_xlabel("SVD Component 1")
ax.set_ylabel("SVD Component 2")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Inspect the most influential terms
#
# Each SVD component assigns a weight to every vocabulary term.
# These weights can help us understand which terms contribute to each component.
#
# A component's meaning is not guaranteed to correspond to a single topic.

# %%
feature_names = np.array(vectorizer.get_feature_names_out())
n_top_terms = min(10, len(feature_names))

for component_index, component in enumerate(svd.components_):
    top_indices = np.argsort(np.abs(component))[::-1][:n_top_terms]

    print(f"\nComponent {component_index + 1}:")
    for index in top_indices:
        print(f"  {feature_names[index]:<20} {component[index]:.4f}")

# %% [markdown]
# ## Key takeaways
#
# - TF-IDF represents text as a sparse document-term matrix.
# - Truncated SVD reduces dimensionality without centering the input matrix.
# - TF-IDF + Truncated SVD is commonly called Latent Semantic Analysis (LSA).
# - SVD can support document visualization, semantic search, and feature reduction.
# - Explained variance ratios from Truncated SVD should be interpreted carefully,
#   especially when working with uncentered sparse matrices.
# - In a predictive workflow, fit the vectorizer and SVD only on training data.
#   A scikit-learn Pipeline helps prevent data leakage.
