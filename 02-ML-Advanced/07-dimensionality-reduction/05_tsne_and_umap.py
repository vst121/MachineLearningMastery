# %% [markdown]
# # t-SNE and UMAP
#
# t-SNE and UMAP are nonlinear dimensionality reduction techniques.
# They are particularly useful for visualizing high-dimensional data in 2D.
#
# Unlike PCA, which finds linear directions of maximum variance, these
# methods aim to preserve aspects of the neighborhood structure of data.
#
# Important:
# - These visualizations are useful for exploration, not proof of clusters.
# - Distances between separate clusters and the axis values require care.
# - The resulting dimensions do not have direct, interpretable meanings.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## 1. Load and standardize the Iris dataset
#
# Standardization prevents features with larger numerical scales from
# dominating distance calculations.

# %%
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset shape: {X_scaled.shape}")
print(f"Classes: {target_names}")

# %% [markdown]
# ## 2. Reduce dimensions with t-SNE
#
# Perplexity influences the effective neighborhood size considered by t-SNE.
# It must be smaller than the number of samples.
#
# The random state makes this example reproducible.

# %%
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    max_iter=1000,
    random_state=42,
)

X_tsne = tsne.fit_transform(X_scaled)

print(f"t-SNE output shape: {X_tsne.shape}")
print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")

# %% [markdown]
# ## 3. Visualize the t-SNE projection
#
# Each point represents one Iris sample.
# Colors represent the known species labels.

# %%
def plot_embedding(X_embedded, y, target_names, title):
    fig, ax = plt.subplots(figsize=(9, 6))

    scatter = ax.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=y,
        cmap="viridis",
        s=65,
        alpha=0.85,
    )

    handles, _ = scatter.legend_elements()
    ax.legend(handles, target_names, title="Species")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


plot_embedding(
    X_tsne,
    y,
    target_names,
    "Iris Dataset: t-SNE Projection",
)

# %% [markdown]
# ## 4. Reduce dimensions with UMAP
#
# UMAP is a nonlinear technique that can preserve local neighborhood structure
# while often retaining more of the broader structure than t-SNE.
#
# Install it if needed:
#
# pip install umap-learn

# %%
try:
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        n_jobs=1,
    )

    X_umap = reducer.fit_transform(X_scaled)

    print(f"UMAP output shape: {X_umap.shape}")

    plot_embedding(
        X_umap,
        y,
        target_names,
        "Iris Dataset: UMAP Projection",
    )

except ImportError:
    print(
        "\nUMAP is not installed. Install it with "
        "'pip install umap-learn' to run the UMAP example."
    )

# %% [markdown]
# ## 5. Understand the key parameters
#
# t-SNE:
# - perplexity: balances attention to local and broader neighborhoods.
# - learning_rate: controls the optimization step size.
# - random_state: helps make results reproducible.
#
# UMAP:
# - n_neighbors: controls the neighborhood scale used to learn the embedding.
# - min_dist: influences how tightly points can be packed together.
# - random_state: helps make results reproducible.
#
# ## Key takeaways
#
# - PCA is a linear technique; t-SNE and UMAP are nonlinear techniques.
# - t-SNE is especially useful for exploring local neighborhoods.
# - UMAP is useful for visualization and can also be used for feature reduction.
# - Neither method guarantees meaningful clusters or class separation.
# - Embeddings can change with parameters, random initialization, and data.
# - For predictive workflows, fit transformations on training data only.
