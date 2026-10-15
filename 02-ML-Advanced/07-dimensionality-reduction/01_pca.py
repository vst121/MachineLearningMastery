# %% [markdown]
# # Principal Component Analysis (PCA)
#
# PCA is a linear dimensionality reduction technique that transforms
# the original features into a smaller set of orthogonal components.
#
# We will:
# 1. Load the Iris dataset.
# 2. Standardize the features.
# 3. Apply PCA to reduce four features to two components.
# 4. Analyze explained variance.
# 5. Visualize the transformed dataset.
#
# Dataset: Iris

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# ## 1. Load the Dataset
#
# Iris contains four numerical features and three flower species.
# PCA does not use the target labels when calculating components.

# %%
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Target classes: {target_names}")

print("\nOriginal feature names:")
print(feature_names)


# %% [markdown]
# ## 2. Standardize the Features
#
# PCA is sensitive to feature scales because it identifies directions
# of maximum variance.
#
# StandardScaler transforms each feature to have approximately
# zero mean and unit variance.

# %%
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Original feature means:")
print(np.round(X.mean(axis=0), 2))

print("\nStandardized feature means:")
print(np.round(X_scaled.mean(axis=0), 6))

print("\nStandardized feature standard deviations:")
print(np.round(X_scaled.std(axis=0), 2))


# %% [markdown]
# ## 3. Apply PCA
#
# Reduce the four original features to two principal components.
#
# PCA is unsupervised: the target labels are not used to learn
# the principal components.

# %%
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_pca.shape}")

print("\nFirst five transformed samples:")
print(np.round(X_pca[:5], 3))


# %% [markdown]
# ## 4. Analyze Explained Variance
#
# Explained variance ratio indicates the proportion of total
# variance captured by each principal component.

# %%
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

for index, variance in enumerate(explained_variance, start=1):
    print(f"PC{index}: {variance:.4f} ({variance * 100:.2f}%)")

print(
    f"\nTotal variance explained by two components: "
    f"{cumulative_variance[-1]:.4f} "
    f"({cumulative_variance[-1] * 100:.2f}%)"
)


# %% [markdown]
# ## 5. Visualize Explained Variance
#
# The first component captures the most variance.
# The second component captures the next largest amount,
# subject to being orthogonal to the first.

# %%
component_names = [
    f"PC{index}"
    for index in range(1, len(explained_variance) + 1)
]

plt.figure(figsize=(8, 5))

plt.bar(
    component_names,
    explained_variance,
    label="Individual explained variance",
)

plt.plot(
    component_names,
    cumulative_variance,
    marker="o",
    linestyle="--",
    label="Cumulative explained variance",
)

plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6. Visualize the Two Principal Components
#
# Each sample is represented by two coordinates instead of
# its four original measurements.
#
# Target labels are used only for coloring the visualization.
# They do not influence PCA's learned components.

# %%
plt.figure(figsize=(9, 6))

for class_index, class_name in enumerate(target_names):
    mask = y == class_index

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=class_name,
        alpha=0.8,
        edgecolors="black",
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Iris Dataset After PCA")
plt.legend(title="Species")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Inspect Principal Component Loadings
#
# Loadings describe how the original standardized features
# contribute to each principal component.
#
# The sign of a component can be reversed without changing
# its mathematical meaning.

# %%
loadings = pca.components_.T

print("Principal component loadings:")
print(
    f"{'Feature':<25}"
    f"{'PC1':>10}"
    f"{'PC2':>10}"
)

print("-" * 45)

for feature, loading in zip(feature_names, loadings):
    print(
        f"{feature:<25}"
        f"{loading[0]:>10.4f}"
        f"{loading[1]:>10.4f}"
    )


# %% [markdown]
# ## 8. Key Takeaways
#
# - PCA transforms the original features into orthogonal components.
# - PCA does not use target labels to learn the components.
# - Feature scaling is important when original features have
#   different scales.
# - The first principal component captures the maximum possible
#   variance along a single direction.
# - Additional components capture remaining variance subject
#   to orthogonality.
# - Explained variance helps assess information retained after
#   dimensionality reduction.
# - PCA can reduce computational complexity but may discard
#   information relevant to a prediction task.
# - Component loadings help interpret how original features
#   contribute to the transformed representation.
