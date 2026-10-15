# %% [markdown]
# # PCA: Explained Variance and Component Selection
#
# PCA can reduce dimensionality by retaining only a subset of
# principal components.
#
# We will:
# 1. Load and standardize the Iris dataset.
# 2. Calculate explained variance for all components.
# 3. Analyze cumulative explained variance.
# 4. Determine the number of components needed for different
#    variance thresholds.
# 5. Visualize individual and cumulative explained variance.
# 6. Compare dimensionality reduction at different thresholds.
#
# Dataset: Iris

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# ## 1. Load and Standardize the Dataset

# %%
iris = load_iris()

X = iris.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Number of samples: {X.shape[0]}")
print(f"Original number of features: {X.shape[1]}")


# %% [markdown]
# ## 2. Fit PCA with All Components
#
# Fitting PCA without limiting n_components allows us to inspect
# the variance captured by every principal component.
#
# The Iris dataset has four original features, so PCA produces
# four principal components.

# %%
pca = PCA()

X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nExplained variance:")
for index, variance in enumerate(explained_variance, start=1):
    print(f"PC{index}: {variance:.4f} ({variance * 100:.2f}%)")

print("\nCumulative explained variance:")
for index, variance in enumerate(cumulative_variance, start=1):
    print(f"First {index} component(s): {variance:.4f} ({variance * 100:.2f}%)")


# %% [markdown]
# ## 3. Determine the Number of Components for Each Threshold
#
# A variance threshold specifies how much total variance we want
# the selected components to retain.
#
# We select the smallest number of components whose cumulative
# explained variance reaches the threshold.

# %%
variance_thresholds = [0.80, 0.90, 0.95, 0.99]

print("\nComponent Selection")
print("-" * 45)
print(f"{'Threshold':<15}{'Components':>15}{'Actual Variance':>15}")
print("-" * 45)

for threshold in variance_thresholds:
    n_components = np.argmax(cumulative_variance >= threshold) + 1

    actual_variance = cumulative_variance[n_components - 1]

    print(
        f"{threshold * 100:>6.0f}%{'':<9}"
        f"{n_components:>10}"
        f"{actual_variance * 100:>14.2f}%"
    )


# %% [markdown]
# ## 4. Visualize Individual Explained Variance
#
# This plot shows how much variance each individual component
# explains.

# %%
component_numbers = np.arange(1, len(explained_variance) + 1)

plt.figure(figsize=(8, 5))

plt.bar(
    component_numbers,
    explained_variance,
    tick_label=[f"PC{i}" for i in component_numbers],
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Individual Explained Variance")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 5. Visualize Cumulative Explained Variance
#
# The cumulative curve helps identify how many components are
# required to retain a chosen proportion of variance.

# %%
plt.figure(figsize=(8, 5))

plt.plot(
    component_numbers,
    cumulative_variance,
    marker="o",
)

for threshold in variance_thresholds:
    plt.axhline(
        y=threshold,
        linestyle="--",
        label=f"{threshold:.0%} threshold",
    )

plt.xticks(component_numbers)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6. Compare Dimensionality at Different Thresholds
#
# PCA can automatically select the minimum number of components
# required to retain a specified proportion of variance.
#
# This example uses 95% explained variance.

# %%
pca_95 = PCA(n_components=0.95)

X_pca_95 = pca_95.fit_transform(X_scaled)

print("\nPCA with 95% Explained Variance")
print(f"Original shape: {X_scaled.shape}")
print(f"Reduced shape: {X_pca_95.shape}")
print(
    f"Number of components: {pca_95.n_components_}"
)
print(
    f"Actual explained variance: "
    f"{pca_95.explained_variance_ratio_.sum():.4f}"
)


# %% [markdown]
# ## 7. Compare Original and Reduced Representations
#
# The reduction ratio describes how many feature dimensions
# were removed relative to the original number of features.
#
# It does not directly measure information loss.

# %%
original_features = X_scaled.shape[1]
reduced_features = X_pca_95.shape[1]

reduction_percentage = (
    (original_features - reduced_features)
    / original_features
    * 100
)

print("\nDimensionality Reduction Summary")
print(f"Original features: {original_features}")
print(f"Reduced features: {reduced_features}")
print(f"Dimensions removed: {original_features - reduced_features}")
print(f"Dimensionality reduction: {reduction_percentage:.2f}%")
print(
    f"Variance retained: "
    f"{pca_95.explained_variance_ratio_.sum() * 100:.2f}%"
)


# %% [markdown]
# ## 8. Key Takeaways
#
# - Explained variance quantifies the variance captured by each
#   principal component.
# - Cumulative explained variance measures the total variance
#   captured by the first k components.
# - A variance threshold can help select the number of components.
# - PCA(n_components=0.95) retains enough components to explain
#   at least 95% of the variance when using the full-SVD solver.
# - High explained variance does not guarantee better predictive
#   performance.
# - The appropriate threshold depends on the task, data, and
#   computational requirements.
