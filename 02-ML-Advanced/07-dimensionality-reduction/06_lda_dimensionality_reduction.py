# %% [markdown]
# # Linear Discriminant Analysis (LDA)
#
# Linear Discriminant Analysis is a supervised dimensionality reduction technique.
#
# PCA finds directions that maximize variance without considering class labels.
# LDA finds directions that aim to maximize separation between known classes.
#
# For C classes, LDA can produce at most C - 1 discriminant components.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## 1. Load the Iris dataset

# %%
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Number of classes: {len(np.unique(y))}")

# %% [markdown]
# ## 2. Apply LDA for dimensionality reduction
#
# Iris has three classes, so LDA can produce at most two components.
# The pipeline fits the scaler and LDA using only the training data.
#
# Scaling is included here to make preprocessing explicit. LDA itself does
# not require standardization in every application.

# %%
lda_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(n_components=2)),
    ]
)

X_train_lda = lda_pipeline.fit_transform(X_train, y_train)
X_test_lda = lda_pipeline.transform(X_test)

lda = lda_pipeline.named_steps["lda"]

print(f"Original number of features: {X.shape[1]}")
print(f"Reduced number of features: {X_train_lda.shape[1]}")
print(f"Training projection shape: {X_train_lda.shape}")
print(f"Test projection shape: {X_test_lda.shape}")
print(f"Explained discriminability ratios: {lda.explained_variance_ratio_}")
print(
    "Cumulative explained discriminability: "
    f"{lda.explained_variance_ratio_.sum():.2%}"
)

# %% [markdown]
# ## 3. Visualize the discriminant components
#
# Fit the visualization projection on the training set and transform the test set.
# The training and test projections are shown separately to demonstrate the split.
#
# Colors indicate the known species labels.

# %%
fig, ax = plt.subplots(figsize=(10, 7))

for class_index, class_name in enumerate(target_names):
    train_mask = y_train == class_index
    test_mask = y_test == class_index

    ax.scatter(
        X_train_lda[train_mask, 0],
        X_train_lda[train_mask, 1],
        label=f"{class_name} (train)",
        s=65,
        alpha=0.75,
    )

    ax.scatter(
        X_test_lda[test_mask, 0],
        X_test_lda[test_mask, 1],
        marker="x",
        s=90,
        linewidths=2,
        label=f"{class_name} (test)",
    )

ax.set_title("LDA: Iris Data in Two Discriminant Dimensions")
ax.set_xlabel("Linear Discriminant 1")
ax.set_ylabel("Linear Discriminant 2")
ax.legend(ncol=2)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Compare PCA and LDA
#
# PCA is unsupervised: it does not use class labels when finding components.
# LDA is supervised: it uses labels to find directions that separate classes.
#
# This comparison fits both transformations on training data only.

# %%
from sklearn.decomposition import PCA

pca_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
    ]
)

X_train_pca = pca_pipeline.fit_transform(X_train)
X_test_pca = pca_pipeline.transform(X_test)

pca = pca_pipeline.named_steps["pca"]

print("\nPCA explained variance ratios:")
print(pca.explained_variance_ratio_)
print(f"PCA cumulative explained variance: {pca.explained_variance_ratio_.sum():.2%}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for class_index, class_name in enumerate(target_names):
    mask = y_train == class_index

    axes[0].scatter(
        X_train_pca[mask, 0],
        X_train_pca[mask, 1],
        label=class_name,
        s=55,
        alpha=0.8,
    )

    axes[1].scatter(
        X_train_lda[mask, 0],
        X_train_lda[mask, 1],
        label=class_name,
        s=55,
        alpha=0.8,
    )

axes[0].set_title("PCA: Maximum Variance")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

axes[1].set_title("LDA: Class Separation")
axes[1].set_xlabel("Linear Discriminant 1")
axes[1].set_ylabel("Linear Discriminant 2")

for ax in axes:
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key takeaways
#
# - LDA is supervised and requires class labels during fitting.
# - PCA is unsupervised and maximizes total variance.
# - LDA seeks directions that maximize between-class separation relative
#   to within-class variation.
# - LDA produces at most min(number of features, number of classes - 1)
#   components in the standard classification setting.
# - Use Pipeline to fit transformations on training data only.
# - LDA is not the same as Logistic Regression: LDA is a discriminant
#   analysis method, while Logistic Regression is a classification model.
