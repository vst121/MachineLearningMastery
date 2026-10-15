# %% [markdown]
# # PCA: Classification Before and After Dimensionality Reduction
#
# PCA reduces the number of features before a classifier learns
# the relationship between features and target labels.
#
# We will:
# 1. Load and split the Iris dataset.
# 2. Build a baseline classification pipeline.
# 3. Build PCA-based classification pipelines.
# 4. Compare cross-validation performance.
# 5. Evaluate the models on a held-out test set.
# 6. Visualize classification performance.
#
# Dataset: Iris

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# ## 1. Load and Split the Dataset
#
# The test set remains separate until final evaluation.
# Stratification preserves the approximate class distribution.

# %%
RANDOM_STATE = 42

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Original features: {X_train.shape[1]}")
print(f"Classes: {iris.target_names}")


# %% [markdown]
# ## 2. Define the Baseline Model
#
# StandardScaler standardizes the original features.
# Logistic Regression learns directly from all four features.
#
# Using a Pipeline ensures scaling is fitted only on the training
# portion of each cross-validation fold.

# %%
baseline_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# %% [markdown]
# ## 3. Define PCA-Based Models
#
# Each PCA pipeline scales the features before applying PCA.
# PCA is unsupervised: it does not use the target labels to learn
# the principal components.
#
# We compare two and three components with the four-feature baseline.

# %%
pca_2_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

pca_3_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=3)),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

models = {
    "Original features (4)": baseline_model,
    "PCA (2 components)": pca_2_model,
    "PCA (3 components)": pca_3_model,
}


# %% [markdown]
# ## 4. Compare Models Using Cross-Validation
#
# Cross-validation evaluates each pipeline on multiple training
# and validation splits.
#
# Scaling and PCA are refitted independently inside each fold,
# avoiding data leakage.

# %%
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

cv_results = {}

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    cv_results[name] = {
        "mean": scores.mean(),
        "std": scores.std(),
        "scores": scores,
    }

    print(f"\n{name}")
    print(f"Fold accuracies: {np.round(scores, 4)}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")


# %% [markdown]
# ## 5. Fit Models and Evaluate the Test Set
#
# After cross-validation, fit each model on the complete training
# set and evaluate it once on the held-out test set.

# %%
test_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "predictions": y_pred,
    }

    print(f"\n{name}")
    print(f"Test accuracy: {test_results[name]['accuracy']:.4f}")


# %% [markdown]
# ## 6. Inspect Explained Variance
#
# PCA explains a proportion of feature variance, not necessarily
# the proportion of information relevant to classification.
#
# The fitted PCA models reveal how much variance their components
# retain from the training data.

# %%
for name in ["PCA (2 components)", "PCA (3 components)"]:
    fitted_pca = models[name].named_steps["pca"]

    explained_variance = fitted_pca.explained_variance_ratio_
    total_variance = explained_variance.sum()

    print(f"\n{name}")
    print(
        "Individual explained variance:",
        np.round(explained_variance, 4),
    )
    print(f"Total variance retained: {total_variance:.4f}")
    print(f"Variance retained: {total_variance * 100:.2f}%")


# %% [markdown]
# ## 7. Visualize Cross-Validation Performance
#
# Error bars represent one standard deviation across folds.

# %%
model_names = list(models.keys())

mean_scores = [
    cv_results[name]["mean"]
    for name in model_names
]

std_scores = [
    cv_results[name]["std"]
    for name in model_names
]

plt.figure(figsize=(10, 6))

plt.bar(
    model_names,
    mean_scores,
    yerr=std_scores,
    capsize=5,
)

plt.ylabel("Cross-Validation Accuracy")
plt.title("Classification Performance: Original Features vs PCA")
plt.ylim(0, 1.05)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 8. Compare Test Accuracy

# %%
test_accuracies = [
    test_results[name]["accuracy"]
    for name in model_names
]

plt.figure(figsize=(10, 6))

plt.bar(
    model_names,
    test_accuracies,
)

plt.ylabel("Test Accuracy")
plt.title("Held-Out Test Performance")
plt.ylim(0, 1.05)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 9. Inspect the Baseline Classification Report
#
# Accuracy summarizes overall correctness.
# Precision, recall, and F1-score provide additional class-level
# information.

# %%
baseline_predictions = test_results[
    "Original features (4)"
]["predictions"]

print(
    classification_report(
        y_test,
        baseline_predictions,
        target_names=iris.target_names,
        zero_division=0,
    )
)


# %% [markdown]
# ## 10. Visualize the Baseline Confusion Matrix

# %%
ConfusionMatrixDisplay.from_predictions(
    y_test,
    baseline_predictions,
    display_labels=iris.target_names,
    cmap="Blues",
)

plt.title("Baseline Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 11. Key Takeaways
#
# - PCA reduces feature dimensionality before classification.
# - Pipelines help prevent data leakage during cross-validation.
# - PCA does not use target labels when learning components.
# - Explained variance does not directly measure predictive value.
# - Dimensionality reduction may improve, preserve, or reduce
#   classification performance.
# - Fewer features can reduce computational cost, although the
#   improvement may be small for a dataset as small as Iris.
# - Use held-out test data for final evaluation.
# - Evaluate whether dimensionality reduction helps the actual
#   prediction task before adopting it in production.
