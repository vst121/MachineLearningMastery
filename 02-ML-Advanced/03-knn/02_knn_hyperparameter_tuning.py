# %% [markdown]
# # KNN Hyperparameter Tuning
#
# Objectives:
# - Understand hyperparameters in KNN.
# - Use GridSearchCV for systematic model selection.
# - Apply stratified cross-validation.
# - Compare configurations using macro F1-score.
# - Evaluate the selected model on the test set.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# ## 1. Load and Split the Dataset
#
# The test set is held out for final evaluation.
# Hyperparameter selection uses only the training set.

# %%
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


# %% [markdown]
# ## 2. Define the Pipeline
#
# Scaling is fitted separately inside each cross-validation fold.
# This prevents information from validation folds leaking into scaling.

# %%
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ]
)


# %% [markdown]
# ## 3. Define the Hyperparameter Search Space
#
# We will investigate:
# - n_neighbors: number of neighbors (k)
# - weights: uniform or distance-weighted voting
# - p: distance power for the Minkowski metric
#
# p=1 corresponds to Manhattan distance.
# p=2 corresponds to Euclidean distance.

# %%
param_grid = {
    "knn__n_neighbors": list(range(1, 21)),
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# %% [markdown]
# ## 4. Run Grid Search
#
# Macro F1 computes the F1-score for each class and averages them
# equally, making it useful for evaluating multiclass performance.

# %%
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

print(f"\nBest mean CV macro F1: {grid_search.best_score_:.4f}")


# %% [markdown]
# ## 5. Inspect the Best Configurations

# %%
results = pd.DataFrame(grid_search.cv_results_)

results = results[
    [
        "param_knn__n_neighbors",
        "param_knn__weights",
        "param_knn__p",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "rank_test_score",
    ]
].sort_values(
    by=["rank_test_score", "std_test_score"]
)

print(results.head(10).to_string(index=False))


# %% [markdown]
# ## 6. Visualize Cross-Validation Results
#
# For a clear comparison, examine the CV score by k separately
# for each voting strategy and distance metric.

# %%
cv_results = pd.DataFrame(grid_search.cv_results_)

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(14, 5),
    sharey=True,
)

for ax, p_value, distance_name in [
    (axes[0], 1, "Manhattan Distance"),
    (axes[1], 2, "Euclidean Distance"),
]:
    subset = cv_results[
        cv_results["param_knn__p"] == p_value
    ]

    for weight in ["uniform", "distance"]:
        weight_subset = subset[
            subset["param_knn__weights"] == weight
        ].sort_values("param_knn__n_neighbors")

        ax.plot(
            weight_subset["param_knn__n_neighbors"].astype(int),
            weight_subset["mean_test_score"],
            marker="o",
            label=f"{weight} voting",
        )

    ax.set_title(distance_name)
    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_xticks(range(1, 21))
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[0].set_ylabel("Mean Cross-Validation Macro F1")

fig.suptitle("KNN Hyperparameter Search")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Evaluate the Selected Model on the Test Set
#
# The best estimator has already been refitted on the complete
# training set by GridSearchCV. The test set is used only here
# for final evaluation.

# %%
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        zero_division=0,
    )
)

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names,
).plot(cmap="Blues")

plt.title("Tuned KNN: Test Confusion Matrix")
plt.tight_layout()
plt.show()
