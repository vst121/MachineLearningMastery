# %% [markdown]
# # GridSearchCV for SVM Classification
#
# GridSearchCV systematically evaluates combinations of
# hyperparameters using cross-validation.
#
# This example combines:
# - Pipeline and StandardScaler
# - Support Vector Classification (SVC)
# - StratifiedKFold
# - GridSearchCV
#
# The Iris dataset is used for multiclass classification.

# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# %% [markdown]
# ## 1. Load the dataset

# %%
iris = load_iris()
X = iris.data
y = iris.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class names: {iris.target_names}")

# %% [markdown]
# ## 2. Split the data
#
# Keep the test set separate from hyperparameter tuning.
# GridSearchCV will only use the training set.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# %% [markdown]
# ## 3. Build the pipeline
#
# Scaling is fitted separately within each cross-validation fold.
# This prevents information from validation folds leaking into
# the training process.

# %%
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC()),
    ]
)

# %% [markdown]
# ## 4. Define the hyperparameter grid
#
# The double underscore syntax addresses parameters of pipeline
# steps: svc__C and svc__kernel.
#
# gamma is relevant to the RBF and polynomial kernels.
# degree is relevant to the polynomial kernel.

# %%
param_grid = [
    {
        "svc__kernel": ["linear"],
        "svc__C": [0.1, 1, 10, 100],
    },
    {
        "svc__kernel": ["rbf"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.01, 0.1, 1],
    },
    {
        "svc__kernel": ["poly"],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", 0.1],
        "svc__degree": [2, 3, 4],
    },
]

# %% [markdown]
# ## 5. Configure stratified cross-validation
#
# StratifiedKFold preserves class proportions in each fold.
# Shuffling makes the fold assignment less dependent on the
# original ordering of the samples.

# %%
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# %% [markdown]
# ## 6. Run GridSearchCV
#
# Accuracy is used as the selection metric.
# refit=True retrains the best configuration on the full
# training set after cross-validation.

# %%
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)

print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
print("Best parameters:")
for parameter, value in grid_search.best_params_.items():
    print(f"  {parameter}: {value}")

# %% [markdown]
# ## 7. Inspect the best model

# %%
best_model = grid_search.best_estimator_

print("\nBest estimator:")
print(best_model)

# %% [markdown]
# ## 8. Evaluate on the held-out test set
#
# The test set was not used during hyperparameter selection.
# Use it only after the best configuration has been selected.

# %%
y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {test_accuracy:.4f}")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        zero_division=0,
    )
)

# %% [markdown]
# ## 9. Compare the top configurations
#
# Cross-validation scores help us understand which parameter
# combinations performed well on the training data.

# %%
results = grid_search.cv_results_
rankings = np.argsort(results["rank_test_score"])[:10]

print("\nTop 10 configurations")
print("-" * 75)

for rank in rankings:
    print(
        f"Rank {results['rank_test_score'][rank]:>2} | "
        f"CV accuracy: {results['mean_test_score'][rank]:.4f} "
        f"+/- {results['std_test_score'][rank]:.4f} | "
        f"Parameters: {results['params'][rank]}"
    )

# %% [markdown]
# ## 10. Visualize the top configurations

# %%
top_scores = results["mean_test_score"][rankings]
top_labels = [
    (
        f"{results['params'][i]['svc__kernel']}, "
        f"C={results['params'][i]['svc__C']}"
    )
    for i in rankings
]

plt.figure(figsize=(10, 6))
plt.barh(range(len(rankings)), top_scores)
plt.yticks(range(len(rankings)), top_labels)
plt.gca().invert_yaxis()
plt.xlabel("Mean Cross-Validation Accuracy")
plt.title("Top 10 SVM Hyperparameter Configurations")
plt.xlim(0, 1.05)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 11. Key Takeaways
#
# - GridSearchCV evaluates all specified parameter combinations.
# - StratifiedKFold preserves class proportions across folds.
# - Pipeline ensures scaling is fitted independently in each fold.
# - svc__C, svc__kernel, svc__gamma, and svc__degree tune the SVC.
# - best_score_ reports the best mean cross-validation score.
# - The held-out test set estimates performance after model selection.
# - A high score on one dataset does not guarantee generalization
#   to new datasets.
