# %% [markdown]
# # RandomizedSearchCV for SVM Classification
#
# RandomizedSearchCV samples a fixed number of hyperparameter
# combinations instead of evaluating every combination.
#
# This example combines:
# - Pipeline and StandardScaler
# - Support Vector Classification (SVC)
# - StratifiedKFold
# - RandomizedSearchCV
#
# The Iris dataset is used for multiclass classification.

# %%
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import loguniform

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
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
print(f"Class distribution: {np.bincount(y)}")
print(f"Class names: {iris.target_names}")


# %% [markdown]
# ## 2. Split the data
#
# The test set remains separate from hyperparameter tuning.

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
# Scaling is fitted independently within each cross-validation
# training fold, preventing preprocessing leakage.

# %%
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC()),
    ]
)


# %% [markdown]
# ## 4. Define the search space
#
# loguniform samples values across orders of magnitude.
# This is useful for positive parameters such as C and gamma.

# %%
param_distributions = [
    {
        "svc__kernel": ["linear"],
        "svc__C": loguniform(1e-3, 1e3),
    },
    {
        "svc__kernel": ["rbf"],
        "svc__C": loguniform(1e-3, 1e3),
        "svc__gamma": loguniform(1e-4, 1e2),
    },
    {
        "svc__kernel": ["poly"],
        "svc__C": loguniform(1e-3, 1e3),
        "svc__gamma": loguniform(1e-4, 1e1),
        "svc__degree": [2, 3, 4, 5],
    },
]


# %% [markdown]
# ## 5. Configure cross-validation
#
# StratifiedKFold maintains class proportions across folds.

# %%
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# %% [markdown]
# ## 6. Run RandomizedSearchCV
#
# n_iter controls the number of sampled configurations.
# Increasing it explores more of the search space but takes
# longer to run.

# %%
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=60,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    refit=True,
    random_state=42,
    return_train_score=True,
)

random_search.fit(X_train, y_train)

print(
    f"Best cross-validation accuracy: "
    f"{random_search.best_score_:.4f}"
)
print("Best parameters:")
for parameter, value in random_search.best_params_.items():
    print(f"  {parameter}: {value}")


# %% [markdown]
# ## 7. Evaluate on the held-out test set
#
# The test set is used only after hyperparameter selection.

# %%
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {test_accuracy:.4f}")
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
# ## 8. Inspect the top configurations

# %%
results = random_search.cv_results_
rankings = np.argsort(results["rank_test_score"])[:10]

print("\nTop sampled configurations")
print("-" * 75)

for index in rankings:
    print(
        f"Rank {results['rank_test_score'][index]:>2} | "
        f"CV accuracy: {results['mean_test_score'][index]:.4f} "
        f"+/- {results['std_test_score'][index]:.4f} | "
        f"Parameters: {results['params'][index]}"
    )


# %% [markdown]
# ## 9. Visualize the top configurations

# %%
top_scores = results["mean_test_score"][rankings]
top_labels = [
    (
        f"{results['params'][i]['svc__kernel']}, "
        f"C={results['params'][i]['svc__C']:.3g}"
    )
    for i in rankings
]

plt.figure(figsize=(10, 6))
plt.barh(range(len(rankings)), top_scores)
plt.yticks(range(len(rankings)), top_labels)
plt.gca().invert_yaxis()
plt.xlabel("Mean Cross-Validation Accuracy")
plt.title("Top 10 Randomly Sampled SVM Configurations")
plt.xlim(0, 1.05)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 10. Key Takeaways
#
# - RandomizedSearchCV samples a fixed number of configurations.
# - loguniform explores positive parameters across multiple scales.
# - n_iter controls the search budget.
# - Randomized search is useful for large hyperparameter spaces.
# - Grid search is useful when exhaustive evaluation of a small,
#   carefully selected grid is affordable.
# - The held-out test set must not be used to select parameters.
