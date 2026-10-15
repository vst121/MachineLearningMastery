# %% [markdown]
# # Randomized Search
#
# Optimize a machine learning model using RandomizedSearchCV.
#
# Workflow:
# 1. Load and split the dataset.
# 2. Establish a baseline.
# 3. Define hyperparameter distributions.
# 4. Run randomized search with cross-validation.
# 5. Inspect and visualize the sampled configurations.
# 6. Evaluate the selected model on the held-out test set.
#
# All preprocessing is inside the pipeline to prevent data leakage.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import loguniform, randint

from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# ## 1. Load and split the dataset
#
# Use stratified splitting to preserve class proportions.
# Keep the test set separate from hyperparameter selection.

# %%
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

target_names = dataset.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])


# %% [markdown]
# ## 2. Establish a baseline
#
# The DummyClassifier ignores the input features and predicts
# the most frequent class.

# %%
baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print("Baseline test accuracy:", f"{baseline_accuracy:.4f}")


# %% [markdown]
# ## 3. Define the pipeline and search distributions
#
# RandomForestClassifier does not require feature scaling, so
# this pipeline deliberately omits StandardScaler.
#
# loguniform samples values across orders of magnitude.
# randint samples integers from the lower bound (inclusive)
# to the upper bound (exclusive).

# %%
pipeline = Pipeline(
    steps=[
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)

param_distributions = {
    "classifier__n_estimators": randint(100, 601),
    "classifier__max_depth": [None, 5, 10, 15, 20, 30],
    "classifier__min_samples_split": randint(2, 21),
    "classifier__min_samples_leaf": randint(1, 11),
    "classifier__max_features": ["sqrt", "log2", 0.5, 0.8],
    "classifier__class_weight": [None, "balanced", "balanced_subsample"],
}

n_iter = 40

print("Search iterations:", n_iter)
print("Cross-validation folds:", 5)
print("Total candidate-fold fits:", n_iter * 5)


# %% [markdown]
# ## 4. Run RandomizedSearchCV
#
# RandomizedSearchCV samples n_iter configurations from the
# supplied distributions and evaluates each using cross-validation.
#
# ROC AUC is used for model selection.

# %%
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=n_iter,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    refit=True,
    return_train_score=True,
)

random_search.fit(X_train, y_train)

print("\nBest parameters:")
for parameter, value in random_search.best_params_.items():
    print(f"  {parameter}: {value}")

print(
    "\nBest mean cross-validation ROC AUC:",
    f"{random_search.best_score_:.4f}",
)


# %% [markdown]
# ## 5. Inspect sampled configurations
#
# Review the top configurations and compare training and
# validation scores. A large gap can indicate overfitting.

# %%
results = pd.DataFrame(random_search.cv_results_)

columns_to_show = [
    "param_classifier__n_estimators",
    "param_classifier__max_depth",
    "param_classifier__min_samples_split",
    "param_classifier__min_samples_leaf",
    "mean_train_score",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
]

results = results[columns_to_show].sort_values("rank_test_score")

results = results.rename(
    columns={
        "param_classifier__n_estimators": "n_estimators",
        "param_classifier__max_depth": "max_depth",
        "param_classifier__min_samples_split": "min_samples_split",
        "param_classifier__min_samples_leaf": "min_samples_leaf",
        "mean_train_score": "mean_train_roc_auc",
        "mean_test_score": "mean_cv_roc_auc",
        "std_test_score": "std_cv_roc_auc",
        "rank_test_score": "rank",
    }
)

print("\nTop sampled configurations:")
print(results.head(10).to_string(index=False))


# %% [markdown]
# ## 6. Visualize the search results
#
# Plot the mean cross-validation ROC AUC of each sampled
# configuration, ordered by performance.

# %%
all_results = pd.DataFrame(random_search.cv_results_)

ordered_results = all_results.sort_values("mean_test_score")

mean_scores = ordered_results["mean_test_score"].to_numpy()
std_scores = ordered_results["std_test_score"].to_numpy()

iterations = np.arange(1, len(mean_scores) + 1)

plt.figure(figsize=(10, 5))

plt.errorbar(
    iterations,
    mean_scores,
    yerr=std_scores,
    fmt="o",
    markersize=4,
    capsize=2,
)

plt.axhline(
    random_search.best_score_,
    linestyle="--",
    label=f"Best CV ROC AUC: {random_search.best_score_:.4f}",
)

plt.xlabel("Configurations (sorted by CV performance)")
plt.ylabel("Mean cross-validation ROC AUC")
plt.title("Randomized Search Results")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Evaluate the selected model on the test set
#
# With refit=True, RandomizedSearchCV refits the best configuration
# on the complete training set.
#
# Evaluate the held-out test set only after the search is complete.

# %%
best_model = random_search.best_estimator_

test_predictions = best_model.predict(X_test)
test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, test_predictions)
test_roc_auc = roc_auc_score(y_test, test_probabilities)

print("\nFinal test results")
print("Baseline accuracy:", f"{baseline_accuracy:.4f}")
print("Tuned model accuracy:", f"{test_accuracy:.4f}")
print("Tuned model ROC AUC:", f"{test_roc_auc:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=target_names,
        zero_division=0,
    )
)

ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    y_test,
    display_labels=target_names,
)

plt.title("Randomized Search: Test-Set Confusion Matrix")
plt.tight_layout()
plt.show()
