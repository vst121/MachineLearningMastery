# %% [markdown]
# # Nested Cross-Validation
#
# Estimate model generalization while accounting for hyperparameter
# selection.
#
# Workflow:
# 1. Load the dataset.
# 2. Define a model pipeline and parameter grid.
# 3. Use inner cross-validation to select hyperparameters.
# 4. Use outer cross-validation to evaluate the complete selection
#    process on held-out folds.
# 5. Inspect outer-fold scores and selected hyperparameters.
# 6. Run a final search on the complete development dataset.
#
# The outer test fold is never used to select hyperparameters.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# ## 1. Load and split the dataset
#
# Reserve a final test set for an optional final evaluation.
# Nested cross-validation operates only on the development split.

# %%
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target
target_names = dataset.target_names

X_development, X_test, y_development, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("Development samples:", X_development.shape[0])
print("Final test samples:", X_test.shape[0])
print("Number of features:", X_development.shape[1])


# %% [markdown]
# ## 2. Define the pipeline and hyperparameter grid
#
# Scaling is inside the pipeline. Each training fold learns its own
# scaling statistics, preventing information leakage.

# %%
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=42,
            ),
        ),
    ]
)

param_grid = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "classifier__class_weight": [None, "balanced"],
}

print("Number of candidate configurations:", 10)


# %% [markdown]
# ## 3. Define inner and outer cross-validation
#
# Inner CV:
# Selects the best hyperparameters using grid search.
#
# Outer CV:
# Evaluates the selected configuration on held-out folds.
#
# The outer folds must not participate in the inner hyperparameter
# selection for their corresponding evaluation.

# %%
inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123,
)

inner_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=inner_cv,
    n_jobs=1,
    refit=True,
    return_train_score=False,
)


# %% [markdown]
# ## 4. Run nested cross-validation
#
# cross_validate clones and fits the complete GridSearchCV estimator
# separately for every outer fold.
#
# Each outer-fold score evaluates a model whose hyperparameters were
# selected using only that outer fold's training partition.

# %%
nested_results = cross_validate(
    estimator=inner_search,
    X=X_development,
    y=y_development,
    cv=outer_cv,
    scoring="roc_auc",
    return_estimator=True,
    n_jobs=-1,
)

outer_scores = nested_results["test_score"]

print("Outer-fold ROC AUC scores:", np.round(outer_scores, 4))
print("Mean outer ROC AUC:", f"{outer_scores.mean():.4f}")
print("Standard deviation:", f"{outer_scores.std():.4f}")


# %% [markdown]
# ## 5. Inspect hyperparameters selected in each outer fold
#
# Different folds may select different configurations.
# This variation helps illustrate how model selection depends
# on the training data available in each fold.

# %%
fold_results = []

for fold_index, (score, fitted_search) in enumerate(
    zip(outer_scores, nested_results["estimator"]),
    start=1,
):
    fold_results.append(
        {
            "outer_fold": fold_index,
            "roc_auc": score,
            "best_C": fitted_search.best_params_["classifier__C"],
            "best_class_weight": fitted_search.best_params_[
                "classifier__class_weight"
            ],
            "inner_best_roc_auc": fitted_search.best_score_,
        }
    )

fold_results = pd.DataFrame(fold_results)

print("\nOuter-fold results:")
print(fold_results.to_string(index=False))


# %% [markdown]
# ## 6. Visualize outer-fold performance
#
# Each bar represents the ROC AUC obtained on one held-out outer fold.
# The dashed line shows the mean outer-fold score.

# %%
plt.figure(figsize=(8, 5))

fold_numbers = np.arange(1, len(outer_scores) + 1)

plt.bar(
    fold_numbers,
    outer_scores,
)

plt.axhline(
    outer_scores.mean(),
    linestyle="--",
    label=f"Mean ROC AUC: {outer_scores.mean():.4f}",
)

plt.xlabel("Outer fold")
plt.ylabel("ROC AUC")
plt.title("Nested Cross-Validation Results")
plt.xticks(fold_numbers)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Run a final hyperparameter search
#
# After estimating performance with nested CV, tune the model on
# the complete development dataset.
#
# This final search selects the configuration for deployment.
# Its best cross-validation score is not a replacement for the
# nested-CV performance estimate.

# %%
final_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=inner_cv,
    n_jobs=-1,
    refit=True,
    return_train_score=False,
)

final_search.fit(X_development, y_development)

print("\nFinal selected parameters:")
for parameter, value in final_search.best_params_.items():
    print(f"  {parameter}: {value}")

print(
    "\nFinal search mean CV ROC AUC:",
    f"{final_search.best_score_:.4f}",
)

print(
    "Nested CV mean ROC AUC:",
    f"{outer_scores.mean():.4f}",
)


# %% [markdown]
# ## 8. Evaluate on the final test set
#
# This optional final evaluation is performed only after model
# selection is complete. The final test set was excluded from both
# nested cross-validation and the final grid search.

# %%
final_model = final_search.best_estimator_

test_predictions = final_model.predict(X_test)
test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, test_predictions)
test_roc_auc = roc_auc_score(y_test, test_probabilities)

print("\nFinal test results")
print("Test accuracy:", f"{test_accuracy:.4f}")
print("Test ROC AUC:", f"{test_roc_auc:.4f}")

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
    final_model,
    X_test,
    y_test,
    display_labels=target_names,
)

plt.title("Nested CV Workflow: Final Test Confusion Matrix")
plt.tight_layout()
plt.show()
