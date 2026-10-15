# %% [markdown]
# # Grid Search
#
# Find a strong hyperparameter configuration using GridSearchCV.
#
# Workflow:
# 1. Load and split the dataset.
# 2. Establish a baseline.
# 3. Define the hyperparameter grid.
# 4. Run grid search with cross-validation.
# 5. Inspect and visualize the results.
# 6. Evaluate the selected model on the test set.
#
# The test set is not used during hyperparameter selection.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
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
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# ## 1. Load and split the dataset
#
# The breast cancer dataset is a binary classification problem.
# Stratification preserves approximately the same class proportions
# in the training and test sets.

# %%
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

feature_names = dataset.feature_names
target_names = dataset.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("Dataset:", dataset.DESCR.splitlines()[0])
print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])
print("Classes:", list(target_names))


# %% [markdown]
# ## 2. Establish a baseline
#
# A DummyClassifier provides a simple reference point.
# It ignores the input features and predicts according to a
# simple strategy.

# %%
baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print("Baseline test accuracy:", f"{baseline_accuracy:.4f}")


# %% [markdown]
# ## 3. Define the pipeline and parameter grid
#
# StandardScaler is fitted inside the pipeline, so during
# cross-validation it learns scaling statistics only from the
# training portion of each fold.
#
# Logistic regression's C parameter controls regularization:
# smaller C means stronger regularization, while larger C
# means weaker regularization.

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
    "classifier__solver": ["lbfgs"],
    "classifier__class_weight": [None, "balanced"],
}

print("Number of candidate configurations:", 10)


# %% [markdown]
# ## 4. Run GridSearchCV
#
# GridSearchCV evaluates every parameter combination using
# stratified cross-validation.
#
# ROC AUC is the selection metric. It measures how well the
# model ranks positive examples above negative examples.

# %%
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:")
for parameter, value in grid_search.best_params_.items():
    print(f"  {parameter}: {value}")

print(
    "\nBest mean cross-validation ROC AUC:",
    f"{grid_search.best_score_:.4f}",
)


# %% [markdown]
# ## 5. Inspect the search results
#
# Each row represents one candidate configuration.
# The standard deviation describes variation across CV folds.

# %%
results = pd.DataFrame(grid_search.cv_results_)

results = results[
    [
        "param_classifier__C",
        "param_classifier__class_weight",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "rank_test_score",
    ]
].sort_values("rank_test_score")

results = results.rename(
    columns={
        "param_classifier__C": "C",
        "param_classifier__class_weight": "class_weight",
        "mean_test_score": "mean_cv_roc_auc",
        "std_test_score": "std_cv_roc_auc",
        "mean_train_score": "mean_train_roc_auc",
        "rank_test_score": "rank",
    }
)

print("\nTop grid-search configurations:")
print(results.head(10).to_string(index=False))


# %% [markdown]
# ## 6. Visualize the search
#
# Plot mean cross-validation ROC AUC for each C value.
# Separate lines show the two class-weighting strategies.

# %%
plot_results = pd.DataFrame(grid_search.cv_results_)

plt.figure(figsize=(9, 5))

for class_weight in [None, "balanced"]:
    subset = plot_results[
        plot_results["param_classifier__class_weight"].map(
            lambda value: (
                value is None
                if class_weight is None
                else value == class_weight
            )
        )
    ].sort_values("param_classifier__C")

    c_values = subset["param_classifier__C"].astype(float)
    mean_scores = subset["mean_test_score"]
    std_scores = subset["std_test_score"]

    label = (
        "No class weighting"
        if class_weight is None
        else "Balanced class weights"
    )

    plt.plot(
        c_values,
        mean_scores,
        marker="o",
        label=label,
    )

    plt.fill_between(
        c_values,
        mean_scores - std_scores,
        mean_scores + std_scores,
        alpha=0.15,
    )

plt.xscale("log")
plt.xlabel("Regularization parameter C (log scale)")
plt.ylabel("Mean cross-validation ROC AUC")
plt.title("Grid Search: Logistic Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Evaluate the selected model on the test set
#
# With refit=True, GridSearchCV has already refitted the best
# configuration on the complete training set.
#
# The test set is evaluated only after the search is complete.

# %%
best_model = grid_search.best_estimator_

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

plt.title("Grid Search: Test-Set Confusion Matrix")
plt.tight_layout()
plt.show()
