# %% [markdown]
# # Ensemble Evaluation and Hyperparameter Tuning
#
# We will:
# 1. Compare multiple ensemble classifiers.
# 2. Evaluate models using stratified cross-validation.
# 3. Tune a Random Forest using GridSearchCV.
# 4. Compare default and tuned models on a held-out test set.
# 5. Visualize cross-validation and test performance.
#
# Dataset: Breast Cancer Wisconsin Classification

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# %% [markdown]
# ## 1. Load and Split the Dataset
#
# Keep the test set separate during model selection.
# Use cross-validation on the training set for fairer model comparison.

# %%
RANDOM_STATE = 42

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")


# %% [markdown]
# ## 2. Define the Models
#
# Logistic Regression and SVM benefit from feature scaling.
# Tree-based ensemble models do not require feature scaling.

# %%
models = {
    "logistic_regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    ),
    "svm": make_pipeline(
        StandardScaler(),
        SVC(random_state=RANDOM_STATE),
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=100,
        random_state=RANDOM_STATE,
    ),
    "voting": VotingClassifier(
        estimators=[
            (
                "logistic_regression",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "hist_gradient_boosting",
                HistGradientBoostingClassifier(
                    max_iter=100,
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        voting="soft",
    ),
}


# %% [markdown]
# ## 3. Evaluate Models Using Cross-Validation
#
# StratifiedKFold preserves the approximate class distribution
# in each fold.
#
# All models are evaluated on the same folds.
# Accuracy and F1-score are calculated for each validation fold.

# %%
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
}

cv_results = {}

for name, model in models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    cv_results[name] = {
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "f1_mean": scores["test_f1"].mean(),
        "f1_std": scores["test_f1"].std(),
    }

    print(f"\n{name}")
    print(
        f"Accuracy: "
        f"{cv_results[name]['accuracy_mean']:.4f} "
        f"+/- {cv_results[name]['accuracy_std']:.4f}"
    )
    print(
        f"F1-score: "
        f"{cv_results[name]['f1_mean']:.4f} "
        f"+/- {cv_results[name]['f1_std']:.4f}"
    )


# %% [markdown]
# ## 4. Tune Random Forest with GridSearchCV
#
# GridSearchCV evaluates each parameter combination using
# cross-validation.
#
# The held-out test set is not used during tuning.

# %%
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=RANDOM_STATE,
    ),
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Random Forest Parameters")
print(grid_search.best_params_)

print(f"\nBest Cross-Validation F1-score: {grid_search.best_score_:.4f}")


# %% [markdown]
# ## 5. Compare Default and Tuned Random Forest
#
# Evaluate both models on the same held-out test set.
# The test set is used only after hyperparameter selection.

# %%
default_model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
)

default_model.fit(X_train, y_train)

default_pred = default_model.predict(X_test)
tuned_pred = best_model.predict(X_test)

default_accuracy = accuracy_score(y_test, default_pred)
default_f1 = f1_score(y_test, default_pred)

tuned_accuracy = accuracy_score(y_test, tuned_pred)
tuned_f1 = f1_score(y_test, tuned_pred)

print("\nTest Set Comparison")
print("-" * 55)
print(f"{'Model':<25} {'Accuracy':>12} {'F1-score':>12}")
print("-" * 55)
print(
    f"{'Default Random Forest':<25} "
    f"{default_accuracy:>12.4f} "
    f"{default_f1:>12.4f}"
)
print(
    f"{'Tuned Random Forest':<25} "
    f"{tuned_accuracy:>12.4f} "
    f"{tuned_f1:>12.4f}"
)


# %% [markdown]
# ## 6. Visualize Cross-Validation Performance
#
# Error bars represent one standard deviation across folds.
# A higher score is better for both metrics.

# %%
model_names = list(cv_results.keys())

accuracy_means = [
    cv_results[name]["accuracy_mean"]
    for name in model_names
]

accuracy_stds = [
    cv_results[name]["accuracy_std"]
    for name in model_names
]

f1_means = [
    cv_results[name]["f1_mean"]
    for name in model_names
]

f1_stds = [
    cv_results[name]["f1_std"]
    for name in model_names
]

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(
    x - width / 2,
    accuracy_means,
    width,
    yerr=accuracy_stds,
    capsize=4,
    label="Accuracy",
)

plt.bar(
    x + width / 2,
    f1_means,
    width,
    yerr=f1_stds,
    capsize=4,
    label="F1-score",
)

plt.xticks(
    x,
    model_names,
    rotation=25,
    ha="right",
)

plt.ylabel("Cross-Validation Score")
plt.title("Ensemble and Baseline Model Comparison")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Visualize Default vs Tuned Random Forest

# %%
model_labels = [
    "Default Random Forest",
    "Tuned Random Forest",
]

accuracy_values = [
    default_accuracy,
    tuned_accuracy,
]

f1_values = [
    default_f1,
    tuned_f1,
]

x = np.arange(len(model_labels))

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    accuracy_values,
    width,
    label="Accuracy",
)

plt.bar(
    x + width / 2,
    f1_values,
    width,
    label="F1-score",
)

plt.xticks(x, model_labels, rotation=10, ha="right")
plt.ylabel("Test Score")
plt.title("Default vs Tuned Random Forest")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 8. Key Takeaways
#
# - Cross-validation provides a more reliable comparison than one split.
# - StratifiedKFold preserves approximate class proportions.
# - GridSearchCV searches a predefined hyperparameter space.
# - Hyperparameter tuning must not use the held-out test set.
# - Accuracy and F1-score measure different aspects of performance.
# - Mean and standard deviation help describe cross-validation results.
# - Tuning does not guarantee better test performance.
# - Model selection should consider generalization, complexity,
#   inference latency, and operational requirements.
