# %% [markdown]
# # Feature Engineering Evaluation
#
# Compare a baseline classifier with polynomial features and
# feature selection.
#
# Workflow:
# 1. Split data into training and test sets.
# 2. Compare candidate pipelines using stratified cross-validation.
# 3. Select the best candidate using mean cross-validation accuracy.
# 4. Evaluate the selected model on the held-out test set.
# 5. Inspect the engineered and selected features.
#
# All learned preprocessing steps are inside sklearn pipelines
# to prevent data leakage.

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# %% [markdown]
# ## 1. Load and split the dataset
#
# The test set is held out until model selection is complete.

# %%
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Original features:", X_train.shape[1])


# %% [markdown]
# ## 2. Define candidate pipelines
#
# Baseline:
# Standardize the original features and train logistic regression.
#
# Polynomial:
# Add degree-two polynomial and interaction features.
#
# Polynomial + selection:
# Generate polynomial features and retain the 10 features with
# the highest ANOVA F-scores.
#
# Scaling, feature generation, and feature selection are fitted
# inside each training fold during cross-validation.

# %%
def create_classifier():
    """Create a logistic regression classifier."""
    return LogisticRegression(
        max_iter=2000,
        random_state=42,
    )


baseline_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", create_classifier()),
    ]
)

polynomial_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "polynomial",
            PolynomialFeatures(
                degree=2,
                include_bias=False,
            ),
        ),
        ("polynomial_scaler", StandardScaler()),
        ("classifier", create_classifier()),
    ]
)

selected_polynomial_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "polynomial",
            PolynomialFeatures(
                degree=2,
                include_bias=False,
            ),
        ),
        ("polynomial_scaler", StandardScaler()),
        (
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=10,
            ),
        ),
        ("classifier", create_classifier()),
    ]
)

models = {
    "Baseline": baseline_pipeline,
    "Polynomial features": polynomial_pipeline,
    "Polynomial + selection": selected_polynomial_pipeline,
}


# %% [markdown]
# ## 3. Compare models with cross-validation
#
# All candidates use the same stratified folds.
# Each pipeline independently learns its transformations from
# the training portion of each fold.

# %%
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_results = {}

for model_name, model in models.items():
    scores = cross_val_score(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    cv_results[model_name] = scores

    print(f"\n{model_name}")
    print("Fold accuracies:", np.round(scores, 4))
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")


# %% [markdown]
# ## 4. Visualize cross-validation results
#
# Error bars show one standard deviation across the folds.
# They illustrate variation between folds, not a confidence interval.

# %%
model_names = list(cv_results.keys())

mean_scores = np.array([
    cv_results[name].mean()
    for name in model_names
])

std_scores = np.array([
    cv_results[name].std()
    for name in model_names
])

plt.figure(figsize=(9, 5))

plt.bar(
    model_names,
    mean_scores,
    yerr=std_scores,
    capsize=5,
)

plt.ylabel("Cross-validation accuracy")
plt.title("Comparing Feature Engineering Strategies")
plt.ylim(0, 1.05)
plt.xticks(rotation=10)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 5. Select the best model
#
# Select the candidate with the highest mean cross-validation
# accuracy. Do not use test-set results to make this decision.

# %%
best_model_name = max(
    cv_results,
    key=lambda name: cv_results[name].mean(),
)

best_model = models[best_model_name]

print("Selected model:", best_model_name)
print(
    "Mean cross-validation accuracy:",
    f"{cv_results[best_model_name].mean():.4f}",
)


# %% [markdown]
# ## 6. Evaluate on the held-out test set
#
# Refit the selected pipeline on the complete training set.
# Evaluate it once on the test set.

# %%
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

print("\nTest accuracy:", f"{test_accuracy:.4f}")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=target_names,
)

plt.title(f"Test Results: {best_model_name}")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Inspect engineered features
#
# PolynomialFeatures creates squared terms and pairwise interactions.
# If feature selection is part of the chosen pipeline, inspect which
# generated features were retained.

# %%
if "polynomial" in best_model.named_steps:
    polynomial_transformer = best_model.named_steps["polynomial"]

    engineered_feature_names = (
        polynomial_transformer.get_feature_names_out(feature_names)
    )

    print(
        "\nFeatures after polynomial expansion:",
        len(engineered_feature_names),
    )

    if "feature_selection" in best_model.named_steps:
        selector = best_model.named_steps["feature_selection"]

        selected_feature_names = engineered_feature_names[
            selector.get_support()
        ]

        print("Features retained:", len(selected_feature_names))
        print("\nSelected features:")

        for name in selected_feature_names:
            print(f" - {name}")
    else:
        print("No feature-selection step was used by the selected model.")
else:
    print("\nThe selected model uses the original features.")
