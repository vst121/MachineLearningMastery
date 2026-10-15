# %% [markdown]
# # KNN Regression
#
# Objectives:
# - Understand how KNN predicts continuous values.
# - Train a KNeighborsRegressor.
# - Understand why feature scaling matters for regression.
# - Evaluate predictions using regression metrics.
# - Select k using cross-validation.
# - Compare KNN with a simple mean-prediction baseline.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    GridSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# ## 1. Load the Dataset
#
# The Diabetes dataset contains numerical features and a continuous
# target representing a measure of disease progression.
#
# The target is a regression value, not a classification label.

# %%
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

print("Dataset shape:", X.shape)
print("Features:", diabetes.feature_names)
print("Target range:", y.min(), "to", y.max())


# %% [markdown]
# ## 2. Split the Dataset
#
# The test set is held out for final evaluation.
# Hyperparameter selection uses only the training set.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


# %% [markdown]
# ## 3. Create a KNN Regression Pipeline
#
# KNeighborsRegressor predicts a value by aggregating the target
# values of its nearest neighbors.
#
# StandardScaler is included in the pipeline to keep preprocessing
# fitted only on the training data.

# %%
k = 5

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=k,
                weights="uniform",
                metric="minkowski",
                p=2,
            ),
        ),
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("First 10 actual values:")
print(y_test[:10])

print("\nFirst 10 predictions:")
print(np.round(y_pred[:10], 2))


# %% [markdown]
# ## 4. Evaluate Regression Performance
#
# MAE: average absolute prediction error.
# MSE: average squared prediction error.
# RMSE: square root of MSE, expressed in target units.
# R²: performance relative to predicting the test target mean.

# %%
def evaluate_regression(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predictions)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


metrics = evaluate_regression(y_test, y_pred)

print("\nTest metrics:")
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")


# %% [markdown]
# ## 5. Compare Against a Mean Baseline
#
# DummyRegressor predicts the mean target value observed in the
# training set for every test observation.
#
# A useful model should be evaluated against a simple baseline.

# %%
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

comparison = pd.DataFrame(
    [
        {
            "Model": "Mean baseline",
            **evaluate_regression(y_test, baseline_pred),
        },
        {
            "Model": f"KNN (k={k})",
            **evaluate_regression(y_test, y_pred),
        },
    ]
)

print(comparison.round(4).to_string(index=False))


# %% [markdown]
# ## 6. Visualize Actual vs Predicted Values
#
# Points close to the diagonal represent accurate predictions.

# %%
plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7,
    edgecolors="black",
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect predictions",
)

plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title(f"KNN Regression: Actual vs Predicted (k={k})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Examine Residuals
#
# A residual is the difference between the actual and predicted value.
#
# Residual = actual value - predicted value.
#
# Patterns in residuals can indicate that a model misses structure
# in the data.

# %%
residuals = y_test - y_pred

plt.figure(figsize=(9, 5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.7,
    edgecolors="black",
)

plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Target")
plt.ylabel("Residual")
plt.title("KNN Regression Residuals")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Mean residual:", round(residuals.mean(), 4))


# %% [markdown]
# ## 8. Select k Using Cross-Validation
#
# GridSearchCV evaluates different values of k using only the
# training data.
#
# Negative MAE is used because scikit-learn maximizes scoring
# values, while lower MAE is better.

# %%
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor()),
    ]
)

param_grid = {
    "knn__n_neighbors": list(range(1, 31)),
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

print(
    "Best mean CV MAE:",
    f"{-grid_search.best_score_:.4f}",
)


# %% [markdown]
# ## 9. Inspect Cross-Validation Results

# %%
cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results["Mean CV MAE"] = -cv_results["mean_test_score"]
cv_results["Mean Training MAE"] = -cv_results["mean_train_score"]

top_results = cv_results[
    [
        "param_knn__n_neighbors",
        "param_knn__weights",
        "param_knn__p",
        "Mean CV MAE",
        "Mean Training MAE",
        "rank_test_score",
    ]
].sort_values("rank_test_score")

print(top_results.head(10).to_string(index=False))


# %% [markdown]
# ## 10. Evaluate the Tuned Model on the Test Set
#
# The test set is used for final evaluation after model selection.

# %%
best_model = grid_search.best_estimator_

tuned_predictions = best_model.predict(X_test)

tuned_metrics = evaluate_regression(y_test, tuned_predictions)

print("\nTuned KNN test metrics:")
for metric_name, value in tuned_metrics.items():
    print(f"{metric_name}: {value:.4f}")


# %%
final_comparison = pd.DataFrame(
    [
        {
            "Model": "Mean baseline",
            **evaluate_regression(y_test, baseline_pred),
        },
        {
            "Model": f"Initial KNN (k={k})",
            **evaluate_regression(y_test, y_pred),
        },
        {
            "Model": "Tuned KNN",
            **tuned_metrics,
        },
    ]
)

print("\nFinal comparison:")
print(final_comparison.round(4).to_string(index=False))
