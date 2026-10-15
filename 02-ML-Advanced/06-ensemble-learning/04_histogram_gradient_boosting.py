# %% [markdown]
# # Histogram-Based Gradient Boosting
#
# Histogram-based gradient boosting discretizes continuous features into
# bins, reducing the computational cost of finding optimal split points.
#
# We will:
# 1. Train a HistGradientBoostingClassifier for classification.
# 2. Train a HistGradientBoostingRegressor for regression.
# 3. Evaluate both models.
# 4. Compare histogram-based boosting with traditional gradient boosting.
# 5. Visualize feature importance alternatives and regression predictions.
#
# Dataset:
# - Breast Cancer Wisconsin: classification
# - Diabetes: regression

# %%
import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# %% [markdown]
# ## 1. Histogram-Based Gradient Boosting for Classification

# %%
cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

hist_classifier = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42,
)

start_time = time.perf_counter()
hist_classifier.fit(X_train, y_train)
hist_training_time = time.perf_counter() - start_time

y_pred = hist_classifier.predict(X_test)

print("Histogram-Based Gradient Boosting Classifier")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Training time: {hist_training_time:.4f} seconds")


# %% [markdown]
# ## 2. Compare Traditional and Histogram-Based Boosting
#
# Both models use gradient boosting, but histogram-based boosting
# groups continuous feature values into bins before finding splits.

# %%
gradient_classifier = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)

start_time = time.perf_counter()
gradient_classifier.fit(X_train, y_train)
gradient_training_time = time.perf_counter() - start_time

gradient_pred = gradient_classifier.predict(X_test)

gradient_accuracy = accuracy_score(y_test, gradient_pred)
hist_accuracy = accuracy_score(y_test, y_pred)

print("\nClassification Comparison")
print("-" * 45)
print(f"{'Model':<35} {'Accuracy':>10}")
print("-" * 45)
print(f"{'Gradient Boosting':<35} {gradient_accuracy:>10.4f}")
print(f"{'Histogram Gradient Boosting':<35} {hist_accuracy:>10.4f}")

print("\nTraining Time Comparison")
print("-" * 45)
print(f"Gradient Boosting:          {gradient_training_time:.4f} seconds")
print(f"Histogram Gradient Boosting: {hist_training_time:.4f} seconds")


# %% [markdown]
# ## 3. Feature Importance with Permutation Importance
#
# Histogram-based gradient boosting does not expose the same built-in
# feature_importances_ attribute as traditional gradient boosting.
#
# Permutation importance measures how much model performance decreases
# when a feature's values are shuffled.

# %%
importance_result = permutation_importance(
    hist_classifier,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

importance_indices = np.argsort(importance_result.importances_mean)[-10:]

plt.figure(figsize=(10, 6))

plt.barh(
    cancer.feature_names[importance_indices],
    importance_result.importances_mean[importance_indices],
    xerr=importance_result.importances_std[importance_indices],
)

plt.xlabel("Mean Decrease in Accuracy")
plt.ylabel("Feature")
plt.title("Top 10 Features: Histogram Gradient Boosting")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4. Histogram-Based Gradient Boosting for Regression

# %%
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

hist_regressor = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=15,
    random_state=42,
)

start_time = time.perf_counter()
hist_regressor.fit(X_train, y_train)
hist_regression_time = time.perf_counter() - start_time

y_pred = hist_regressor.predict(X_test)

hist_mse = mean_squared_error(y_test, y_pred)
hist_rmse = np.sqrt(hist_mse)
hist_r2 = r2_score(y_test, y_pred)

print("\nHistogram-Based Gradient Boosting Regressor")
print(f"RMSE: {hist_rmse:.4f}")
print(f"R²: {hist_r2:.4f}")
print(f"Training time: {hist_regression_time:.4f} seconds")


# %% [markdown]
# ## 5. Compare Regression Models

# %%
gradient_regressor = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)

start_time = time.perf_counter()
gradient_regressor.fit(X_train, y_train)
gradient_regression_time = time.perf_counter() - start_time

gradient_pred = gradient_regressor.predict(X_test)

gradient_rmse = np.sqrt(mean_squared_error(y_test, gradient_pred))
gradient_r2 = r2_score(y_test, gradient_pred)

print("\nRegression Comparison")
print("-" * 55)
print(f"{'Model':<35} {'RMSE':>10} {'R²':>10}")
print("-" * 55)
print(f"{'Gradient Boosting':<35} {gradient_rmse:>10.4f} {gradient_r2:>10.4f}")
print(f"{'Histogram Gradient Boosting':<35} {hist_rmse:>10.4f} {hist_r2:>10.4f}")

print("\nTraining Time Comparison")
print("-" * 45)
print(f"Gradient Boosting:          {gradient_regression_time:.4f} seconds")
print(f"Histogram Gradient Boosting: {hist_regression_time:.4f} seconds")


# %% [markdown]
# ## 6. Visualize Regression Predictions

# %%
plt.figure(figsize=(8, 6))

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
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Histogram Gradient Boosting: Actual vs Predicted")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Key Takeaways
#
# - Histogram-based boosting bins continuous feature values.
# - Binning reduces the number of split candidates considered during training.
# - HistGradientBoostingClassifier supports classification.
# - HistGradientBoostingRegressor supports regression.
# - Histogram-based boosting can be particularly useful for large datasets.
# - It is not guaranteed to be faster on every dataset.
# - Training time and predictive performance should both be evaluated.
# - Permutation importance provides a model-agnostic feature importance method.
