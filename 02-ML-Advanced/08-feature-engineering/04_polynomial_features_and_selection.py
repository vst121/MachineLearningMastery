# %% [markdown]
# # Polynomial Features and Feature Selection
#
# Feature engineering can improve a model by creating useful representations
# of the original inputs. Feature selection can then remove irrelevant or
# redundant features.
#
# This example covers:
# - Polynomial features and interaction terms
# - VarianceThreshold
# - SelectKBest with statistical tests
# - Model-based feature selection
# - Recursive Feature Elimination (RFE)
#
# We use a synthetic regression dataset with known informative features
# so that the techniques can be explored without downloading data.

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    f_regression,
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# ## 1. Create a dataset with informative and irrelevant features
#
# We generate a regression dataset with five informative features and
# additional noise features.
#
# The target depends on the informative features, while the remaining
# features contain no intentional predictive signal.

# %%
X, y = make_regression(
    n_samples=500,
    n_features=10,
    n_informative=5,
    noise=15.0,
    random_state=42,
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print(f"Dataset shape: {X.shape}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Number of input features: {X.shape[1]}")

# %% [markdown]
# ## 2. Create polynomial features and interactions
#
# PolynomialFeatures creates powers of features and combinations of features.
#
# With degree=2 and include_bias=False, it generates:
# - Original features
# - Squared features
# - Pairwise interaction terms
#
# The number of features can grow quickly as the number of input features
# or the polynomial degree increases.

# %%
polynomial_transformer = PolynomialFeatures(
    degree=2,
    include_bias=False,
)

X_train_poly = polynomial_transformer.fit_transform(X_train)
X_test_poly = polynomial_transformer.transform(X_test)

polynomial_feature_names = polynomial_transformer.get_feature_names_out(
    feature_names
)

print("\nPolynomial feature generation:")
print(f"Original feature count: {X_train.shape[1]}")
print(f"Polynomial feature count: {X_train_poly.shape[1]}")

print("\nExample generated features:")
for name in polynomial_feature_names[:20]:
    print(f"  {name}")

# %% [markdown]
# ## 3. Compare linear and polynomial regression
#
# A linear regression model can learn linear relationships.
# Polynomial features allow the same estimator to model nonlinear patterns.
#
# We evaluate both models on the same held-out test set.
# This dataset is generated from a linear relationship, so polynomial
# features are not guaranteed to improve performance.

# %%
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

polynomial_model = Pipeline(
    steps=[
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ]
)
polynomial_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)
y_pred_polynomial = polynomial_model.predict(X_test)

linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))
polynomial_rmse = np.sqrt(mean_squared_error(y_test, y_pred_polynomial))

linear_r2 = r2_score(y_test, y_pred_linear)
polynomial_r2 = r2_score(y_test, y_pred_polynomial)

print("\nRegression model comparison:")
print(f"Linear Regression RMSE: {linear_rmse:.3f}")
print(f"Polynomial Regression RMSE: {polynomial_rmse:.3f}")
print(f"Linear Regression R²: {linear_r2:.3f}")
print(f"Polynomial Regression R²: {polynomial_r2:.3f}")

# %%
model_names = ["Linear", "Polynomial"]
rmse_values = [linear_rmse, polynomial_rmse]

plt.figure(figsize=(8, 5))
plt.bar(model_names, rmse_values)
plt.ylabel("Test RMSE (lower is better)")
plt.title("Linear vs. Polynomial Regression")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Remove low-variance features
#
# VarianceThreshold removes features whose variance is below a chosen threshold.
# It does not use the target values.
#
# This technique is most useful when some features are constant or nearly
# constant. It does not determine whether a feature predicts the target.

# %%
variance_selector = VarianceThreshold(threshold=0.01)

X_train_variance = variance_selector.fit_transform(X_train)
X_test_variance = variance_selector.transform(X_test)

variance_support = variance_selector.get_support()

print("\nVarianceThreshold:")
print(f"Features before selection: {X_train.shape[1]}")
print(f"Features after selection: {X_train_variance.shape[1]}")
print(
    "Selected features:",
    [name for name, keep in zip(feature_names, variance_support) if keep],
)

# %% [markdown]
# ## 5. Select features using a statistical test
#
# SelectKBest with f_regression scores each feature based on its linear
# relationship with the continuous target.
#
# This method is univariate: it evaluates each feature individually.
# It may miss features that are useful mainly through interactions.

# %%
k_best_selector = SelectKBest(
    score_func=f_regression,
    k=5,
)

X_train_kbest = k_best_selector.fit_transform(X_train, y_train)
X_test_kbest = k_best_selector.transform(X_test)

kbest_support = k_best_selector.get_support()
kbest_scores = k_best_selector.scores_

print("\nSelectKBest:")
print(f"Features before selection: {X_train.shape[1]}")
print(f"Features after selection: {X_train_kbest.shape[1]}")
print(
    "Selected features:",
    [name for name, keep in zip(feature_names, kbest_support) if keep],
)

print("\nFeature scores:")
for name, score in zip(feature_names, kbest_scores):
    print(f"{name}: {score:.3f}")

# %% [markdown]
# ## 6. Select features using a model
#
# RandomForestRegressor provides feature importance estimates.
# SelectFromModel can use those estimates to select features.
#
# Feature importance is model-dependent and should not automatically
# be interpreted as causal importance.

# %%
from sklearn.feature_selection import SelectFromModel

forest = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
)
forest.fit(X_train, y_train)

model_selector = SelectFromModel(
    estimator=forest,
    threshold="median",
)

model_selector.fit(X_train, y_train)

X_train_model_selected = model_selector.transform(X_train)
X_test_model_selected = model_selector.transform(X_test)

model_support = model_selector.get_support()

print("\nModel-based selection:")
print(f"Features before selection: {X_train.shape[1]}")
print(f"Features after selection: {X_train_model_selected.shape[1]}")
print(
    "Selected features:",
    [name for name, keep in zip(feature_names, model_support) if keep],
)

# %%
importances = forest.feature_importances_
sorted_indices = np.argsort(importances)

plt.figure(figsize=(9, 6))
plt.barh(
    np.array(feature_names)[sorted_indices],
    importances[sorted_indices],
)
plt.xlabel("Random Forest Feature Importance")
plt.title("Model-Based Feature Importance")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Recursive Feature Elimination (RFE)
#
# RFE repeatedly fits an estimator and removes the least important features.
# The number of retained features is specified in advance.
#
# Here we use LinearRegression, which exposes coefficients for feature ranking.

# %%
rfe_selector = RFE(
    estimator=LinearRegression(),
    n_features_to_select=5,
)

rfe_selector.fit(X_train, y_train)

X_train_rfe = rfe_selector.transform(X_train)
X_test_rfe = rfe_selector.transform(X_test)

rfe_support = rfe_selector.get_support()

print("\nRecursive Feature Elimination:")
print(f"Features before selection: {X_train.shape[1]}")
print(f"Features after selection: {X_train_rfe.shape[1]}")
print(
    "Selected features:",
    [name for name, keep in zip(feature_names, rfe_support) if keep],
)

# %% [markdown]
# ## 8. Evaluate selected features
#
# Compare the same regression estimator using all features and selected features.
# The selection process is fitted only on the training data.

# %%
selection_results = {}

selection_methods = {
    "All features": None,
    "SelectKBest": SelectKBest(score_func=f_regression, k=5),
    "RFE": RFE(
        estimator=LinearRegression(),
        n_features_to_select=5,
    ),
}

for method_name, selector in selection_methods.items():
    if selector is None:
        X_train_selected = X_train
        X_test_selected = X_test
    else:
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

    estimator = LinearRegression()
    estimator.fit(X_train_selected, y_train)

    predictions = estimator.predict(X_test_selected)

    selection_results[method_name] = {
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
        "features": X_train_selected.shape[1],
    }

print("\nFeature selection evaluation:")
for method_name, result in selection_results.items():
    print(
        f"{method_name:<15} "
        f"features={result['features']}, "
        f"RMSE={result['rmse']:.3f}, "
        f"R²={result['r2']:.3f}"
    )

# %% [markdown]
# ## Key takeaways
#
# - PolynomialFeatures creates powers and interactions that can model
#   nonlinear relationships.
# - More features do not automatically improve generalization.
# - VarianceThreshold removes constant or low-variance features without
#   considering the target.
# - SelectKBest evaluates features individually using a statistical score.
# - Model-based selection uses importance or coefficient information.
# - RFE recursively removes features based on an estimator.
# - Fit feature selectors on training data only.
# - For reliable comparisons, perform selection inside cross-validation
#   pipelines rather than selecting once before cross-validation.
