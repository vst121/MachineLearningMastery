# %% 1. Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% 2. Load the Dataset
housing = fetch_california_housing()

X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="MedHouseVal")

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])


# %% 3. Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))


# %% 4. Define Models
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Ridge alpha=1.0": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),
    "Lasso alpha=0.01": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.01, max_iter=20000)),
    ]),
}

results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
    })

    fitted_models[name] = model


# %% 5. Compare Predictive Performance
results_df = pd.DataFrame(results).sort_values("RMSE")

print("\nTest Set Performance:")
print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


# %% 6. Compare Coefficients
coefficient_data = {"Feature": X.columns}

for name, model in fitted_models.items():
    estimator = model.named_steps["model"]
    coefficient_data[name] = estimator.coef_

coefficients_df = pd.DataFrame(coefficient_data)

print("\nModel Coefficients:")
print(coefficients_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


# %% 7. Compare Coefficient Norms and Feature Selection
summary = []

for name, model in fitted_models.items():
    estimator = model.named_steps["model"]
    coefficients = estimator.coef_

    summary.append({
        "Model": name,
        "Coefficient L1 Norm": np.sum(np.abs(coefficients)),
        "Coefficient L2 Norm": np.linalg.norm(coefficients),
        "Nonzero Coefficients": np.count_nonzero(coefficients),
        "Zero Coefficients": np.count_nonzero(coefficients == 0),
    })

summary_df = pd.DataFrame(summary)

print("\nCoefficient Summary:")
print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


# %% 8. Visualize Coefficients
coefficients_df.set_index("Feature").plot(
    kind="bar",
    figsize=(13, 6),
)

plt.title("Linear Regression, Ridge, and Lasso Coefficients")
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Model")
plt.tight_layout()
plt.show()


# %% 9. Compare Prediction Errors
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(results_df["Model"], results_df["RMSE"])
axes[0].set_title("Test RMSE Comparison")
axes[0].set_ylabel("RMSE")
axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(results_df["Model"], results_df["R2"])
axes[1].set_title("Test R² Comparison")
axes[1].set_ylabel("R²")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.show()


# %% 10. Compare L1 Coefficient Penalty
fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    summary_df["Model"],
    summary_df["Coefficient L1 Norm"],
)

ax.set_title("Sum of Absolute Coefficient Values")
ax.set_ylabel("Coefficient L1 Norm")
ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.show()


# %% 11. Inspect Lasso Feature Selection
lasso_model = fitted_models["Lasso alpha=0.01"].named_steps["model"]

selected_features = X.columns[lasso_model.coef_ != 0].tolist()
excluded_features = X.columns[lasso_model.coef_ == 0].tolist()

print("\nLasso Feature Selection (alpha=0.01)")
print("Selected features:", selected_features)
print("Excluded features:", excluded_features)
print("Selected feature count:", len(selected_features))
print("Excluded feature count:", len(excluded_features))
