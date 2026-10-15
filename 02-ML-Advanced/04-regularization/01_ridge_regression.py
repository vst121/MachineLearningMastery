# %% 1. Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% 2. Load the Dataset
housing = fetch_california_housing()

X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="MedHouseVal")

print("Dataset shape:", X.shape)
print("\nFeatures:")
print(X.head())

print("\nTarget statistics:")
print(y.describe())


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
    "Mean Baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Ridge alpha=0.1": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=0.1)),
    ]),
    "Ridge alpha=1.0": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),
    "Ridge alpha=10.0": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
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


# %% 6. Compare Coefficient Magnitudes
feature_names = X.columns

linear_model = fitted_models["Linear Regression"].named_steps["model"]
linear_coefficients = linear_model.coef_

coefficient_data = {
    "Feature": feature_names,
    "Linear Regression": linear_coefficients,
}

for alpha in [0.1, 1.0, 10.0]:
    model_name = f"Ridge alpha={alpha}"
    ridge_model = fitted_models[model_name].named_steps["model"]
    coefficient_data[model_name] = ridge_model.coef_

coefficients_df = pd.DataFrame(coefficient_data)

print("\nModel Coefficients:")
print(coefficients_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

coefficient_columns = [
    "Linear Regression",
    "Ridge alpha=0.1",
    "Ridge alpha=1.0",
    "Ridge alpha=10.0",
]

coefficient_norms = {
    column: np.linalg.norm(coefficients_df[column].to_numpy())
    for column in coefficient_columns
}

print("\nCoefficient L2 Norms:")
for model_name, norm in coefficient_norms.items():
    print(f"{model_name}: {norm:.4f}")


# %% 7. Visualize Coefficient Shrinkage
coefficients_df.set_index("Feature")[coefficient_columns].plot(
    kind="bar",
    figsize=(12, 6),
)

plt.title("Ridge Regularization: Coefficient Shrinkage")
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Model")
plt.tight_layout()
plt.show()


# %% 8. Investigate Regularization Strength
alphas = np.logspace(-3, 3, 30)

alpha_results = []

for alpha in alphas:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    ridge_model = model.named_steps["ridge"]

    alpha_results.append({
        "Alpha": alpha,
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
        "Coefficient L2 Norm": np.linalg.norm(ridge_model.coef_),
    })

alpha_df = pd.DataFrame(alpha_results)

print("\nRegularization Strength Experiment:")
print(
    alpha_df.iloc[[0, 5, 10, 15, 20, 25, 29]].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 9. Plot Performance and Coefficient Norm
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].semilogx(alpha_df["Alpha"], alpha_df["RMSE"], marker="o")
axes[0].set_title("Test RMSE vs. Alpha")
axes[0].set_xlabel("Alpha (regularization strength)")
axes[0].set_ylabel("RMSE")
axes[0].grid(True)

axes[1].semilogx(
    alpha_df["Alpha"],
    alpha_df["Coefficient L2 Norm"],
    marker="o",
)
axes[1].set_title("Coefficient Magnitude vs. Alpha")
axes[1].set_xlabel("Alpha (regularization strength)")
axes[1].set_ylabel("Coefficient L2 Norm")
axes[1].grid(True)

plt.tight_layout()
plt.show()


# %% 10. Evaluate Against the Baseline
baseline_rmse = results_df.loc[
    results_df["Model"] == "Mean Baseline", "RMSE"
].iloc[0]

best_ridge_row = results_df[
    results_df["Model"].str.startswith("Ridge")
].sort_values("RMSE").iloc[0]

print("\nBaseline RMSE:", round(baseline_rmse, 4))
print("Best tested Ridge configuration:", best_ridge_row["Model"])
print("Best tested Ridge RMSE:", round(best_ridge_row["RMSE"], 4))
print("Best tested Ridge R2:", round(best_ridge_row["R2"], 4))
