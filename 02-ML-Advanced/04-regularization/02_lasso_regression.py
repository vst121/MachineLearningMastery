# %% 1. Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression
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


# %% 4. Define and Evaluate Models
models = {
    "Mean Baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Lasso alpha=0.001": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.001, max_iter=20000)),
    ]),
    "Lasso alpha=0.01": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.01, max_iter=20000)),
    ]),
    "Lasso alpha=0.1": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1, max_iter=20000)),
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
linear_model = fitted_models["Linear Regression"].named_steps["model"]

coefficient_data = {
    "Feature": X.columns,
    "Linear Regression": linear_model.coef_,
}

for alpha in [0.001, 0.01, 0.1]:
    model_name = f"Lasso alpha={alpha}"
    lasso_model = fitted_models[model_name].named_steps["model"]
    coefficient_data[model_name] = lasso_model.coef_

coefficients_df = pd.DataFrame(coefficient_data)

print("\nModel Coefficients:")
print(coefficients_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


# %% 7. Identify Selected and Excluded Features
print("\nFeature Selection:")
for alpha in [0.001, 0.01, 0.1]:
    model_name = f"Lasso alpha={alpha}"
    lasso_model = fitted_models[model_name].named_steps["model"]

    selected_features = X.columns[lasso_model.coef_ != 0].tolist()
    excluded_features = X.columns[lasso_model.coef_ == 0].tolist()

    print(f"\n{model_name}")
    print("Selected features:", selected_features)
    print("Excluded features:", excluded_features)
    print(f"Number of selected features: {len(selected_features)}")
    print(f"Number of excluded features: {len(excluded_features)}")


# %% 8. Visualize Coefficient Shrinkage
coefficients_df.set_index("Feature").plot(
    kind="bar",
    figsize=(13, 6),
)

plt.title("Lasso Regularization: Coefficient Shrinkage")
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Model")
plt.tight_layout()
plt.show()


# %% 9. Investigate Regularization Strength
alphas = np.logspace(-4, 0, 30)

alpha_results = []

for alpha in alphas:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=alpha, max_iter=20000)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    lasso_model = model.named_steps["lasso"]

    alpha_results.append({
        "Alpha": alpha,
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
        "Nonzero Coefficients": np.count_nonzero(lasso_model.coef_),
    })

alpha_df = pd.DataFrame(alpha_results)

print("\nRegularization Strength Experiment:")
print(
    alpha_df.iloc[[0, 5, 10, 15, 20, 25, 29]].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 10. Plot Performance and Feature Selection
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].semilogx(alpha_df["Alpha"], alpha_df["RMSE"], marker="o")
axes[0].set_title("Test RMSE vs. Alpha")
axes[0].set_xlabel("Alpha (regularization strength)")
axes[0].set_ylabel("RMSE")
axes[0].grid(True)

axes[1].semilogx(
    alpha_df["Alpha"],
    alpha_df["Nonzero Coefficients"],
    marker="o",
)
axes[1].set_title("Number of Nonzero Coefficients vs. Alpha")
axes[1].set_xlabel("Alpha (regularization strength)")
axes[1].set_ylabel("Number of nonzero coefficients")
axes[1].set_yticks(range(len(X.columns) + 1))
axes[1].grid(True)

plt.tight_layout()
plt.show()


# %% 11. Evaluate Against the Baseline
baseline_rmse = results_df.loc[
    results_df["Model"] == "Mean Baseline", "RMSE"
].iloc[0]

best_lasso_row = results_df[
    results_df["Model"].str.startswith("Lasso")
].sort_values("RMSE").iloc[0]

print("\nBaseline RMSE:", round(baseline_rmse, 4))
print("Best tested Lasso configuration:", best_lasso_row["Model"])
print("Best tested Lasso RMSE:", round(best_lasso_row["RMSE"], 4))
print("Best tested Lasso R2:", round(best_lasso_row["R2"], 4))
