# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


# %% 2. Load the dataset
data = fetch_california_housing()

X = data.data
y = data.target

feature_names = data.feature_names
target_name = "Median house value (in units of $100,000)"

print("Dataset shape:", X.shape)
print("Features:", feature_names)
print("Target:", target_name)
print("Target minimum:", round(y.min(), 3))
print("Target maximum:", round(y.max(), 3))
print("Target mean:", round(y.mean(), 3))


# %% 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))


# %% 4. Define the models
linear_model = make_pipeline(
    StandardScaler(),
    LinearRegression(),
)

tree_model = DecisionTreeRegressor(
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
)

models = {
    "Linear Regression": linear_model,
    "Decision Tree": tree_model,
}


# %% 5. Train and evaluate the models
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "predictions": y_pred,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }

    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")


# %% 6. Compare the metrics
metric_names = ["MAE", "MSE", "RMSE", "R2"]

print("\nModel comparison")
print("-" * 66)
print(
    f"{'Model':<22}"
    f"{'MAE':>10}"
    f"{'MSE':>10}"
    f"{'RMSE':>10}"
    f"{'R²':>10}"
)

for name, result in results.items():
    print(
        f"{name:<22}"
        f"{result['MAE']:>10.4f}"
        f"{result['MSE']:>10.4f}"
        f"{result['RMSE']:>10.4f}"
        f"{result['R2']:>10.4f}"
    )


# %% 7. Plot actual versus predicted values
fig, axes = plt.subplots(
    1,
    len(results),
    figsize=(14, 6),
)

for ax, (name, result) in zip(axes, results.items()):
    y_pred = result["predictions"]

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.25,
    )

    # Reference line for perfect predictions.
    lower = min(y_test.min(), y_pred.min())
    upper = max(y_test.max(), y_pred.max())

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Perfect prediction",
    )

    ax.set_title(name)
    ax.set_xlabel("Actual value")
    ax.set_ylabel("Predicted value")
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle("Actual vs Predicted House Values")
plt.tight_layout()
plt.show()


# %% 8. Analyze residuals
fig, axes = plt.subplots(
    1,
    len(results),
    figsize=(14, 5),
)

for ax, (name, result) in zip(axes, results.items()):
    residuals = y_test - result["predictions"]

    ax.scatter(
        result["predictions"],
        residuals,
        alpha=0.25,
    )

    ax.axhline(
        y=0,
        linestyle="--",
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.grid(alpha=0.3)

plt.suptitle("Residual Analysis")
plt.tight_layout()
plt.show()


# %% 9. Understand the effect of outliers
actual = np.array([10.0, 20.0, 30.0, 40.0])
good_predictions = np.array([11.0, 19.0, 31.0, 39.0])
outlier_predictions = np.array([11.0, 19.0, 31.0, 60.0])

print("\nOutlier experiment")
print("-" * 30)

for name, predictions in [
    ("Small errors", good_predictions),
    ("One large error", outlier_predictions),
]:
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)

    print(f"\n{name}")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")


# %% 10. Compare with a simple baseline
# Predict the mean target value from the training set for every test sample.
baseline_predictions = np.full(
    shape=len(y_test),
    fill_value=y_train.mean(),
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions,
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions,
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions,
)

print("\nMean baseline")
print("-" * 30)
print(f"MAE:  {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"R²:   {baseline_r2:.4f}")
