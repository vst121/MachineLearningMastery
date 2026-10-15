# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree


# %% 2. Load the Dataset
housing = fetch_california_housing()

X = pd.DataFrame(
    housing.data,
    columns=housing.feature_names,
)

y = pd.Series(
    housing.target,
    name="MedianHouseValue",
)

print("Dataset shape:", X.shape)
print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget summary:")
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


# %% 4. Define Regression Models
models = {
    "Mean Baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Decision Tree (Depth 3)": DecisionTreeRegressor(
        max_depth=3,
        random_state=42,
    ),
    "Decision Tree (Depth 10)": DecisionTreeRegressor(
        max_depth=10,
        random_state=42,
    ),
}

# An unrestricted tree can grow much deeper.
models["Decision Tree (Unrestricted)"] = DecisionTreeRegressor(
    random_state=42,
)


# %% 5. Train and Evaluate Models
results = {}
trained_models = {}
predictions_by_model = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    trained_models[name] = model
    predictions_by_model[name] = predictions

    results[name] = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
        "Tree Depth": (
            model.get_depth()
            if isinstance(model, DecisionTreeRegressor)
            else np.nan
        ),
        "Leaves": (
            model.get_n_leaves()
            if isinstance(model, DecisionTreeRegressor)
            else np.nan
        ),
    }

results_df = pd.DataFrame(results).T

print("\nRegression Model Comparison:")
print(
    results_df.to_string(
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 6. Visualize Actual vs Predicted Values
selected_models = [
    "Linear Regression",
    "Decision Tree (Depth 3)",
    "Decision Tree (Depth 10)",
    "Decision Tree (Unrestricted)",
]

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.ravel()

for axis, name in zip(axes, selected_models):
    predictions = predictions_by_model[name]

    axis.scatter(
        y_test,
        predictions,
        alpha=0.25,
        s=12,
    )

    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())

    axis.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    axis.set_xlabel("Actual Median House Value")
    axis.set_ylabel("Predicted Median House Value")
    axis.set_title(
        f"{name}\n"
        f"R² = {r2_score(y_test, predictions):.3f}"
    )

plt.suptitle("Actual vs Predicted Housing Values")
plt.tight_layout()
plt.show()


# %% 7. Visualize the Effect of Tree Depth
tree_models = {
    name: model
    for name, model in trained_models.items()
    if isinstance(model, DecisionTreeRegressor)
}

depth_results = []

for name, model in tree_models.items():
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    depth_results.append({
        "Model": name,
        "Depth": model.get_depth(),
        "Leaves": model.get_n_leaves(),
        "Train RMSE": np.sqrt(
            mean_squared_error(y_train, train_predictions)
        ),
        "Test RMSE": np.sqrt(
            mean_squared_error(y_test, test_predictions)
        ),
    })

depth_df = pd.DataFrame(depth_results)

print("\nTree Complexity Analysis:")
print(
    depth_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

depth_df.set_index("Model")[["Train RMSE", "Test RMSE"]].plot(
    kind="bar",
    figsize=(11, 5),
)

plt.ylabel("RMSE")
plt.title("Training vs Test Error by Tree Complexity")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()


# %% 8. Inspect Feature Importances
tree_model = trained_models["Decision Tree (Depth 10)"]

feature_importances = pd.Series(
    tree_model.feature_importances_,
    index=X.columns,
).sort_values(ascending=True)

print("\nFeature Importances:")
print(feature_importances.sort_values(ascending=False))

feature_importances.plot(
    kind="barh",
    figsize=(9, 6),
)

plt.xlabel("Importance")
plt.title("Decision Tree Regression Feature Importances")
plt.tight_layout()
plt.show()


# %% 9. Visualize a Small Regression Tree
small_tree = trained_models["Decision Tree (Depth 3)"]

plt.figure(figsize=(22, 10))

plot_tree(
    small_tree,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    impurity=True,
    precision=3,
    fontsize=8,
)

plt.title("Decision Tree Regressor (Maximum Depth 3)")
plt.tight_layout()
plt.show()


# %% 10. Predict a New Example
new_house = X_test.iloc[[0]].copy()

predicted_value = small_tree.predict(new_house)[0]
actual_value = y_test.iloc[0]

print("\nExample Prediction:")
print("Actual value:", round(actual_value, 3))
print("Predicted value:", round(predicted_value, 3))
print("Input features:")
print(new_house.to_string(index=False))
