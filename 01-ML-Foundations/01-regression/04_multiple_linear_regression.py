"""
Machine Learning Foundations
Topic: Multiple Linear Regression

Dataset:
Scikit-learn Diabetes dataset.

Learning objectives:
- Load and explore a built-in dataset.
- Use multiple input features for prediction.
- Understand the intercept and model coefficients.
- Split data into training and test sets.
- Evaluate and compare model performance.
- Visualize a regression plane using two features.
- Compare actual and predicted values.
- Analyze prediction errors using residuals.
- Make predictions for new observations.
"""

# %%
# 1. Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# %%
# 2. Load and inspect the dataset

dataset = load_diabetes(as_frame=True)
df = dataset.frame

print(f"Number of observations: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Available features: {dataset.feature_names}")

df.head()


# %%
# 3. Explore the data

print("Summary statistics:")
display(df.describe().round(2))

print("\nMissing values:")
display(df.isnull().sum())


# %%
# 4. Prepare features and target

# Use all ten input features.
X = df.drop(columns="target")

# Predict the target value.
y = df["target"]

print(f"Number of features: {X.shape[1]}")
print(f"Number of observations: {X.shape[0]}")


# %%
# 5. Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print(f"Training observations: {len(X_train)}")
print(f"Test observations: {len(X_test)}")


# %%
# 6. Train the multiple linear regression model

model = LinearRegression()

# Learn from training data only.
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_:.2f}")


# %%
# 7. Inspect the learned coefficients

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_,
})

coefficients = coefficients.sort_values(
    "Coefficient",
    ascending=False,
)

display(coefficients.round(2))


# %%
# 8. Make predictions

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

predictions = pd.DataFrame({
    "Actual": y_test.to_numpy()[:5],
    "Predicted": y_test_pred[:5],
})

print("First five test predictions:")
display(predictions.round(2))


# %%
# 9. Evaluate the model

def evaluate_model(y_true, y_pred):
    """Calculate common regression evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


train_metrics = evaluate_model(y_train, y_train_pred)
test_metrics = evaluate_model(y_test, y_test_pred)

comparison = pd.DataFrame({
    "Training": train_metrics,
    "Test": test_metrics,
})

print("Training vs. test performance:")
display(comparison.round(2))


# %%
# 10. Visualize multiple linear regression in 3D

# A model with two features can be visualized as a plane.
# Use BMI and blood pressure for this illustration.
#
# This is a separate model. The main model above still
# uses all ten features.

visualization_features = ["bmi", "bp"]

visualization_model = LinearRegression()

visualization_model.fit(
    X_train[visualization_features],
    y_train,
)

# Create a grid across the training feature ranges.
bmi_values = np.linspace(
    X_train["bmi"].min(),
    X_train["bmi"].max(),
    30,
)

bp_values = np.linspace(
    X_train["bp"].min(),
    X_train["bp"].max(),
    30,
)

bmi_grid, bp_grid = np.meshgrid(
    bmi_values,
    bp_values,
)

grid = pd.DataFrame({
    "bmi": bmi_grid.ravel(),
    "bp": bp_grid.ravel(),
})

# Predict a target value at every point on the grid.
target_grid = visualization_model.predict(grid)
target_grid = target_grid.reshape(bmi_grid.shape)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X_test["bmi"],
    X_test["bp"],
    y_test,
    alpha=0.6,
    label="Actual test observations",
)

ax.plot_surface(
    bmi_grid,
    bp_grid,
    target_grid,
    cmap="viridis",
    alpha=0.4,
)

ax.set_title("Multiple Linear Regression: Regression Plane")
ax.set_xlabel("BMI (standardized)")
ax.set_ylabel("Blood pressure (standardized)")
ax.set_zlabel("Target value")
ax.legend()

plt.tight_layout()
plt.show()


# %%
# 11. Visualize actual vs. predicted values

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_test_pred,
})

min_value = min(
    results["Actual"].min(),
    results["Predicted"].min(),
)

max_value = max(
    results["Actual"].max(),
    results["Predicted"].max(),
)

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=results,
    x="Actual",
    y="Predicted",
    s=70,
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="red",
    linestyle="--",
    label="Perfect prediction",
)

plt.title("Multiple Linear Regression: Actual vs. Predicted")
plt.xlabel("Actual target value")
plt.ylabel("Predicted target value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 12. Visualize residuals vs. predicted values

# Residual = actual value - predicted value.
residuals = y_test.to_numpy() - y_test_pred

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=y_test_pred,
    y=residuals,
    s=70,
)

plt.axhline(
    0,
    color="red",
    linestyle="--",
    label="Zero residual",
)

plt.title("Residuals vs. Predicted Values")
plt.xlabel("Predicted target value")
plt.ylabel("Residual (actual - predicted)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 13. Visualize the distribution of residuals

plt.figure(figsize=(8, 5))

sns.histplot(
    residuals,
    bins=15,
    kde=True,
)

plt.axvline(
    0,
    color="red",
    linestyle="--",
    label="Zero residual",
)

plt.title("Distribution of Prediction Errors")
plt.xlabel("Residual (actual - predicted)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


# %%
# 14. Predict for a new observation

# Use the mean feature values from the training set
# as an illustrative input.
new_observation = X_train.mean().to_frame().T

prediction = model.predict(new_observation)

print("New observation:")
display(new_observation.round(3))

print(f"Predicted target value: {prediction[0]:.2f}")
