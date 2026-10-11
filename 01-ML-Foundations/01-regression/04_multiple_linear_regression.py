"""
Machine Learning Foundations
Topic: Multiple Linear Regression

Learning objectives:
- Use multiple input features for prediction.
- Understand model coefficients.
- Split data into training and test sets.
- Evaluate and compare models.
- Visualize a regression plane using two features.
- Visualize predictions and residuals.
"""

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


# ============================================================
# 1. Load the dataset
# ============================================================

dataset = load_diabetes(as_frame=True)
df = dataset.frame

print("\n1. Dataset")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nFeatures:")
print(dataset.feature_names)


# ============================================================
# 2. Explore the data
# ============================================================

print("\n2. Summary statistics")
print(df.describe().round(2))

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. Prepare features and target
# ============================================================

# Use all ten input features.
X = df.drop(columns="target")

# Predict the target.
y = df["target"]

print("\n3. Features and target")
print("Number of features:", X.shape[1])
print("Number of observations:", X.shape[0])


# ============================================================
# 4. Split the dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\n4. Dataset split")
print("Training observations:", len(X_train))
print("Test observations:", len(X_test))


# ============================================================
# 5. Train the multiple linear regression model
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\n5. Model parameters")
print(f"Intercept: {model.intercept_:.2f}")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_,
})

print("\nLearned coefficients:")
print(
    coefficients
    .sort_values("Coefficient", ascending=False)
    .to_string(index=False)
)


# ============================================================
# 6. Make predictions
# ============================================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n6. First five test predictions")

predictions = pd.DataFrame({
    "Actual": y_test.to_numpy()[:5],
    "Predicted": y_test_pred[:5],
})

print(predictions.round(2))


# ============================================================
# 7. Evaluate the model
# ============================================================

def evaluate_model(y_true, y_pred):
    """Calculate common regression metrics."""
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

print("\n7. Model evaluation")
print(comparison.round(2))


# ============================================================
# 8. Visualize multiple linear regression in 3D
# ============================================================

# A model with two features can be visualized as a plane.
# Use BMI and blood pressure for this illustration.
#
# Important:
# This is a separate two-feature model.
# The main model above still uses all ten features.

visualization_features = ["bmi", "bp"]

visualization_model = LinearRegression()

visualization_model.fit(
    X_train[visualization_features],
    y_train,
)

# Create a grid covering the training data's feature ranges.
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

bmi_grid, bp_grid = np.meshgrid(bmi_values, bp_values)

grid = pd.DataFrame({
    "bmi": bmi_grid.ravel(),
    "bp": bp_grid.ravel(),
})

# Predict a target value for each point on the grid.
target_grid = visualization_model.predict(grid)
target_grid = target_grid.reshape(bmi_grid.shape)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Show actual test observations.
ax.scatter(
    X_test["bmi"],
    X_test["bp"],
    y_test,
    color="blue",
    alpha=0.6,
    label="Actual test observations",
)

# Show the fitted regression plane.
ax.plot_surface(
    bmi_grid,
    bp_grid,
    target_grid,
    cmap="viridis",
    alpha=0.4,
)

ax.set_title("Multiple Linear Regression: Regression Plane")
ax.set_xlabel("BMI")
ax.set_ylabel("Blood pressure")
ax.set_zlabel("Target value")
ax.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 9. Visualize actual vs. predicted values
# ============================================================

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_test_pred,
})

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=results,
    x="Actual",
    y="Predicted",
    s=70,
)

min_value = min(
    results["Actual"].min(),
    results["Predicted"].min(),
)
max_value = max(
    results["Actual"].max(),
    results["Predicted"].max(),
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
plt.tight_layout()
plt.show()


# ============================================================
# 10. Visualize residuals
# ============================================================

residuals = y_test.to_numpy() - y_test_pred

plt.figure(figsize=(8, 5))

sns.histplot(
    residuals,
    bins=15,
    kde=True,
)

plt.axvline(0, color="red", linestyle="--")

plt.title("Distribution of Prediction Errors")
plt.xlabel("Residual (actual - predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ============================================================
# 11. Predict for a new observation
# ============================================================

# Use the mean feature values as an example input.
new_observation = X_train.mean().to_frame().T

prediction = model.predict(new_observation)

print("\n11. New prediction")
print(f"Predicted target value: {prediction[0]:.2f}")
