"""
Machine Learning Foundations
Topic: Multiple Linear Regression

Learning objectives:
- Use multiple input features for prediction.
- Understand model coefficients.
- Split data into training and test sets.
- Evaluate and compare models.
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
# 5. Train the model
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
# 8. Visualize actual vs. predicted values
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

min_value = min(results["Actual"].min(),
                results["Predicted"].min())
max_value = max(results["Actual"].max(),
                results["Predicted"].max())

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
# 9. Visualize residuals
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
# 10. Predict for a new observation
# ============================================================

# Use the mean feature values as a valid example input.
new_observation = X_train.mean().to_frame().T

prediction = model.predict(new_observation)

print("\n10. New prediction")
print(f"Predicted target value: {prediction[0]:.2f}")
