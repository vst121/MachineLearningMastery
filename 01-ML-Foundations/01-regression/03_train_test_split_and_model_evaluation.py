"""
Machine Learning Foundations
Topic: Train/Test Split and Model Evaluation

Dataset:
Scikit-learn Diabetes dataset.

Learning objectives:
- Load and inspect a built-in dataset.
- Split a dataset into training and test sets.
- Train a model using training data only.
- Understand the slope and intercept of linear regression.
- Evaluate predictions using MAE, MSE, RMSE, and R².
- Compare training and test performance.
- Visualize the relationship between a feature and the target.
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

# Use BMI as the single input feature.
X = df[["bmi"]]
y = df["target"]

print(f"Number of observations: {len(df)}")
print(f"Number of features: {X.shape[1]}")

df.head()


# %%
# 3. Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print(f"Training observations: {len(X_train)}")
print(f"Test observations: {len(X_test)}")


# %%
# 4. Train the model

model = LinearRegression()

# Learn the relationship using training data only.
model.fit(X_train, y_train)

intercept = model.intercept_
slope = model.coef_[0]

print(f"Intercept: {intercept:.2f}")
print(f"Slope: {slope:.2f}")

print("\nLinear regression equation:")
print(
    f"Predicted target = {intercept:.2f} "
    f"+ ({slope:.2f} * BMI)"
)


# %%
# 5. Visualize the relationship and regression line

# Generate BMI values across the observed feature range.
bmi_range = pd.DataFrame({
    "bmi": np.linspace(
        X["bmi"].min(),
        X["bmi"].max(),
        100,
    )
})

target_predictions = model.predict(bmi_range)

plt.figure(figsize=(8, 5))

plt.scatter(
    X_train["bmi"],
    y_train,
    alpha=0.6,
    label="Training observations",
)

plt.scatter(
    X_test["bmi"],
    y_test,
    alpha=0.6,
    marker="x",
    label="Test observations",
)

plt.plot(
    bmi_range["bmi"],
    target_predictions,
    color="red",
    linewidth=2,
    label="Fitted regression line",
)

plt.title("Simple Linear Regression: BMI vs. Target")
plt.xlabel("BMI (standardized)")
plt.ylabel("Target value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 6. Make predictions

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

predictions = pd.DataFrame({
    "Actual": y_test.to_numpy()[:5],
    "Predicted": y_test_pred[:5],
})

print("First five test predictions:")
predictions.round(2)


# %%
# 7. Evaluate the model

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
comparison.round(2)


# %%
# 8. Visualize actual vs. predicted values using Matplotlib

min_value = min(y_test.min(), y_test_pred.min())
max_value = max(y_test.max(), y_test_pred.max())

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_test_pred,
    alpha=0.7,
)

# Points on this line represent perfect predictions.
plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="red",
    label="Perfect prediction",
)

plt.title("Actual vs. Predicted Values (Test Set)")
plt.xlabel("Actual target value")
plt.ylabel("Predicted target value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 9. Visualize actual vs. predicted values using Seaborn

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

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="red",
    label="Perfect prediction",
)

plt.title("Model Performance on Unseen Data")
plt.xlabel("Actual target value")
plt.ylabel("Predicted target value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 10. Visualize prediction errors (residuals)

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

plt.title("Residuals vs. Predicted Values (Test Set)")
plt.xlabel("Predicted target value")
plt.ylabel("Residual (actual - predicted)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 11. Visualize the distribution of residuals

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

plt.title("Distribution of Prediction Errors (Test Set)")
plt.xlabel("Residual (actual - predicted)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


# %%
# 12. Make a prediction for a new observation

# BMI is standardized in this dataset, not expressed as
# a person's usual BMI value.
new_observation = pd.DataFrame({
    "bmi": [0.05],
})

prediction = model.predict(new_observation)

print(f"Standardized BMI: {new_observation['bmi'].iloc[0]}")
print(f"Predicted target value: {prediction[0]:.2f}")
