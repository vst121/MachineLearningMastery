
"""
03_train_test_split_and_model_evaluation.py

Machine Learning Foundations
Topic: Train/Test Split and Model Evaluation

Learning objectives:
- Split a dataset into training and test sets.
- Train a model using training data only.
- Evaluate predictions on unseen data.
- Understand MAE, MSE, RMSE, and R².
- Visualize model performance using Matplotlib and Seaborn.
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

# Use BMI to predict the target.
X = df[["bmi"]]
y = df["target"]

print("\n1. Dataset")
print(f"Number of observations: {len(df)}")
print(f"Number of features: {X.shape[1]}")
print(df.head())


# ============================================================
# 2. Split the dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\n2. Dataset split")
print(f"Training observations: {len(X_train)}")
print(f"Test observations: {len(X_test)}")


# ============================================================
# 3. Train the model
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\n3. Trained model")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")


# ============================================================
# 4. Make predictions
# ============================================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n4. Predictions")
print("First five test predictions:")

print(
    pd.DataFrame({
        "Actual": y_test.iloc[:5].to_numpy(),
        "Predicted": y_test_pred[:5],
    })
)


# ============================================================
# 5. Evaluate the model
# ============================================================

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

comparison = pd.DataFrame(
    {
        "Training": train_metrics,
        "Test": test_metrics,
    }
)

print("\n5. Model evaluation")
print(comparison.round(2))


# ============================================================
# 6. Visualize actual vs. predicted values using Matplotlib
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_test_pred, alpha=0.7)

min_value = min(y_test.min(), y_test_pred.min())
max_value = max(y_test.max(), y_test_pred.max())

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


# ============================================================
# 7. Visualize actual vs. predicted values using Seaborn
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
plt.tight_layout()
plt.show()


# ============================================================
# 8. Visualize prediction errors (residuals)
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
# 9. Make a prediction for a new observation
# ============================================================

new_observation = pd.DataFrame({
    "bmi": [0.05],
})

prediction = model.predict(new_observation)

print("\n9. Prediction for a new observation")
print(f"Standardized BMI: {new_observation['bmi'].iloc[0]}")
print(f"Predicted target value: {prediction[0]:.2f}")
