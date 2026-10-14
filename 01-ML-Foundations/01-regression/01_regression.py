"""
Machine Learning Foundations
Topic: Simple Linear Regression Demo

Dataset:
A small, illustrative dataset relating study hours to exam scores.

Learning objectives:
- Prepare a dataset using NumPy and pandas.
- Train a simple linear regression model.
- Understand the intercept and slope.
- Make predictions for new observations.
- Evaluate predictions using MSE and R².
- Visualize the regression line.
- Compare actual and predicted values.
- Understand prediction errors (residuals).
"""

# %%
# 1. Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# %%
# 2. Prepare the dataset

# Study hours are the input feature.
X = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)

# Exam scores are the target values.
y = np.array([50, 55, 65, 70, 75, 85, 90])

df = pd.DataFrame({
    "Hours": X.ravel(),
    "Exam Score": y,
})

print("Dataset:")
print(df)


# %%
# 3. Train the model

model = LinearRegression()

model.fit(X, y)

print("Model trained successfully.")


# %%
# 4. Inspect the learned parameters

intercept = model.intercept_
slope = model.coef_[0]

print(f"Intercept: {intercept:.2f}")
print(f"Slope: {slope:.2f}")

print(
    f"Predicted score = {intercept:.2f} "
    f"+ ({slope:.2f} * study hours)"
)


# %%
# 5. Make predictions

# Predict the score for a student studying 8 hours.
new_student = np.array([[8]])

prediction = model.predict(new_student)

print(f"Study hours: {new_student[0, 0]}")
print(f"Predicted exam score: {prediction[0]:.2f}")


# %%
# 6. Evaluate the model on the training data

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}")

results = pd.DataFrame({
    "Hours": X.ravel(),
    "Actual Score": y,
    "Predicted Score": y_pred,
    "Residual": y - y_pred,
})

print("\nActual vs. predicted scores:")
print(results.round(2))


# %%
# 7. Visualize the data and regression line

plt.figure(figsize=(8, 5))

plt.scatter(
    X.ravel(),
    y,
    label="Actual scores",
    s=70,
)

plt.plot(
    X.ravel(),
    y_pred,
    color="red",
    label="Regression line",
)

# Show the prediction for a student studying 8 hours.
plt.scatter(
    [8],
    prediction,
    marker="x",
    s=120,
    color="green",
    label="Prediction for 8 hours",
)

plt.xlabel("Hours studied")
plt.ylabel("Exam score")
plt.title("Linear Regression: Study Hours vs. Exam Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 8. Visualize actual vs. predicted scores using Seaborn

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=results,
    x="Actual Score",
    y="Predicted Score",
    s=100,
)

# Perfect predictions lie on this diagonal.
min_score = min(y.min(), y_pred.min())
max_score = max(y.max(), y_pred.max())

plt.plot(
    [min_score, max_score],
    [min_score, max_score],
    linestyle="--",
    color="red",
    label="Perfect prediction",
)

plt.title("Actual vs. Predicted Exam Scores")
plt.xlabel("Actual score")
plt.ylabel("Predicted score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 9. Visualize prediction errors (residuals)

# Residual = actual value - predicted value.
residuals = y - y_pred

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=y_pred,
    y=residuals,
    s=100,
)

plt.axhline(
    0,
    color="red",
    linestyle="--",
    label="Zero residual",
)

plt.title("Residuals vs. Predicted Exam Scores")
plt.xlabel("Predicted score")
plt.ylabel("Residual (actual - predicted)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
