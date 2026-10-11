"""
Machine Learning Foundations
Topic: Data Visualization and Exploratory Data Analysis (EDA)

Dataset:
Scikit-learn Diabetes dataset.

Learning objectives:
- Load a built-in dataset.
- Inspect data with Pandas.
- Visualize distributions and relationships.
- Train a simple linear regression model.
- Compare actual and predicted values.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. Load the dataset
# ============================================================

dataset = load_diabetes(as_frame=True)

df = dataset.frame

print("\n1. Dataset")
print(df.head())


# ============================================================
# 2. Inspect the dataset
# ============================================================

print("\n2. Dataset shape")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. Visualize the target distribution with Matplotlib
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(df["target"], bins=20, edgecolor="black")

plt.title("Distribution of Disease Progression")
plt.xlabel("Disease progression measure")
plt.ylabel("Number of patients")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 4. Explore BMI and the target using Matplotlib
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(df["bmi"], df["target"])

plt.title("BMI vs Disease Progression")
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease progression measure")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 5. Explore the relationship using Seaborn
# ============================================================

sns.set_theme(style="whitegrid")

sns.scatterplot(
    data=df,
    x="bmi",
    y="target",
    s=60,
)

plt.title("BMI vs Disease Progression")
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease progression measure")
plt.tight_layout()
plt.show()


# ============================================================
# 6. Train a linear regression model
# ============================================================

X = df[["bmi"]]
y = df["target"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("\n6. Linear regression model")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")


# ============================================================
# 7. Visualize the regression line using Matplotlib
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(X["bmi"], y, label="Actual observations", alpha=0.6)
plt.plot(X["bmi"], y_pred, label="Regression predictions")

plt.title("Linear Regression: BMI vs Disease Progression")
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease progression measure")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 8. Visualize the regression relationship using Seaborn
# ============================================================

plt.figure(figsize=(8, 5))

sns.regplot(
    data=df,
    x="bmi",
    y="target",
    ci=None,
    line_kws={"color": "red"},
)

plt.title("Linear Regression with Seaborn")
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease progression measure")
plt.tight_layout()
plt.show()


# ============================================================
# 9. Compare actual and predicted values
# ============================================================

results = pd.DataFrame({
    "Actual": y.to_numpy(),
    "Predicted": y_pred,
})

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=results,
    x="Actual",
    y="Predicted",
    s=60,
)

min_value = min(results["Actual"].min(), results["Predicted"].min())
max_value = max(results["Actual"].max(), results["Predicted"].max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="red",
    label="Perfect prediction",
)

plt.title("Actual vs Predicted Values")
plt.xlabel("Actual value")
plt.ylabel("Predicted value")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 10. Evaluate the model
# ============================================================

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\n10. Model evaluation")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.3f}")


# ============================================================
# 11. Predict for a new observation
# ============================================================

new_observation = pd.DataFrame({
    "bmi": [0.05]
})

prediction = model.predict(new_observation)

print("\n11. New prediction")
print(f"Standardized BMI: {new_observation['bmi'].iloc[0]}")
print(f"Predicted target value: {prediction[0]:.2f}")
