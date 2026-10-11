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
- Evaluate model performance on unseen data.
- Visualize prediction errors (residuals).
"""

# %%
# 1. Imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# %%
# 2. Load the dataset

dataset = load_diabetes(as_frame=True)
df = dataset.frame

print("Dataset preview:")
df.head()


# %%
# 3. Inspect the dataset

print(f"Number of observations: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())


# %%
# 4. Visualize the target distribution

plt.figure(figsize=(8, 5))

plt.hist(
    df["target"],
    bins=20,
    edgecolor="black",
)

plt.title("Distribution of Disease Progression")
plt.xlabel("Disease progression measure")
plt.ylabel("Number of patients")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 5. Explore BMI and the target using Matplotlib

plt.figure(figsize=(8, 5))

plt.scatter(
    df["bmi"],
    df["target"],
    alpha=0.7,
)

plt.title("BMI vs. Disease Progression")
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease progression measure")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 6. Explore the relationship using Seaborn

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="bmi",
    y="target",
    s=60,
)

plt.title("BMI vs. Disease Progression")
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease progression measure")
plt.tight_layout()
plt.show()


# %%
# 7. Prepare the training and test sets

# Use BMI as the single input feature.
X = df[["bmi"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print(f"Training observations: {len(X_train)}")
print(f"Test observations: {len(X_test)}")


# %%
# 8. Train the linear regression model

model = LinearRegression()

# Fit the model using training data only.
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")

print(
    f"Predicted target = {model.intercept_:.2f} "
    f"+ ({model.coef_[0]:.2f} * BMI)"
)


# %%
# 9. Visualize the regression line

# Generate evenly spaced BMI values so the line is plotted smoothly.
bmi_range = pd.DataFrame({
    "bmi": [
        X["bmi"].min(),
        X["bmi"].max(),
    ]
})

# Use more points between the minimum and maximum BMI.
import numpy as np

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
    alpha=0.5,
    label="Training observations",
)

plt.scatter(
    X_test["bmi"],
    y_test,
    alpha=0.7,
    marker="x",
    label="Test observations",
)

plt.plot(
    bmi_range["bmi"],
    target_predictions,
    color="red",
    linewidth=2,
    label="Regression line",
)

plt.title("Linear Regression: BMI vs. Disease Progression")
plt.xlabel("BMI (standardized)")
plt.ylabel("Disease progression measure")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 10. Compare actual and predicted values

y_test_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_test_pred,
})

print(results.head())

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=results,
    x="Actual",
    y="Predicted",
    s=60,
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
    linestyle="--",
    color="red",
    label="Perfect prediction",
)

plt.title("Actual vs. Predicted Values (Test Set)")
plt.xlabel("Actual value")
plt.ylabel("Predicted value")
plt.legend()
plt.tight_layout()
plt.show()


# %%
# 11. Evaluate the model

mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.3f}")


# %%
# 12. Visualize prediction errors (residuals)

# Residual = actual value - predicted value.
residuals = y_test.to_numpy() - y_test_pred

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=y_test_pred,
    y=residuals,
    s=60,
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
plt.tight_layout()
plt.show()


# %%
# 13. Predict for a new observation

# BMI is standardized in this dataset.
new_observation = pd.DataFrame({
    "bmi": [0.05],
})

prediction = model.predict(new_observation)

print(f"Standardized BMI: {new_observation['bmi'].iloc[0]}")
print(f"Predicted target value: {prediction[0]:.2f}")
