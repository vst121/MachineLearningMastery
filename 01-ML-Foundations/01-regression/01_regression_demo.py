
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

# 1. Prepare the dataset
X = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y = np.array([50, 55, 65, 70, 75, 85, 90])

# 2. Create the model
model = LinearRegression()

# 3. Train the model
model.fit(X, y)

# 4. Inspect the learned parameters
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

# 5. Predict a score for a student studying 8 hours
new_student = np.array([[8]])
prediction = model.predict(new_student)

print("Predicted score:", prediction[0])

# 6. Evaluate the model on the training data
y_pred = model.predict(X)

print("MSE:", mean_squared_error(y, y_pred))
print("R²:", r2_score(y, y_pred))

# 7. Plot actual data and regression line
plt.scatter(X.ravel(), y, label="Actual scores")
plt.plot(X.ravel(), y_pred, label="Regression line")

# Show the new prediction
plt.scatter([8], prediction, marker="x", s=100,
            label="Prediction for 8 hours")

plt.xlabel("Hours studied")
plt.ylabel("Exam score")
plt.title("Linear Regression: Study Hours vs Exam Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8. Visualize actual vs. predicted scores using Seaborn
results = pd.DataFrame({
    "Hours": X.ravel(),
    "Actual Score": y,
    "Predicted Score": y_pred
})

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=results,
    x="Actual Score",
    y="Predicted Score",
    s=100
)

# Perfect predictions lie on this diagonal
min_score = min(y.min(), y_pred.min())
max_score = max(y.max(), y_pred.max())

plt.plot(
    [min_score, max_score],
    [min_score, max_score],
    linestyle="--",
    color="red",
    label="Perfect prediction"
)

plt.title("Actual vs. Predicted Scores")
plt.xlabel("Actual score")
plt.ylabel("Predicted score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
