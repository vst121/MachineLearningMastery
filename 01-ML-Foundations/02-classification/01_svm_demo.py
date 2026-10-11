"""
Machine Learning Foundations
Topic: Support Vector Machine (SVM) Demo

Dataset:
Scikit-learn Iris dataset.

Learning objectives:
- Understand the basic idea of classification.
- Train a linear SVM classifier.
- Understand the decision boundary and margin.
- Identify support vectors.
- Make predictions for new observations.
- Evaluate classification accuracy.
- Visualize the decision boundary in 2D.
"""

# %%
# 1. Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# %%
# 2. Prepare the dataset

dataset = load_iris(as_frame=True)
df = dataset.frame

# Use only two species to create a binary classification problem.
# Setosa = 0, Versicolor = 1, Virginica = 2.
df = df[df["target"].isin([0, 1])].copy()

# Select two features so the decision boundary can be visualized in 2D.
feature_names = ["petal length (cm)", "petal width (cm)"]

X = df[feature_names]
y = df["target"]

target_names = dataset.target_names[:2]

print("Dataset preview:")
print(df[feature_names + ["target"]].head())

print("\nClasses:")
for class_id, class_name in enumerate(target_names):
    print(f"{class_id} = {class_name}")

print(f"\nNumber of observations: {len(df)}")
print(f"Number of features: {X.shape[1]}")


# %%
# 3. Visualize the data

plt.figure(figsize=(8, 5))

for class_id, class_name in enumerate(target_names):
    class_data = df[df["target"] == class_id]

    plt.scatter(
        class_data[feature_names[0]],
        class_data[feature_names[1]],
        label=class_name,
        s=60,
        alpha=0.8,
    )

plt.title("Iris Flower Classification")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 4. Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training observations: {len(X_train)}")
print(f"Test observations: {len(X_test)}")


# %%
# 5. Train the linear SVM model

# A linear SVM learns a straight decision boundary
# that separates the two classes with the largest possible margin.
model = SVC(kernel="linear")

model.fit(X_train, y_train)

print("SVM model trained successfully.")
print(f"Number of support vectors: {model.n_support_.sum()}")


# %%
# 6. Make predictions

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred,
})

results["Actual species"] = results["Actual"].map(
    dict(enumerate(target_names))
)

results["Predicted species"] = results["Predicted"].map(
    dict(enumerate(target_names))
)

print("\nPrediction results:")
print(results.head(10))


# %%
# 7. Evaluate the model

accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.2%}")


# %%
# 8. Visualize the decision boundary and margin

# Create a grid covering the feature space.
feature_1_min = X[feature_names[0]].min() - 0.5
feature_1_max = X[feature_names[0]].max() + 0.5

feature_2_min = X[feature_names[1]].min() - 0.5
feature_2_max = X[feature_names[1]].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(feature_1_min, feature_1_max, 300),
    np.linspace(feature_2_min, feature_2_max, 300),
)

grid = pd.DataFrame({
    feature_names[0]: xx.ravel(),
    feature_names[1]: yy.ravel(),
})

# Predict the class for every point in the grid.
grid_predictions = model.predict(grid).reshape(xx.shape)

# Calculate the SVM decision function.
# For a binary linear SVM:
#   0  = decision boundary
#  -1  = one margin boundary
#  +1  = the other margin boundary
decision_values = model.decision_function(grid).reshape(xx.shape)

plt.figure(figsize=(10, 7))

# Shade the regions classified as each species.
plt.contourf(
    xx,
    yy,
    grid_predictions,
    alpha=0.2,
    cmap="coolwarm",
)

# Draw the decision boundary and the two margin boundaries.
plt.contour(
    xx,
    yy,
    decision_values,
    levels=[-1, 0, 1],
    colors="black",
    linestyles=["--", "-", "--"],
)

# Plot the training observations.
for class_id, class_name in enumerate(target_names):
    class_data = X_train[y_train == class_id]

    plt.scatter(
        class_data[feature_names[0]],
        class_data[feature_names[1]],
        label=class_name,
        s=60,
        edgecolors="black",
        alpha=0.8,
    )

# Highlight the support vectors.
support_vectors = model.support_vectors_

plt.scatter(
    support_vectors[:, 0],
    support_vectors[:, 1],
    s=180,
    facecolors="none",
    edgecolors="black",
    linewidths=1.8,
    label="Support vectors",
)

plt.title("Linear SVM: Decision Boundary and Margin")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 9. Predict for a new observation

new_observation = pd.DataFrame({
    "petal length (cm)": [2.0],
    "petal width (cm)": [0.5],
})

prediction = model.predict(new_observation)[0]

print("\nNew observation:")
print(new_observation)

print(f"Predicted species: {target_names[prediction]}")
