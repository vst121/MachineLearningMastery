# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% 2. Load the dataset
iris = load_iris()

# Select two classes for binary classification
class_mask = iris.target < 2
X = iris.data[class_mask, :2]
y = iris.target[class_mask]

feature_names = iris.feature_names[:2]
class_names = iris.target_names[:2]

print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Classes: {class_names}")
print(f"Class counts: {np.bincount(y)}")

# %% 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# %% 4. Build and train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, max_iter=1000, random_state=42),
)

model.fit(X_train, y_train)

# %% 5. Generate predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("First 10 actual labels:     ", y_test[:10])
print("First 10 predicted labels:  ", y_pred[:10])
print("First 10 predicted probabilities:")
print(np.round(y_proba[:10], 3))

# %% 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
)

# %% 7. Visualize the decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = model.predict(grid).reshape(xx.shape)
grid_probabilities = model.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(9, 6))

plt.contourf(
    xx,
    yy,
    grid_predictions,
    alpha=0.25,
    cmap="coolwarm",
)

plt.contour(
    xx,
    yy,
    grid_probabilities,
    levels=[0.5],
    colors="black",
    linewidths=2,
)

scatter = plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap="coolwarm",
    edgecolors="black",
    s=55,
)

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Logistic Regression: Iris Binary Classification")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=list(class_names),
    title="Class",
)
plt.tight_layout()
plt.show()
