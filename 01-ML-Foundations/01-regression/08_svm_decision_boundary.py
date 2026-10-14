# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# %% Load and prepare the Iris dataset
iris = load_iris()

# Use two features so the decision boundaries can be visualized in 2D.
# Compare Versicolor (1) and Virginica (2).
X = iris.data[iris.target != 0, 2:4]
y = iris.target[iris.target != 0]

feature_names = iris.feature_names[2:]
class_names = iris.target_names[1:]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# %% Train linear and RBF SVM models
models = {
    "Linear kernel": SVC(kernel="linear", C=1.0),
    "RBF kernel": SVC(kernel="rbf", C=1.0, gamma="scale"),
}

trained_models = {}

for name, svm in models.items():
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", svm),
        ]
    )

    model.fit(X_train, y_train)
    trained_models[name] = model

    predictions = model.predict(X_test)
    print(f"{name}:")
    print(f"  Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"  Support vectors: {model.named_steps['svc'].n_support_}")
    print()


# %% Visualize decision boundaries and support vectors
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ["tab:blue", "tab:orange"]
class_labels = {
    1: class_names[0],
    2: class_names[1],
}

for ax, (name, model) in zip(axes, trained_models.items()):
    scaler = model.named_steps["scaler"]
    svm = model.named_steps["svc"]

    # Transform features because the model was trained on scaled data.
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
    y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.column_stack([xx.ravel(), yy.ravel()])
    decision_values = svm.decision_function(grid).reshape(xx.shape)

    # Shade the predicted class regions.
    ax.contourf(
        xx,
        yy,
        decision_values,
        levels=[-np.inf, 0, np.inf],
        colors=colors,
        alpha=0.15,
    )

    # Plot the decision boundary and margins.
    ax.contour(
        xx,
        yy,
        decision_values,
        levels=[-1, 0, 1],
        colors="black",
        linestyles=["--", "-", "--"],
        linewidths=[1, 2, 1],
    )

    # Plot training samples.
    for class_id, color in zip([1, 2], colors):
        mask = y_train == class_id
        ax.scatter(
            X_train_scaled[mask, 0],
            X_train_scaled[mask, 1],
            color=color,
            label=class_labels[class_id],
            edgecolors="black",
            s=45,
        )

    # Highlight support vectors.
    support_vectors = svm.support_vectors_
    ax.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=150,
        facecolors="none",
        edgecolors="red",
        linewidths=1.8,
        label="Support vectors",
    )

    ax.set_title(name)
    ax.set_xlabel(f"{feature_names[0]} (standardized)")
    ax.set_ylabel(f"{feature_names[1]} (standardized)")
    ax.legend()

plt.suptitle("SVM Decision Boundaries: Linear vs RBF")
plt.tight_layout()
plt.show()

## Key concepts
# - Decision boundary: The line or curve that separates the predicted classes. It is where the decision function equals zero.
# - Margins: The dashed lines show decision-function levels of -1 and +1. For a linear SVM, these correspond to the margin boundaries.
# - Support vectors: The red circles identify training samples that determine the learned boundary.
# - Linear kernel: Learns a straight decision boundary in the feature space.
# - RBF kernel: Can learn nonlinear boundaries by representing more complex relationships between features.
# - Feature scaling: Standardization helps prevent features with larger numerical ranges from dominating the model.