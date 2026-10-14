# %% Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# %% Load and prepare the Iris dataset
iris = load_iris()

# Use two classes and two features for clear visualization.
# Classes: Versicolor and Virginica.
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


# %% Train RBF SVM models with different gamma values
gamma_values = [0.01, 0.1, 1, 10, 100]
models = {}

for gamma in gamma_values:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma=gamma)),
        ]
    )

    model.fit(X_train, y_train)
    models[gamma] = model

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    print(f"Gamma = {gamma}")
    print(f"  Training accuracy: {train_accuracy:.3f}")
    print(f"  Test accuracy:     {test_accuracy:.3f}")
    print(f"  Support vectors:   {model.named_steps['svc'].n_support_}")
    print()


# %% Plot decision boundaries
fig, axes = plt.subplots(1, len(gamma_values), figsize=(20, 4.5))

# Transform data using the scaler fitted by each pipeline.
X_train_scaled = models[gamma_values[0]].named_steps["scaler"].transform(
    X_train
)

x_min = X_train_scaled[:, 0].min() - 0.7
x_max = X_train_scaled[:, 0].max() + 0.7
y_min = X_train_scaled[:, 1].min() - 0.7
y_max = X_train_scaled[:, 1].max() + 0.7

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 250),
    np.linspace(y_min, y_max, 250),
)

grid = np.column_stack([xx.ravel(), yy.ravel()])

for ax, gamma in zip(axes, gamma_values):
    model = models[gamma]
    scaler = model.named_steps["scaler"]
    svm = model.named_steps["svc"]

    # The grid must be transformed with the corresponding fitted scaler.
    grid_scaled = grid
    decision_values = svm.decision_function(grid_scaled).reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        decision_values,
        levels=[-np.inf, 0, np.inf],
        colors=["tab:blue", "tab:orange"],
        alpha=0.2,
    )

    ax.contour(
        xx,
        yy,
        decision_values,
        levels=[-1, 0, 1],
        colors="black",
        linestyles=["--", "-", "--"],
        linewidths=[1, 2, 1],
    )

    for class_id, color in zip([1, 2], ["tab:blue", "tab:orange"]):
        mask = y_train == class_id
        ax.scatter(
            X_train_scaled[mask, 0],
            X_train_scaled[mask, 1],
            color=color,
            edgecolors="black",
            s=35,
            label=class_names[class_id - 1],
        )

    support_vectors = svm.support_vectors_
    ax.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        label="Support vectors",
    )

    ax.set_title(f"gamma = {gamma}")
    ax.set_xlabel(f"{feature_names[0]} (standardized)")
    ax.set_ylabel(f"{feature_names[1]} (standardized)")

axes[0].legend(loc="best", fontsize=8)
fig.suptitle("Effect of Gamma on RBF SVM Decision Boundaries")
plt.tight_layout()
plt.show()


# %% Compare training and test accuracy
train_scores = [
    accuracy_score(y_train, models[g].predict(X_train))
    for g in gamma_values
]
test_scores = [
    accuracy_score(y_test, models[g].predict(X_test))
    for g in gamma_values
]

plt.figure(figsize=(8, 5))
plt.plot(gamma_values, train_scores, marker="o", label="Training accuracy")
plt.plot(gamma_values, test_scores, marker="s", label="Test accuracy")

plt.xscale("log")
plt.xlabel("Gamma (log scale)")
plt.ylabel("Accuracy")
plt.title("Training vs Test Accuracy for Different Gamma Values")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
