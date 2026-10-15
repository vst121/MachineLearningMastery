# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# %% 2. Load Iris with Two Features
iris = load_iris()

# Use petal length and petal width for 2D visualization.
feature_indices = [2, 3]
feature_names = [
    iris.feature_names[index]
    for index in feature_indices
]

X = pd.DataFrame(
    iris.data[:, feature_indices],
    columns=feature_names,
)

y = iris.target

class_names = iris.target_names

print("Features:", feature_names)
print("Dataset shape:", X.shape)
print("Classes:", class_names)


# %% 3. Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))


# %% 4. Train Trees with Different Depths
depths = [1, 2, 3, None]

models = {}

results = []

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        criterion="gini",
        random_state=42,
    )

    model.fit(X_train, y_train)
    models[depth] = model

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    results.append({
        "Max Depth": str(depth),
        "Actual Depth": model.get_depth(),
        "Leaves": model.get_n_leaves(),
        "Train Accuracy": accuracy_score(
            y_train,
            train_predictions,
        ),
        "Test Accuracy": accuracy_score(
            y_test,
            test_predictions,
        ),
    })

results_df = pd.DataFrame(results)

print("\nTree Complexity and Accuracy:")
print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 5. Create a Mesh Grid for Decision Boundaries
x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400),
)

grid = pd.DataFrame(
    np.c_[xx.ravel(), yy.ravel()],
    columns=feature_names,
)


# %% 6. Visualize Decision Boundaries
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.ravel()

for axis, (depth, model) in zip(axes, models.items()):
    grid_predictions = model.predict(grid).reshape(xx.shape)

    axis.contourf(
        xx,
        yy,
        grid_predictions,
        alpha=0.25,
        levels=np.arange(len(class_names) + 1) - 0.5,
        cmap="viridis",
    )

    for class_index, class_name in enumerate(class_names):
        mask = y_train == class_index

        axis.scatter(
            X_train.iloc[mask, 0],
            X_train.iloc[mask, 1],
            label=class_name,
            edgecolors="black",
            s=45,
        )

    axis.set_xlabel(feature_names[0])
    axis.set_ylabel(feature_names[1])
    axis.set_title(
        f"Max Depth = {depth}\n"
        f"Train: {accuracy_score(y_train, model.predict(X_train)):.3f}, "
        f"Test: {accuracy_score(y_test, model.predict(X_test)):.3f}"
    )
    axis.legend()

plt.suptitle(
    "Decision Tree Boundaries at Different Depths",
    fontsize=15,
)
plt.tight_layout()
plt.show()


# %% 7. Visualize the Tree Structures
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for axis, depth in zip(axes, [1, 2, 3]):
    model = models[depth]

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        ax=axis,
        fontsize=8,
    )

    axis.set_title(
        f"Depth Limit: {depth}\n"
        f"Actual Depth: {model.get_depth()}, "
        f"Leaves: {model.get_n_leaves()}"
    )

plt.tight_layout()
plt.show()


# %% 8. Inspect the Decision Rules
model = models[3]

print("\nDecision Rules for the Depth-3 Tree:")

from sklearn.tree import export_text

rules = export_text(
    model,
    feature_names=feature_names,
    decimals=2,
)

print(rules)


# %% 9. Predict Example Samples
samples = pd.DataFrame(
    [
        [1.5, 0.2],
        [4.5, 1.5],
        [5.5, 2.0],
    ],
    columns=feature_names,
)

predictions = model.predict(samples)
probabilities = model.predict_proba(samples)

prediction_results = samples.copy()

prediction_results["Predicted Species"] = [
    class_names[index]
    for index in predictions
]

for class_index, class_name in enumerate(class_names):
    prediction_results[f"P({class_name})"] = probabilities[:, class_index]

print("\nExample Predictions:")
print(
    prediction_results.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)
