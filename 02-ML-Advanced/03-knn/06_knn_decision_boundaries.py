# %% [markdown]
# # KNN Decision Boundaries
#
# Objectives:
# - Visualize how KNN classifies a two-dimensional feature space.
# - Understand how k affects decision boundaries.
# - Compare training observations with predicted regions.
# - Observe the relationship between model complexity and generalization.

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# ## 1. Load Two Features from the Iris Dataset
#
# We use petal length and petal width because they provide
# a useful two-dimensional view of Iris classification.

# %%
iris = load_iris()

feature_indices = [2, 3]
feature_names = [
    iris.feature_names[i] for i in feature_indices
]

X = iris.data[:, feature_indices]
y = iris.target

print("Features:", feature_names)
print("Dataset shape:", X.shape)
print("Classes:", iris.target_names)


# %% [markdown]
# ## 2. Split the Dataset

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# %% [markdown]
# ## 3. Plot the Training Observations
#
# Each point represents a flower. Its color indicates the actual class.

# %%
plt.figure(figsize=(9, 6))

for class_index, class_name in enumerate(iris.target_names):
    mask = y_train == class_index

    plt.scatter(
        X_train[mask, 0],
        X_train[mask, 1],
        label=class_name,
        edgecolors="black",
        alpha=0.8,
    )

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Iris Training Data")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4. Create a Grid Across the Feature Space
#
# The grid represents possible flower measurements.
# KNN predicts a class for every point on the grid.

# %%
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

grid_step = 0.02

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, grid_step),
    np.arange(y_min, y_max, grid_step),
)

grid_points = np.c_[xx.ravel(), yy.ravel()]

print("Grid points:", len(grid_points))


# %% [markdown]
# ## 5. Visualize Decision Boundaries for Different k Values
#
# Each model uses standardization and uniform voting.
# Only k changes between experiments.

# %%
k_values = [1, 5, 15]

fig, axes = plt.subplots(
    1,
    len(k_values),
    figsize=(18, 5),
    sharex=True,
    sharey=True,
)

for ax, k in zip(axes, k_values):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k,
                    weights="uniform",
                    metric="minkowski",
                    p=2,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    grid_predictions = model.predict(grid_points).reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        grid_predictions,
        alpha=0.25,
        levels=np.arange(-0.5, len(iris.target_names) + 0.5, 1),
        cmap="viridis",
    )

    for class_index, class_name in enumerate(iris.target_names):
        mask = y_train == class_index

        ax.scatter(
            X_train[mask, 0],
            X_train[mask, 1],
            label=class_name,
            edgecolors="black",
            s=30,
        )

    ax.set_title(f"k = {k}")
    ax.set_xlabel(feature_names[0])
    ax.grid(alpha=0.15)

axes[0].set_ylabel(feature_names[1])
axes[-1].legend(loc="best")

fig.suptitle("KNN Decision Boundaries for Different k Values")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6. Compare Training and Test Accuracy
#
# These results help relate the decision-boundary shapes
# to model performance. They are exploratory, not a method
# for selecting k from the test set.

# %%
from sklearn.metrics import accuracy_score

accuracy_results = []

for k in k_values:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(n_neighbors=k),
            ),
        ]
    )

    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train,
        model.predict(X_train),
    )

    test_accuracy = accuracy_score(
        y_test,
        model.predict(X_test),
    )

    accuracy_results.append(
        {
            "k": k,
            "Training Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
        }
    )

for result in accuracy_results:
    print(
        f"k={result['k']:2d} | "
        f"Training accuracy={result['Training Accuracy']:.4f} | "
        f"Test accuracy={result['Test Accuracy']:.4f}"
    )
