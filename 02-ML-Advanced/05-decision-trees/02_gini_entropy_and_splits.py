# %% 1. Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


# %% 2. Define Gini Impurity and Entropy
def gini_impurity(labels):
    """Calculate Gini impurity for a collection of class labels."""
    if len(labels) == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    return 1.0 - np.sum(probabilities**2)


def entropy(labels):
    """Calculate entropy in bits for a collection of class labels."""
    if len(labels) == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    return -np.sum(
        probabilities * np.log2(probabilities)
    )


def weighted_impurity(left_labels, right_labels, impurity_function):
    """Calculate the weighted impurity after a binary split."""
    total_samples = len(left_labels) + len(right_labels)

    if total_samples == 0:
        return 0.0

    left_weight = len(left_labels) / total_samples
    right_weight = len(right_labels) / total_samples

    return (
        left_weight * impurity_function(left_labels)
        + right_weight * impurity_function(right_labels)
    )


# %% 3. Understand Impurity with Simple Examples
examples = {
    "Pure": np.array(["A", "A", "A", "A"]),
    "Balanced": np.array(["A", "A", "B", "B"]),
    "Mostly A": np.array(["A", "A", "A", "B"]),
}

example_results = []

for name, labels in examples.items():
    example_results.append({
        "Example": name,
        "Gini": gini_impurity(labels),
        "Entropy": entropy(labels),
    })

example_df = pd.DataFrame(example_results)

print("Impurity of Different Class Distributions:")
print(example_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# %% 4. Calculate Information Gain for a Manual Split
parent = np.array(["A", "A", "A", "B", "B", "B"])

left = np.array(["A", "A", "A", "B"])
right = np.array(["B", "B"])

parent_entropy = entropy(parent)

child_entropy = weighted_impurity(
    left,
    right,
    entropy,
)

information_gain = parent_entropy - child_entropy

parent_gini = gini_impurity(parent)

child_gini = weighted_impurity(
    left,
    right,
    gini_impurity,
)

gini_reduction = parent_gini - child_gini

print("\nManual Split Analysis")
print("Parent entropy:", round(parent_entropy, 4))
print("Weighted child entropy:", round(child_entropy, 4))
print("Information gain:", round(information_gain, 4))

print("\nParent Gini:", round(parent_gini, 4))
print("Weighted child Gini:", round(child_gini, 4))
print("Gini reduction:", round(gini_reduction, 4))


# %% 5. Load Iris and Compare Split Criteria
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names,
)

y = iris.target

models = {
    "Gini": DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=42,
    ),
    "Entropy": DecisionTreeClassifier(
        criterion="entropy",
        max_depth=3,
        random_state=42,
    ),
}

model_results = []

for name, model in models.items():
    model.fit(X, y)

    model_results.append({
        "Criterion": name,
        "Training Accuracy": model.score(X, y),
        "Tree Depth": model.get_depth(),
        "Number of Leaves": model.get_n_leaves(),
        "Number of Nodes": model.tree_.node_count,
        "Root Feature": X.columns[model.tree_.feature[0]],
        "Root Threshold": model.tree_.threshold[0],
    })

results_df = pd.DataFrame(model_results)

print("\nGini vs Entropy on Iris:")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# %% 6. Visualize Impurity for Binary Classification
positive_probability = np.linspace(0.001, 0.999, 300)

gini_values = 2 * positive_probability * (1 - positive_probability)

entropy_values = -(
    positive_probability * np.log2(positive_probability)
    + (1 - positive_probability)
    * np.log2(1 - positive_probability)
)

plt.figure(figsize=(9, 5))

plt.plot(
    positive_probability,
    gini_values,
    label="Gini Impurity",
)

plt.plot(
    positive_probability,
    entropy_values,
    label="Entropy",
)

plt.xlabel("Probability of Class A")
plt.ylabel("Impurity")
plt.title("Gini Impurity vs Entropy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% 7. Visualize Both Learned Trees
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for axis, (name, model) in zip(axes, models.items()):
    from sklearn.tree import plot_tree

    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        ax=axis,
        fontsize=7,
    )

    axis.set_title(f"Decision Tree Using {name}")

plt.tight_layout()
plt.show()
