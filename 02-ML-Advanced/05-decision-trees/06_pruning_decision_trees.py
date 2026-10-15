# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier


# %% 2. Load the Dataset
cancer = load_breast_cancer()

X = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names,
)

y = pd.Series(
    cancer.target,
    name="target",
)

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])


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


# %% 4. Train an Unpruned Tree and Calculate the Pruning Path
initial_tree = DecisionTreeClassifier(
    random_state=42,
)

initial_tree.fit(X_train, y_train)

path = initial_tree.cost_complexity_pruning_path(
    X_train,
    y_train,
)

ccp_alphas = path.ccp_alphas
impurities = path.impurities

print("\nNumber of pruning candidates:", len(ccp_alphas))
print("First alpha values:", ccp_alphas[:10])
print("Last alpha value:", ccp_alphas[-1])


# %% 5. Visualize the Pruning Path
plt.figure(figsize=(9, 5))

plt.plot(
    ccp_alphas,
    impurities,
    marker="o",
    drawstyle="steps-post",
)

plt.xlabel("Cost-Complexity Alpha")
plt.ylabel("Total Leaf Impurity")
plt.title("Cost-Complexity Pruning Path")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% 6. Train Trees for Different Alpha Values
# Exclude the final alpha because it typically produces a root-only tree.
candidate_alphas = np.unique(
    np.quantile(
        ccp_alphas,
        np.linspace(0, 1, 30),
    )
)

candidate_alphas = candidate_alphas[
    candidate_alphas < ccp_alphas[-1]
]

candidate_alphas = np.unique(
    np.concatenate(([0.0], candidate_alphas))
)

tree_results = []
trees = []

for alpha in candidate_alphas:
    model = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=alpha,
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    trees.append(model)

    tree_results.append({
        "CCP Alpha": alpha,
        "Depth": model.get_depth(),
        "Leaves": model.get_n_leaves(),
        "Train Accuracy": accuracy_score(
            y_train,
            train_predictions,
        ),
        "Test Accuracy": accuracy_score(
            y_test,
            test_predictions,
        ),
        "Train F1": f1_score(
            y_train,
            train_predictions,
        ),
        "Test F1": f1_score(
            y_test,
            test_predictions,
        ),
    })

results_df = pd.DataFrame(tree_results)

print("\nPruning Results:")
print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 7. Visualize Tree Complexity
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(
    results_df["CCP Alpha"],
    results_df["Leaves"],
    marker="o",
)

axes[0].set_xlabel("Cost-Complexity Alpha")
axes[0].set_ylabel("Number of Leaves")
axes[0].set_title("Pruning Alpha vs Tree Size")
axes[0].grid(alpha=0.3)

axes[1].plot(
    results_df["CCP Alpha"],
    results_df["Depth"],
    marker="o",
)

axes[1].set_xlabel("Cost-Complexity Alpha")
axes[1].set_ylabel("Tree Depth")
axes[1].set_title("Pruning Alpha vs Tree Depth")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %% 8. Compare Train and Test Performance
plt.figure(figsize=(10, 5))

plt.plot(
    results_df["CCP Alpha"],
    results_df["Train F1"],
    marker="o",
    label="Training F1",
)

plt.plot(
    results_df["CCP Alpha"],
    results_df["Test F1"],
    marker="o",
    label="Test F1 (Exploratory Only)",
)

plt.xlabel("Cost-Complexity Alpha")
plt.ylabel("F1 Score")
plt.title("Performance Across Pruning Levels")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% 9. Select Alpha Using Cross-Validation
# The test set must not be used to choose the pruning parameter.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

search = GridSearchCV(
    estimator=DecisionTreeClassifier(
        random_state=42,
    ),
    param_grid={
        "ccp_alpha": candidate_alphas,
    },
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest CCP Alpha:", search.best_params_["ccp_alpha"])
print("Best Mean CV F1:", round(search.best_score_, 4))
print("Selected Tree Depth:", best_model.get_depth())
print("Selected Number of Leaves:", best_model.get_n_leaves())


# %% 10. Final Test Evaluation
test_predictions = best_model.predict(X_test)

print("\nFinal Test Accuracy:", round(
    accuracy_score(y_test, test_predictions),
    4,
))

print("Final Test F1:", round(
    f1_score(y_test, test_predictions),
    4,
))


# %% 11. Visualize the Selected Pruned Tree
plt.figure(figsize=(22, 10))

from sklearn.tree import plot_tree

plot_tree(
    best_model,
    feature_names=X.columns,
    class_names=cancer.target_names,
    filled=True,
    rounded=True,
    fontsize=7,
)

plt.title(
    f"Selected Pruned Decision Tree "
    f"(alpha={search.best_params_['ccp_alpha']:.6f})"
)

plt.tight_layout()
plt.show()
