# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
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
print("\nClass distribution:")
print(y.value_counts().rename(index={
    0: "malignant",
    1: "benign",
}))


# %% 3. Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

X_train_sub, X_validation, y_train_sub, y_validation = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    stratify=y_train,
    random_state=42,
)

print("\nTraining samples:", len(X_train_sub))
print("Validation samples:", len(X_validation))
print("Test samples:", len(X_test))


# %% 4. Experiment with Maximum Tree Depth
depth_values = [1, 2, 3, 4, 5, 7, 10, 15, None]

depth_results = []
depth_models = {}

for depth in depth_values:
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42,
    )

    model.fit(X_train_sub, y_train_sub)

    train_predictions = model.predict(X_train_sub)
    validation_predictions = model.predict(X_validation)

    depth_models[depth] = model

    depth_results.append({
        "Max Depth": str(depth),
        "Actual Depth": model.get_depth(),
        "Leaves": model.get_n_leaves(),
        "Train Accuracy": accuracy_score(
            y_train_sub,
            train_predictions,
        ),
        "Validation Accuracy": accuracy_score(
            y_validation,
            validation_predictions,
        ),
        "Train F1": f1_score(
            y_train_sub,
            train_predictions,
        ),
        "Validation F1": f1_score(
            y_validation,
            validation_predictions,
        ),
    })

depth_df = pd.DataFrame(depth_results)

print("\nEffect of Maximum Depth:")
print(
    depth_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 5. Plot Accuracy vs Tree Depth
x_positions = np.arange(len(depth_df))

plt.figure(figsize=(11, 5))

plt.plot(
    x_positions,
    depth_df["Train Accuracy"],
    marker="o",
    label="Training Accuracy",
)

plt.plot(
    x_positions,
    depth_df["Validation Accuracy"],
    marker="o",
    label="Validation Accuracy",
)

plt.xticks(
    x_positions,
    depth_df["Max Depth"],
)

plt.xlabel("Maximum Depth (None = Unrestricted)")
plt.ylabel("Accuracy")
plt.title("Decision Tree Complexity and Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% 6. Plot F1 vs Tree Depth
plt.figure(figsize=(11, 5))

plt.plot(
    x_positions,
    depth_df["Train F1"],
    marker="o",
    label="Training F1",
)

plt.plot(
    x_positions,
    depth_df["Validation F1"],
    marker="o",
    label="Validation F1",
)

plt.xticks(
    x_positions,
    depth_df["Max Depth"],
)

plt.xlabel("Maximum Depth (None = Unrestricted)")
plt.ylabel("F1 Score")
plt.title("Decision Tree Complexity and F1 Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% 7. Experiment with Minimum Leaf Size
leaf_sizes = [1, 2, 5, 10, 20, 30]

leaf_results = []

for leaf_size in leaf_sizes:
    model = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=leaf_size,
        random_state=42,
    )

    model.fit(X_train_sub, y_train_sub)

    train_predictions = model.predict(X_train_sub)
    validation_predictions = model.predict(X_validation)

    leaf_results.append({
        "Min Samples per Leaf": leaf_size,
        "Actual Depth": model.get_depth(),
        "Leaves": model.get_n_leaves(),
        "Train Accuracy": accuracy_score(
            y_train_sub,
            train_predictions,
        ),
        "Validation Accuracy": accuracy_score(
            y_validation,
            validation_predictions,
        ),
        "Validation F1": f1_score(
            y_validation,
            validation_predictions,
        ),
    })

leaf_df = pd.DataFrame(leaf_results)

print("\nEffect of Minimum Leaf Size:")
print(
    leaf_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 8. Plot the Effect of Minimum Leaf Size
plt.figure(figsize=(10, 5))

plt.plot(
    leaf_df["Min Samples per Leaf"],
    leaf_df["Train Accuracy"],
    marker="o",
    label="Training Accuracy",
)

plt.plot(
    leaf_df["Min Samples per Leaf"],
    leaf_df["Validation Accuracy"],
    marker="o",
    label="Validation Accuracy",
)

plt.xlabel("Minimum Samples per Leaf")
plt.ylabel("Accuracy")
plt.title("Minimum Leaf Size and Generalization")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% 9. Select a Configuration Using Cross-Validation
# Use training data only. The validation set above was for exploration.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

candidate_depths = [2, 3, 4, 5, 7, 10, None]
cv_results = []

for depth in candidate_depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42,
    )

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1": "f1",
        },
        return_train_score=True,
        n_jobs=-1,
    )

    cv_results.append({
        "Max Depth": str(depth),
        "Mean CV Accuracy": scores["test_accuracy"].mean(),
        "Std CV Accuracy": scores["test_accuracy"].std(),
        "Mean CV F1": scores["test_f1"].mean(),
        "Mean Train Accuracy": scores["train_accuracy"].mean(),
    })

cv_df = pd.DataFrame(cv_results)

print("\nCross-Validation Results:")
print(
    cv_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

best_row = cv_df.loc[cv_df["Mean CV F1"].idxmax()]
best_depth_label = best_row["Max Depth"]

best_depth = (
    None
    if best_depth_label == "None"
    else int(best_depth_label)
)

print("\nSelected Maximum Depth:", best_depth)


# %% 10. Final Test Evaluation
# Refit the selected configuration on all training data.
final_model = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=42,
)

final_model.fit(X_train, y_train)

test_predictions = final_model.predict(X_test)

print("\nFinal Test Accuracy:", round(
    accuracy_score(y_test, test_predictions),
    4,
))

print("Final Test F1:", round(
    f1_score(y_test, test_predictions),
    4,
))

print("Final Tree Depth:", final_model.get_depth())
print("Final Number of Leaves:", final_model.get_n_leaves())
