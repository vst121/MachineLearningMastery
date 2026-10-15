# %% [markdown]
# # KNN: Uniform vs Distance-Weighted Voting
#
# Objectives:
# - Understand how KNN combines neighboring labels.
# - Compare uniform and distance-weighted voting.
# - Evaluate both strategies using stratified cross-validation.
# - Compare cross-validation performance with final test performance.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# ## 1. Load and Split the Dataset

# %%
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


# %% [markdown]
# ## 2. Understand the Voting Strategies
#
# Uniform voting:
# Every neighbor contributes equally to the prediction.
#
# Distance-weighted voting:
# Closer neighbors receive greater influence.
#
# We use the same k and Euclidean distance for both strategies.

# %%
k = 5

weighting_strategies = ["uniform", "distance"]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# %% [markdown]
# ## 3. Compare Using Cross-Validation

# %%
cv_results = []

for weighting in weighting_strategies:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weighting,
                    metric="minkowski",
                    p=2,
                ),
            ),
        ]
    )

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
        },
        return_train_score=True,
        n_jobs=-1,
    )

    cv_results.append(
        {
            "Weighting": weighting,
            "Mean CV Accuracy": scores["test_accuracy"].mean(),
            "Std CV Accuracy": scores["test_accuracy"].std(),
            "Mean CV Macro F1": scores["test_f1_macro"].mean(),
            "Std CV Macro F1": scores["test_f1_macro"].std(),
            "Mean Training Accuracy": scores["train_accuracy"].mean(),
        }
    )

cv_results_df = pd.DataFrame(cv_results)

print("Cross-validation results:")
print(cv_results_df.round(4).to_string(index=False))


# %% [markdown]
# ## 4. Visualize Cross-Validation Results

# %%
ax = cv_results_df.plot(
    x="Weighting",
    y=["Mean CV Accuracy", "Mean CV Macro F1"],
    kind="bar",
    figsize=(9, 5),
)

ax.set_title(f"KNN Voting Strategy Comparison (k={k})")
ax.set_ylabel("Mean Cross-Validation Score")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 5. Fit and Evaluate on the Test Set
#
# The test set is used for final evaluation, not for choosing
# the better voting strategy.

# %%
test_results = []

for weighting in weighting_strategies:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weighting,
                    metric="minkowski",
                    p=2,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    test_results.append(
        {
            "Weighting": weighting,
            "Test Accuracy": accuracy,
            "Test Macro F1": macro_f1,
        }
    )

    print(f"\nVoting strategy: {weighting}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=iris.target_names,
            zero_division=0,
        )
    )

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred),
        display_labels=iris.target_names,
    ).plot(cmap="Blues")

    plt.title(f"KNN Confusion Matrix: {weighting} voting")
    plt.tight_layout()
    plt.show()


# %%
test_results_df = pd.DataFrame(test_results)

print("\nFinal test comparison:")
print(test_results_df.round(4).to_string(index=False))


# %% [markdown]
# ## 6. Inspect Neighbor Weights
#
# Distance weighting gives closer neighbors more influence.
# This example examines the distances and labels of the neighbors
# for one test observation.

# %%
example_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=k,
                weights="distance",
                metric="minkowski",
                p=2,
            ),
        ),
    ]
)

example_model.fit(X_train, y_train)

sample = X_test[0:1]

scaled_sample = example_model.named_steps["scaler"].transform(sample)
scaled_training_data = example_model.named_steps["scaler"].transform(
    X_train
)

distances, indices = example_model.named_steps["knn"].kneighbors(
    scaled_sample
)

neighbor_labels = y_train[indices[0]]

print("Actual class:", iris.target_names[y_test[0]])
print("Predicted class:", iris.target_names[example_model.predict(sample)[0]])
print("\nNeighbor details:")

for rank, (distance, index, label) in enumerate(
    zip(distances[0], indices[0], neighbor_labels),
    start=1,
):
    print(
        f"Neighbor {rank}: "
        f"distance={distance:.4f}, "
        f"class={iris.target_names[label]}"
    )
