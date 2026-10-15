# %% [markdown]
# # KNN: The Impact of Feature Scaling
#
# Objectives:
# - Understand why KNN is sensitive to feature scales.
# - Compare KNN with and without standardization.
# - Examine how scaling changes neighbor distances.
# - Evaluate both approaches using cross-validation and a held-out test set.

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
# ## 1. Load and Inspect the Dataset

# %%
iris = load_iris()

X = iris.data
y = iris.target

feature_summary = pd.DataFrame(
    {
        "Feature": iris.feature_names,
        "Minimum": X.min(axis=0),
        "Maximum": X.max(axis=0),
        "Mean": X.mean(axis=0),
        "Standard Deviation": X.std(axis=0),
    }
)

print(feature_summary.round(3).to_string(index=False))


# %% [markdown]
# ## 2. Split the Dataset
#
# Both experiments use exactly the same split.

# %%
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
# ## 3. Define the Two Approaches
#
# Approach A: KNN without scaling.
# Approach B: StandardScaler followed by KNN.
#
# The scaler in Approach B is fitted only on training data.
# The same k, distance metric, and voting strategy are used.

# %%
k = 5

models = {
    "Without scaling": Pipeline(
        steps=[
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k,
                    weights="uniform",
                    metric="minkowski",
                    p=2,
                ),
            )
        ]
    ),
    "With standardization": Pipeline(
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
    ),
}


# %% [markdown]
# ## 4. Compare Using Cross-Validation
#
# Each pipeline is evaluated within the training set.
# Scaling is fitted independently within each cross-validation fold.

# %%
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_results = []

for approach, model in models.items():
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
            "Approach": approach,
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
# ## 5. Visualize Cross-Validation Results

# %%
ax = cv_results_df.plot(
    x="Approach",
    y=["Mean CV Accuracy", "Mean CV Macro F1"],
    kind="bar",
    figsize=(10, 5),
)

ax.set_title(f"KNN Feature Scaling Comparison (k={k})")
ax.set_ylabel("Mean Cross-Validation Score")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6. Inspect the Feature Scales
#
# Standardization changes the feature representation by centering
# each feature and scaling it to approximately unit variance.

# %%
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

scale_comparison = pd.DataFrame(
    {
        "Feature": iris.feature_names,
        "Original Training Std": X_train.std(axis=0),
        "Standardized Training Std": X_train_scaled.std(axis=0),
    }
)

print(scale_comparison.round(4).to_string(index=False))


# %% [markdown]
# ## 7. Compare Neighbor Distances
#
# Select one test observation and compare its nearest neighbors
# under the original and standardized feature representations.

# %%
sample = X_test[0:1]

scaler_for_inspection = StandardScaler()
X_train_scaled = scaler_for_inspection.fit_transform(X_train)
sample_scaled = scaler_for_inspection.transform(sample)

knn_original = KNeighborsClassifier(n_neighbors=k)
knn_original.fit(X_train, y_train)

knn_scaled = KNeighborsClassifier(n_neighbors=k)
knn_scaled.fit(X_train_scaled, y_train)

distances_original, indices_original = knn_original.kneighbors(sample)
distances_scaled, indices_scaled = knn_scaled.kneighbors(sample_scaled)

print("Actual class:", iris.target_names[y_test[0]])

print("\nNearest neighbors without scaling:")
for rank, (distance, index) in enumerate(
    zip(distances_original[0], indices_original[0]),
    start=1,
):
    print(
        f"Neighbor {rank}: "
        f"distance={distance:.4f}, "
        f"class={iris.target_names[y_train[index]]}"
    )

print("\nNearest neighbors with standardization:")
for rank, (distance, index) in enumerate(
    zip(distances_scaled[0], indices_scaled[0]),
    start=1,
):
    print(
        f"Neighbor {rank}: "
        f"distance={distance:.4f}, "
        f"class={iris.target_names[y_train[index]]}"
    )


# %% [markdown]
# ## 8. Final Test Evaluation
#
# Fit each pipeline on the full training set and evaluate it on
# the held-out test set.

# %%
test_results = []

for approach, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    test_results.append(
        {
            "Approach": approach,
            "Test Accuracy": accuracy,
            "Test Macro F1": macro_f1,
        }
    )

    print(f"\n{approach}")
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

    plt.title(f"Confusion Matrix: {approach}")
    plt.tight_layout()
    plt.show()


# %%
test_results_df = pd.DataFrame(test_results)

print("\nFinal test comparison:")
print(test_results_df.round(4).to_string(index=False))
