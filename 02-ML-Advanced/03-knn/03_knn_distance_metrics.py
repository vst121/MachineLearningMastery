# %% [markdown]
# # KNN Distance Metrics
#
# Objectives:
# - Understand Euclidean and Manhattan distance.
# - Compare distance metrics under consistent conditions.
# - Evaluate performance using stratified cross-validation.
# - Compare cross-validation results with held-out test performance.

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
# ## 2. Define the Distance Metrics
#
# Minkowski distance is a general distance function:
#
# p=1: Manhattan distance
# p=2: Euclidean distance
#
# Both models use the same k and uniform voting so that
# the distance metric is the main variable being investigated.

# %%
k = 5

distance_metrics = {
    "Manhattan": 1,
    "Euclidean": 2,
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# %% [markdown]
# ## 3. Compare with Cross-Validation
#
# The pipeline fits the scaler independently within each fold.
# Macro F1 gives each Iris species equal weight.

# %%
results = []

for metric_name, p_value in distance_metrics.items():
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k,
                    weights="uniform",
                    metric="minkowski",
                    p=p_value,
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

    results.append(
        {
            "Distance": metric_name,
            "Mean CV Accuracy": scores["test_accuracy"].mean(),
            "Std CV Accuracy": scores["test_accuracy"].std(),
            "Mean CV Macro F1": scores["test_f1_macro"].mean(),
            "Std CV Macro F1": scores["test_f1_macro"].std(),
            "Mean Training Accuracy": scores["train_accuracy"].mean(),
        }
    )

results_df = pd.DataFrame(results)

print(results_df.round(4).to_string(index=False))


# %% [markdown]
# ## 4. Visualize Cross-Validation Performance

# %%
ax = results_df.plot(
    x="Distance",
    y=["Mean CV Accuracy", "Mean CV Macro F1"],
    kind="bar",
    figsize=(9, 5),
)

ax.set_title(f"KNN Distance Comparison (k={k})")
ax.set_ylabel("Mean Cross-Validation Score")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 5. Fit and Evaluate Each Model on the Test Set
#
# Cross-validation is used to understand performance on training data.
# The test set is kept separate for final evaluation.

# %%
test_results = []

for metric_name, p_value in distance_metrics.items():
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k,
                    weights="uniform",
                    metric="minkowski",
                    p=p_value,
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
            "Distance": metric_name,
            "Test Accuracy": accuracy,
            "Test Macro F1": macro_f1,
        }
    )

    print(f"\n{metric_name} distance")
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

    plt.title(f"{metric_name} Distance: Test Confusion Matrix")
    plt.tight_layout()
    plt.show()


# %%
test_results_df = pd.DataFrame(test_results)

print("\nFinal test comparison:")
print(test_results_df.round(4).to_string(index=False))
