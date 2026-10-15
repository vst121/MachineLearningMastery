
# %% [markdown]
# # K-Nearest Neighbors (KNN) Classification
#
# Objectives:
# - Train a KNN classifier on the Iris dataset.
# - Understand the role of k.
# - Apply feature scaling.
# - Evaluate classification performance.
# - Explore how different values of k affect predictions.

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% [markdown]
# ## 1. Load the Dataset

# %%
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)
print("Class distribution:", np.bincount(y))


# %% [markdown]
# ## 2. Split the Dataset
#
# Stratification preserves approximately the same class proportions
# in the training and test sets.

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
# ## 3. Build the KNN Pipeline
#
# StandardScaler is fitted only on the training data.
# KNN then calculates distances using standardized features.

# %%
k = 5

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


# %% [markdown]
# ## 4. Make Predictions

# %%
y_pred = model.predict(X_test)

print("First 10 actual labels:    ", y_test[:10])
print("First 10 predicted labels: ", y_pred[:10])


# %% [markdown]
# ## 5. Evaluate the Classifier

# %%
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        zero_division=0,
    )
)

cm = confusion_matrix(y_test, y_pred)

print("Confusion matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names,
).plot(cmap="Blues")

plt.title(f"KNN Confusion Matrix (k={k})")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6. Predict a New Flower
#
# The input features must follow the same order as iris.feature_names.

# %%
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

predicted_class = model.predict(new_flower)[0]
probabilities = model.predict_proba(new_flower)[0]

print("Predicted species:", iris.target_names[predicted_class])
print("\nClass probabilities:")

for class_name, probability in zip(
    iris.target_names,
    probabilities,
):
    print(f"{class_name}: {probability:.4f}")


# %% [markdown]
# ## 7. Explore the Effect of k
#
# This experiment is exploratory. We compare training accuracy with
# test accuracy to understand model behavior.
#
# In a proper model-selection workflow, use cross-validation on the
# training data to choose k, then evaluate on the test set once.

# %%
k_values = range(1, 21)

train_accuracies = []
test_accuracies = []

for current_k in k_values:
    experiment_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(n_neighbors=current_k),
            ),
        ]
    )

    experiment_model.fit(X_train, y_train)

    train_pred = experiment_model.predict(X_train)
    test_pred = experiment_model.predict(X_test)

    train_accuracies.append(
        accuracy_score(y_train, train_pred)
    )
    test_accuracies.append(
        accuracy_score(y_test, test_pred)
    )


# %%
plt.figure(figsize=(10, 6))

plt.plot(
    list(k_values),
    train_accuracies,
    marker="o",
    label="Training accuracy",
)

plt.plot(
    list(k_values),
    test_accuracies,
    marker="s",
    label="Test accuracy",
)

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("KNN: Effect of k on Accuracy")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
