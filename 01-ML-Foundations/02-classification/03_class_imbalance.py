"""
Machine Learning Foundations
Topic: Class Imbalance

Learning objectives:
- Understand balanced and imbalanced datasets.
- See why accuracy can be misleading.
- Compare accuracy, precision, recall, and F1-score.
- Establish a majority-class baseline.
- Visualize class distribution and the confusion matrix.
"""

# %%
# 1. Imports

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# %%
# 2. Create an imbalanced dataset

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    weights=[0.95, 0.05],
    random_state=42,
)

print("Dataset shape:", X.shape)

class_counts = pd.Series(y).value_counts().sort_index()

print("\nClass distribution:")
for class_id, count in class_counts.items():
    print(f"Class {class_id}: {count} observations ({count / len(y):.1%})")


# %%
# 3. Visualize the class distribution

plt.figure(figsize=(7, 5))

plt.bar(
    ["Majority class (0)", "Minority class (1)"],
    [class_counts.get(0, 0), class_counts.get(1, 0)],
)

plt.title("Class Distribution")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.show()


# %%
# 4. Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training observations: {len(X_train)}")
print(f"Test observations: {len(X_test)}")

print("\nTest-set class distribution:")
print(pd.Series(y_test).value_counts().sort_index())


# %%
# 5. Train a majority-class baseline

# This classifier always predicts the most frequent class
# found in the training data.
model = DummyClassifier(strategy="most_frequent")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Majority-class baseline trained.")
print("Predicted classes:", sorted(set(y_pred)))


# %%
# 6. Evaluate accuracy

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")

# The classifier may achieve high accuracy by predicting class 0
# for every observation, despite never detecting class 1.


# %%
# 7. Examine the confusion matrix

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Majority (0)", "Minority (1)"],
).plot(cmap="Blues")

plt.title("Majority-Class Baseline")
plt.tight_layout()
plt.show()


# %%
# 8. Examine precision, recall, and F1-score

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Majority class", "Minority class"],
        zero_division=0,
    )
)

# Pay special attention to the minority class:
# its recall is zero because the classifier never predicts it.


# %%
# 9. Compare with a simple learned classifier

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42,
)

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

print("\nDecision tree results:")
print(f"Accuracy: {accuracy_score(y_test, tree_predictions):.2%}")

print(
    classification_report(
        y_test,
        tree_predictions,
        labels=[0, 1],
        target_names=["Majority class", "Minority class"],
        zero_division=0,
    )
)
