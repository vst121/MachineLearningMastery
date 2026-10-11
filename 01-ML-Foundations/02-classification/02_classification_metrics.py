"""
Machine Learning Foundations
Topic: Classification Metrics

Dataset:
Scikit-learn Iris dataset.

Learning objectives:
- Understand accuracy and the confusion matrix.
- Understand precision, recall, and F1-score.
- Read a classification report.
- Understand why accuracy alone may not be sufficient.
"""

# %%
# 1. Imports

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt


# %%
# 2. Prepare the dataset

dataset = load_iris(as_frame=True)
df = dataset.frame

# Keep only Setosa and Versicolor for binary classification.
df = df[df["target"].isin([0, 1])].copy()

feature_names = ["petal length (cm)", "petal width (cm)"]

X = df[feature_names]
y = df["target"]

target_names = dataset.target_names[:2]

print("Dataset preview:")
print(df[feature_names + ["target"]].head())

print("\nClasses:")
for class_id, class_name in enumerate(target_names):
    print(f"{class_id} = {class_name}")

print(f"\nNumber of observations: {len(df)}")


# %%
# 3. Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training observations: {len(X_train)}")
print(f"Test observations: {len(X_test)}")


# %%
# 4. Train the SVM classifier

model = SVC(kernel="linear")
model.fit(X_train, y_train)

print("SVM model trained successfully.")


# %%
# 5. Make predictions

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred,
})

results["Actual species"] = results["Actual"].map(
    dict(enumerate(target_names))
)

results["Predicted species"] = results["Predicted"].map(
    dict(enumerate(target_names))
)

print("Prediction results:")
print(results.head(10))


# %%
# 6. Accuracy

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2%}")

# Accuracy is the proportion of all predictions that are correct.


# %%
# 7. Confusion matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion matrix:")
print(cm)

# Rows represent actual classes.
# Columns represent predicted classes.
#
# For binary classification, with labels [0, 1]:
# [[TN, FP],
#  [FN, TP]]
#
# TN = True Negatives
# FP = False Positives
# FN = False Negatives
# TP = True Positives

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names,
).plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# %%
# 8. Precision, recall, and F1-score

# We explicitly choose class 1 (Versicolor) as the positive class.
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print(f"Precision (Versicolor): {precision:.2%}")
print(f"Recall (Versicolor):    {recall:.2%}")
print(f"F1-score (Versicolor):  {f1:.2%}")

# Precision:
# Of all observations predicted as Versicolor, how many were Versicolor?
#
# Recall:
# Of all actual Versicolor observations, how many did we identify?
#
# F1-score:
# The harmonic mean of precision and recall.


# %%
# 9. Classification report

report = classification_report(
    y_test,
    y_pred,
    target_names=target_names,
    zero_division=0,
)

print("\nClassification report:")
print(report)
