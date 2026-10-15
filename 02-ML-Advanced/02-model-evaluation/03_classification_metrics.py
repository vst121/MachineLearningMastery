
# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% 2. Load the dataset
data = load_breast_cancer()

X = data.data

# Original labels: 0 = malignant, 1 = benign.
# Redefine the positive class as malignant: 1 = malignant, 0 = benign.
y = (data.target == 0).astype(int)

class_names = ["Benign", "Malignant"]

print(f"Dataset shape: {X.shape}")
print(f"Class counts (0 = benign, 1 = malignant): {np.bincount(y)}")

# %% 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# %% 4. Train a classification model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=5000,
        random_state=42,
    ),
)

model.fit(X_train, y_train)

# %% 5. Generate predictions
y_pred = model.predict(X_test)

# %% 6. Build the confusion matrix
# Rows = actual classes; columns = predicted classes.
tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
).ravel()

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print(f"\nTrue negatives:  {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives:  {tp}")

# %% 7. Calculate metrics manually
total = tn + fp + fn + tp

accuracy = (tp + tn) / total if total else 0.0
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
specificity = tn / (tn + fp) if (tn + fp) else 0.0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall)
    else 0.0
)

print("\nMetrics calculated manually")
print(f"Accuracy:    {accuracy:.3f}")
print(f"Precision:   {precision:.3f}")
print(f"Recall:      {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"F1-score:    {f1:.3f}")

# %% 8. Verify metrics using scikit-learn
print("\nMetrics from scikit-learn")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(
    f"Precision: "
    f"{precision_score(y_test, y_pred, zero_division=0):.3f}"
)
print(
    f"Recall:    "
    f"{recall_score(y_test, y_pred, zero_division=0):.3f}"
)
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")

# %% 9. Display a normalized confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=[0, 1],
    display_labels=class_names,
    normalize="true",
    values_format=".2f",
    cmap="Blues",
)

plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()

# %% 10. Display the classification report
print("\nClassification report")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0,
    )
)

# %% 11. Demonstrate accuracy under class imbalance
# Suppose 990 samples are negative and 10 are positive.
y_imbalanced = np.array([0] * 990 + [1] * 10)

# A naive model predicts every sample as negative.
predictions_all_negative = np.zeros_like(y_imbalanced)

print("\nImbalanced dataset example")
print(
    "Accuracy:",
    accuracy_score(y_imbalanced, predictions_all_negative),
)
print(
    "Positive-class precision:",
    precision_score(
        y_imbalanced,
        predictions_all_negative,
        zero_division=0,
    ),
)
print(
    "Positive-class recall:",
    recall_score(
        y_imbalanced,
        predictions_all_negative,
        zero_division=0,
    ),
)
print(
    "Positive-class F1:",
    f1_score(
        y_imbalanced,
        predictions_all_negative,
        zero_division=0,
    ),
)
