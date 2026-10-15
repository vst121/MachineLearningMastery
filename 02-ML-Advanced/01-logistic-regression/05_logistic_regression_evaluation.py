# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Class counts (0 = benign, 1 = malignant): {np.bincount(y)}")

# %% 3. Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

# %% 4. Train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=1.0,
        max_iter=5000,
        random_state=42,
    ),
)

model.fit(X_train, y_train)

# %% 5. Evaluate predictions on the test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
).ravel()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
specificity = tn / (tn + fp) if (tn + fp) else 0.0
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)
probability_log_loss = log_loss(y_test, model.predict_proba(X_test))

print("\nTest-set metrics")
print(f"Accuracy:          {accuracy:.3f}")
print(f"Precision:         {precision:.3f}")
print(f"Recall:            {recall:.3f}")
print(f"Specificity:       {specificity:.3f}")
print(f"F1-score:          {f1:.3f}")
print(f"ROC-AUC:           {roc_auc:.3f}")
print(f"Average precision: {average_precision:.3f}")
print(f"Log loss:          {probability_log_loss:.3f}")

print("\nConfusion matrix (0 = benign, 1 = malignant):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        zero_division=0,
    )
)

# %% 6. Visualize the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=[0, 1],
    display_labels=["Benign", "Malignant"],
    cmap="Blues",
    values_format="d",
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# %% 7. Plot the ROC curve
false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
    y_test,
    y_proba,
)

plt.figure(figsize=(7, 6))
plt.plot(
    false_positive_rate,
    true_positive_rate,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})",
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Random ranking")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% 8. Plot the precision-recall curve
precisions, recalls, pr_thresholds = precision_recall_curve(
    y_test,
    y_proba,
)

plt.figure(figsize=(7, 6))
plt.plot(
    recalls,
    precisions,
    label=f"Average precision = {average_precision:.3f}",
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% 9. Inspect threshold-dependent performance
thresholds = np.arange(0.1, 0.91, 0.05)
threshold_precision = []
threshold_recall = []
threshold_f1 = []

for threshold in thresholds:
    predictions = (y_proba >= threshold).astype(int)

    threshold_precision.append(
        precision_score(y_test, predictions, zero_division=0)
    )
    threshold_recall.append(
        recall_score(y_test, predictions, zero_division=0)
    )
    threshold_f1.append(
        f1_score(y_test, predictions, zero_division=0)
    )

plt.figure(figsize=(9, 5))
plt.plot(thresholds, threshold_precision, marker="o", label="Precision")
plt.plot(thresholds, threshold_recall, marker="s", label="Recall")
plt.plot(thresholds, threshold_f1, marker="^", label="F1-score")

plt.xlabel("Classification threshold")
plt.ylabel("Score")
plt.title("Metrics vs. Classification Threshold")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% 10. Inspect probability calibration
# This is a diagnostic plot, not a recalibration step.
CalibrationDisplay.from_estimator(
    model,
    X_test,
    y_test,
    n_bins=8,
    strategy="uniform",
)

plt.title("Probability Calibration")
plt.tight_layout()
plt.show()
