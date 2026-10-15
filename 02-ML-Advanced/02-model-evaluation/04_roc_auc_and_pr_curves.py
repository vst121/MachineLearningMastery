# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    precision_recall_curve,
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
# Redefine the positive class: 1 = malignant, 0 = benign.
y = (data.target == 0).astype(int)

print("Dataset shape:", X.shape)
print("Positive class: Malignant")
print("Negative class: Benign")
print("Malignant cases:", np.sum(y == 1))
print("Benign cases:", np.sum(y == 0))
print("Positive class prevalence:", round(np.mean(y), 4))


# %% 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))
print("Training positive prevalence:", round(np.mean(y_train), 4))
print("Test positive prevalence:", round(np.mean(y_test), 4))


# %% 4. Train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42),
)

model.fit(X_train, y_train)

# Obtain the probability of the positive class.
# Identify the class explicitly instead of assuming a column position.
positive_class_index = list(model.classes_).index(1)

y_prob = model.predict_proba(X_test)[:, positive_class_index]

print("\nModel trained successfully.")
print("Model classes:", model.classes_)
print("First five malignant probabilities:", np.round(y_prob[:5], 4))


# %% 5. Calculate ROC-AUC and Average Precision
roc_auc = roc_auc_score(y_test, y_prob)
average_precision = average_precision_score(y_test, y_prob)

print("\nEvaluation metrics")
print("------------------")
print(f"ROC-AUC:           {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")
print(f"Positive prevalence: {np.mean(y_test):.4f}")


# %% 6. Calculate ROC curve points
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)

print("\nFirst five ROC curve points")
print("---------------------------")
for i in range(min(5, len(fpr))):
    print(
        f"Threshold: {roc_thresholds[i]:.4f}, "
        f"FPR: {fpr[i]:.4f}, "
        f"TPR: {tpr[i]:.4f}"
    )


# %% 7. Plot the ROC curve
fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name=f"Logistic Regression (AUC = {roc_auc:.3f})",
    ax=ax,
)

# Random ranking reference line.
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random ranking (AUC = 0.5)",
)

ax.set_title("ROC Curve: Malignant Tumor Detection")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %% 8. Calculate Precision-Recall curve points
precision, recall, pr_thresholds = precision_recall_curve(
    y_test,
    y_prob,
)

baseline_precision = np.mean(y_test)

print("\nFirst five Precision-Recall curve points")
print("----------------------------------------")
for i in range(min(5, len(precision))):
    print(
        f"Recall: {recall[i]:.4f}, "
        f"Precision: {precision[i]:.4f}"
    )


# %% 9. Plot the Precision-Recall curve
fig, ax = plt.subplots(figsize=(8, 6))

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_prob,
    name=f"Logistic Regression (AP = {average_precision:.3f})",
    ax=ax,
)

# Baseline precision for a classifier with no ranking ability.
ax.axhline(
    y=baseline_precision,
    linestyle="--",
    label=f"Prevalence baseline ({baseline_precision:.3f})",
)

ax.set_title("Precision-Recall Curve: Malignant Tumor Detection")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %% 10. Compare different classification thresholds
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

print("\nThreshold comparison")
print("-" * 70)
print(
    f"{'Threshold':>10} "
    f"{'Precision':>12} "
    f"{'Recall':>12} "
    f"{'F1 Score':>12}"
)

from sklearn.metrics import f1_score, precision_score, recall_score

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    precision_value = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall_value = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1_value = f1_score(
        y_test, y_pred, zero_division=0
    )

    print(
        f"{threshold:>10.2f} "
        f"{precision_value:>12.4f} "
        f"{recall_value:>12.4f} "
        f"{f1_value:>12.4f}"
    )


# %% 11. Plot precision and recall against the threshold
# precision_recall_curve returns one more precision/recall
# point than the number of thresholds.
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    pr_thresholds,
    precision[:-1],
    label="Precision",
)

ax.plot(
    pr_thresholds,
    recall[:-1],
    label="Recall",
)

ax.set_title("Precision and Recall Across Thresholds")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
