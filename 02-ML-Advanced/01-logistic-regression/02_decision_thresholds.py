# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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

print(f"Training samples:   {len(y_train)}")
print(f"Validation samples: {len(y_val)}")
print(f"Test samples:       {len(y_test)}")

# %% 4. Train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42),
)

model.fit(X_train, y_train)

# Find the probability column corresponding to malignant (class 1).
positive_class_index = list(model.classes_).index(1)

val_probabilities = model.predict_proba(X_val)[:, positive_class_index]

# %% 5. Compare thresholds on the validation set
thresholds_to_compare = [0.3, 0.5, 0.7]

print("\nValidation-set threshold comparison")
print("Threshold | Precision | Recall | F1-score")
print("-" * 43)

for threshold in thresholds_to_compare:
    predictions = (val_probabilities >= threshold).astype(int)

    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    print(
        f"{threshold:9.2f} | {precision:9.3f} | "
        f"{recall:6.3f} | {f1:8.3f}"
    )

# %% 6. Find the threshold with the best validation F1-score
thresholds = np.arange(0.1, 0.91, 0.01)
validation_f1_scores = []

for threshold in thresholds:
    predictions = (val_probabilities >= threshold).astype(int)
    score = f1_score(y_val, predictions, zero_division=0)
    validation_f1_scores.append(score)

best_index = int(np.argmax(validation_f1_scores))
best_threshold = float(thresholds[best_index])

print(f"\nBest validation threshold: {best_threshold:.2f}")
print(f"Best validation F1-score: {validation_f1_scores[best_index]:.3f}")

# %% 7. Plot precision and recall across thresholds
validation_precision = []
validation_recall = []

for threshold in thresholds:
    predictions = (val_probabilities >= threshold).astype(int)

    validation_precision.append(
        precision_score(y_val, predictions, zero_division=0)
    )
    validation_recall.append(
        recall_score(y_val, predictions, zero_division=0)
    )

plt.figure(figsize=(9, 5))
plt.plot(thresholds, validation_precision, label="Precision")
plt.plot(thresholds, validation_recall, label="Recall")

plt.axvline(
    best_threshold,
    color="black",
    linestyle="--",
    label=f"Best F1 threshold = {best_threshold:.2f}",
)

plt.xlabel("Classification threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs. Classification Threshold")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% 8. Evaluate once on the test set
test_probabilities = model.predict_proba(X_test)[:, positive_class_index]
test_predictions = (test_probabilities >= best_threshold).astype(int)

print("\nTest-set results")
print(f"Threshold: {best_threshold:.2f}")
print(f"Accuracy:  {accuracy_score(y_test, test_predictions):.3f}")
print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, test_predictions, zero_division=0):.3f}")
print(f"F1-score:  {f1_score(y_test, test_predictions, zero_division=0):.3f}")

print("\nConfusion matrix (0 = benign, 1 = malignant):")
print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        zero_division=0,
    )
)
