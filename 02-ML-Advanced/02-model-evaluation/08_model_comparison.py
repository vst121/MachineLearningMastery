# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# %% 2. Load the dataset
data = load_breast_cancer()

X = data.data

# Define malignant tumors as the positive class.
# Original labels: 0 = malignant, 1 = benign.
y = (data.target == 0).astype(int)

print("Dataset shape:", X.shape)
print("Positive class: Malignant")
print("Negative class: Benign")
print("Class distribution:", np.bincount(y))


# %% 3. Create development and test sets
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\nDevelopment samples:", len(X_dev))
print("Test samples:", len(X_test))


# %% 4. Define Logistic Regression configurations
# Smaller C means stronger regularization.
# Larger C means weaker regularization.
configurations = [
    {"name": "C=0.01", "C": 0.01},
    {"name": "C=0.1", "C": 0.1},
    {"name": "C=1", "C": 1.0},
    {"name": "C=10", "C": 10.0},
    {"name": "C=100", "C": 100.0},
]

def create_model(C):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            max_iter=5000,
            random_state=42,
        ),
    )


# %% 5. Evaluate configurations using cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

cv_results = []

for configuration in configurations:
    model = create_model(configuration["C"])

    scores = cross_validate(
        estimator=model,
        X=X_dev,
        y=y_dev,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
    )

    result = {
        "Configuration": configuration["name"],
        "C": configuration["C"],
        "CV Accuracy": scores["test_accuracy"].mean(),
        "CV Precision": scores["test_precision"].mean(),
        "CV Recall": scores["test_recall"].mean(),
        "CV F1": scores["test_f1"].mean(),
        "CV ROC-AUC": scores["test_roc_auc"].mean(),
        "CV AP": scores["test_average_precision"].mean(),
        "CV Train Accuracy": scores["train_accuracy"].mean(),
        "CV Accuracy Std": scores["test_accuracy"].std(),
    }

    cv_results.append(result)


# %% 6. Display the cross-validation comparison
results_df = pd.DataFrame(cv_results)

print("\nCross-validation results")
print("=" * 100)

print(
    results_df.round(4).to_string(index=False)
)


# %% 7. Select a configuration using a predefined metric
# Here, F1 is the selection metric for demonstration.
# In a real application, choose the metric based on the cost
# of false positives and false negatives.
selection_metric = "CV F1"

best_index = results_df[selection_metric].idxmax()
best_result = results_df.loc[best_index]

best_C = float(best_result["C"])

print("\nSelected configuration")
print("-" * 35)
print("Selection metric:", selection_metric)
print("Selected configuration:", best_result["Configuration"])
print("Selected C:", best_C)
print(f"Mean CV F1: {best_result['CV F1']:.4f}")


# %% 8. Visualize cross-validation results
metrics_to_plot = [
    "CV Accuracy",
    "CV Precision",
    "CV Recall",
    "CV F1",
    "CV ROC-AUC",
]

fig, ax = plt.subplots(figsize=(11, 6))

for metric in metrics_to_plot:
    ax.plot(
        results_df["Configuration"],
        results_df[metric],
        marker="o",
        label=metric,
    )

ax.set_xlabel("Logistic Regression Configuration")
ax.set_ylabel("Mean Cross-Validation Score")
ax.set_title("Model Comparison Across Evaluation Metrics")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %% 9. Compare training and validation performance
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    results_df["Configuration"],
    results_df["CV Train Accuracy"],
    marker="o",
    label="Training accuracy",
)

ax.errorbar(
    results_df["Configuration"],
    results_df["CV Accuracy"],
    yerr=results_df["CV Accuracy Std"],
    marker="s",
    capsize=4,
    label="Validation accuracy",
)

ax.set_xlabel("Logistic Regression Configuration")
ax.set_ylabel("Accuracy")
ax.set_title("Training vs Cross-Validation Performance")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %% 10. Fit the selected configuration on all development data
final_model = create_model(best_C)

final_model.fit(X_dev, y_dev)

test_predictions = final_model.predict(X_test)
test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_metrics = {
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "F1": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(y_test, test_probabilities),
    "Average Precision": average_precision_score(
        y_test, test_probabilities
    ),
}

print("\nFinal test evaluation")
print("=" * 35)

for metric, value in test_metrics.items():
    print(f"{metric:<20}: {value:.4f}")


# %% 11. Compare selected CV performance with test performance
print("\nCross-validation vs test")
print("-" * 45)
print(f"{'Metric':<22}{'CV Mean':>12}{'Test':>12}")

metric_mapping = {
    "Accuracy": ("CV Accuracy", "Accuracy"),
    "Precision": ("CV Precision", "Precision"),
    "Recall": ("CV Recall", "Recall"),
    "F1": ("CV F1", "F1"),
    "ROC-AUC": ("CV ROC-AUC", "ROC-AUC"),
    "Average Precision": ("CV AP", "Average Precision"),
}

for metric, (cv_metric, test_metric) in metric_mapping.items():
    print(
        f"{metric:<22}"
        f"{best_result[cv_metric]:>12.4f}"
        f"{test_metrics[test_metric]:>12.4f}"
    )


# %% 12. Final interpretation
print("\nInterpretation")
print("-" * 35)
print("The configuration was selected using cross-validation.")
print("The test set was reserved for final evaluation.")
print("The selected model is not necessarily optimal for every use case.")
print("Threshold selection and business costs may change the final choice.")
