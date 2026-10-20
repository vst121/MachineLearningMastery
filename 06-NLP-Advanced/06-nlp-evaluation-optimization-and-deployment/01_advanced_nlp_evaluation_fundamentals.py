"""
Advanced NLP Evaluation Fundamentals

Experiments:
- Compare accuracy, precision, recall, and F1-score.
- Evaluate class-level performance using macro and weighted averages.
- Compare micro and macro averaging for imbalanced datasets.
- Analyze confusion matrices and per-class errors.
- Select a classification threshold using validation data.
- Compare default and threshold-adjusted predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationResult:
    """Store classification evaluation metrics."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    micro_f1: float


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> EvaluationResult:
    """Calculate complementary classification metrics."""
    return EvaluationResult(
        accuracy=accuracy_score(y_true, y_pred),
        macro_precision=precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        macro_recall=recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        macro_f1=f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        weighted_f1=f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        micro_f1=f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
    )


def print_metrics(
    title: str,
    result: EvaluationResult,
) -> None:
    """Display metrics in a consistent format."""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Accuracy:       {result.accuracy:.3f}")
    print(f"Macro precision:{result.macro_precision:>7.3f}")
    print(f"Macro recall:   {result.macro_recall:>7.3f}")
    print(f"Macro F1:       {result.macro_f1:>7.3f}")
    print(f"Weighted F1:    {result.weighted_f1:>7.3f}")
    print(f"Micro F1:       {result.micro_f1:>7.3f}")


def inspect_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> None:
    """Display the confusion matrix and class-level report."""
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
    )

    print("\nConfusion Matrix")
    print("Rows = actual classes; columns = predicted classes")
    print(matrix)

    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
        )
    )


def select_threshold(
    y_true: np.ndarray,
    positive_scores: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    """Select the threshold with the highest validation F1-score."""
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (positive_scores >= threshold).astype(int)
        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def threshold_experiment() -> None:
    """Compare default and validation-selected binary thresholds."""
    # Synthetic validation data: scores represent model confidence
    # for the positive class, not necessarily calibrated probabilities.
    y_validation = np.array(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    )
    positive_scores = np.array(
        [0.08, 0.21, 0.32, 0.41, 0.56, 0.63,
         0.37, 0.52, 0.67, 0.74, 0.82, 0.93]
    )

    thresholds = np.arange(0.1, 0.91, 0.05)
    best_threshold, best_f1 = select_threshold(
        y_validation,
        positive_scores,
        thresholds,
    )

    default_predictions = (positive_scores >= 0.5).astype(int)
    adjusted_predictions = (
        positive_scores >= best_threshold
    ).astype(int)

    print("\nThreshold Selection Experiment")
    print("------------------------------")
    print(
        "Default threshold (0.50) F1: "
        f"{f1_score(y_validation, default_predictions):.3f}"
    )
    print(f"Selected threshold:           {best_threshold:.2f}")
    print(f"Best validation F1:            {best_f1:.3f}")
    print(
        "Adjusted threshold F1:        "
        f"{f1_score(y_validation, adjusted_predictions):.3f}"
    )

    print("\nDefault predictions:", default_predictions)
    print("Adjusted predictions:", adjusted_predictions)


def main() -> None:
    """Run advanced evaluation experiments."""
    class_labels = ["negative", "neutral", "positive"]

    # Synthetic multiclass evaluation dataset with class imbalance.
    y_true = np.array(
        [0] * 12 + [1] * 6 + [2] * 3
    )
    y_pred = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2,
         1, 1, 1, 1, 0, 2,
         2, 2, 1]
    )

    print("Advanced NLP Evaluation Fundamentals")
    print("====================================")

    result = calculate_metrics(y_true, y_pred)
    print_metrics("Overall Evaluation", result)

    inspect_confusion_matrix(
        y_true,
        y_pred,
        class_labels,
    )

    print("\nMetric Interpretation")
    print("---------------------")
    print(
        "Accuracy measures overall correctness but can hide "
        "poor performance on minority classes."
    )
    print(
        "Macro F1 weights each class equally, while weighted F1 "
        "weights classes by their support."
    )
    print(
        "For single-label multiclass classification, micro F1 "
        "equals accuracy when all classes are included."
    )

    threshold_experiment()


if __name__ == "__main__":
    main()
