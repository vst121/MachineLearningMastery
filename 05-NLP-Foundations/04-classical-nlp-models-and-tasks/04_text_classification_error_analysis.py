"""
Analyze errors made by a classical text classification model.

This experiment covers:
1. Training a TF-IDF and Logistic Regression pipeline.
2. Evaluating predictions on a stratified held-out test set.
3. Identifying false positives and false negatives.
4. Inspecting predicted probabilities for misclassified examples.
5. Comparing class-specific precision, recall, and F1-score.
6. Investigating errors caused by ambiguous or mixed-topic text.
7. Demonstrating why error analysis requires representative test data.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
LABEL_ORDER = ["sports", "technology"]

DOCUMENTS = [
    "the team won the football match",
    "the player scored a goal",
    "the coach prepared the team",
    "the football match was exciting",
    "the team practiced before the match",
    "the player won the championship",
    "the coach trained the players",
    "the football team scored twice",
    "the match ended with a victory",
    "the player practiced football every day",
    "the team celebrated its championship",
    "the coach discussed the next match",
    "the software update fixed a bug",
    "the computer runs the application",
    "the developer wrote new software",
    "the application crashed after the update",
    "the computer processed the data",
    "the developer fixed the software bug",
    "the system installed a security update",
    "the application runs on the computer",
    "the software developer tested the application",
    "the system detected a computer error",
    "the developer released a software update",
    "the application processed the input data",
]

LABELS = [
    "sports", "sports", "sports", "sports",
    "sports", "sports", "sports", "sports",
    "sports", "sports", "sports", "sports",
    "technology", "technology", "technology", "technology",
    "technology", "technology", "technology", "technology",
    "technology", "technology", "technology", "technology",
]

# Additional examples for qualitative inspection after model training.
# These examples are not used to fit the vectorizer or classifier.
INSPECTION_DOCUMENTS = [
    "the football team won the match",
    "the developer fixed the software bug",
    "the team released a new application",
    "the coach installed a computer system",
    "the player analyzed the match data",
    "the software tracks player performance",
    "a completely unfamiliar phrase xyzabc qwerty",
]


def create_pipeline() -> Pipeline:
    """Create a TF-IDF and Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("vectorizer", TfidfVectorizer()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def experiment_test_set_errors() -> tuple[Pipeline, list[str], list[str], list[str]]:
    """Train the model and inspect held-out test predictions."""
    X_train, X_test, y_train, y_test = train_test_split(
        DOCUMENTS,
        LABELS,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=LABELS,
    )

    model = create_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.named_steps["classifier"].classes_

    results = pd.DataFrame(
        {
            "text": X_test,
            "actual": y_test,
            "predicted": predictions,
            "confidence": probabilities.max(axis=1),
        }
    )
    results["correct"] = results["actual"] == results["predicted"]

    print("=== Held-Out Test Evaluation ===")
    print(f"Training documents: {len(X_train)}")
    print(f"Test documents: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=LABEL_ORDER,
            zero_division=0,
        )
    )

    print("Confusion Matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=LABEL_ORDER))

    print("\n=== Individual Test Predictions ===")
    for _, row in results.iterrows():
        status = "CORRECT" if row["correct"] else "ERROR"
        print(f"\n[{status}] {row['text']}")
        print(f"  Actual: {row['actual']}")
        print(f"  Predicted: {row['predicted']}")
        print(f"  Maximum predicted probability: {row['confidence']:.4f}")

        sample_index = X_test.index(row["text"])
        for label, probability in zip(classes, probabilities[sample_index] if False else model.predict_proba([row["text"]])[0]):
            print(f"  P({label} | text) = {probability:.4f}")

    errors = results.loc[~results["correct"]].copy()

    print("\n=== Misclassified Test Examples ===")
    if errors.empty:
        print("No errors on this test split.")
    else:
        print(errors.to_string(index=False))

    return model, X_test, y_test, predictions


def experiment_class_specific_metrics(
    y_test: list[str],
    predictions: list[str],
) -> None:
    """Calculate precision, recall, and F1-score for each class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    metrics = pd.DataFrame(
        {
            "class": LABEL_ORDER,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    )

    print("\n=== Class-Specific Metrics ===")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    for _, row in metrics.iterrows():
        print(
            f"\n{row['class']}: "
            f"precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, "
            f"F1={row['f1_score']:.4f}, "
            f"support={int(row['support'])}"
        )


def experiment_ambiguous_and_unseen_text(model: Pipeline) -> None:
    """Inspect predictions for mixed-topic and unfamiliar examples."""
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]

    vectors = vectorizer.transform(INSPECTION_DOCUMENTS)
    predictions = model.predict(INSPECTION_DOCUMENTS)
    probabilities = model.predict_proba(INSPECTION_DOCUMENTS)

    print("\n=== Qualitative Error Analysis ===")

    for index, document in enumerate(INSPECTION_DOCUMENTS):
        known_terms = [
            term
            for term in document.lower().split()
            if term in vectorizer.vocabulary_
        ]

        print(f"\nText: {document}")
        print(f"Known terms: {known_terms}")
        print(f"Nonzero TF-IDF features: {vectors[index].nnz}")
        print(f"Predicted class: {predictions[index]}")

        for label, probability in zip(classifier.classes_, probabilities[index]):
            print(f"  P({label} | text) = {probability:.4f}")

        if vectors[index].nnz == 0:
            print(
                "Warning: no known features were found. "
                "The prediction is based on the model's learned intercept."
            )

    print(
        "\nThese are qualitative examples, not labeled test cases. "
        "Their predictions do not establish model accuracy."
    )


def experiment_error_analysis_principles() -> None:
    """Summarize practical questions to ask when reviewing errors."""
    questions = [
        "Is the label correct and unambiguous?",
        "Does the document contain words associated with multiple classes?",
        "Are important terms missing from the training vocabulary?",
        "Is one class underrepresented in the training data?",
        "Does the evaluation set reflect real application traffic?",
        "Would additional data, different features, or a different model help?",
    ]

    print("\n=== Error Analysis Checklist ===")
    for index, question in enumerate(questions, start=1):
        print(f"{index}. {question}")


def main() -> None:
    model, _, y_test, predictions = experiment_test_set_errors()

    experiment_class_specific_metrics(y_test, predictions)
    experiment_ambiguous_and_unseen_text(model)
    experiment_error_analysis_principles()


if __name__ == "__main__":
    main()