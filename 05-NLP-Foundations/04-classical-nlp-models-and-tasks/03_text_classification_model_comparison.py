"""
Compare classical machine learning models for text classification.

This experiment covers:
1. Building a reproducible, self-contained text classification dataset.
2. Comparing Bag of Words and TF-IDF feature representations.
3. Evaluating Multinomial Naive Bayes and Logistic Regression.
4. Using the same stratified train-test split for every experiment.
5. Comparing accuracy, precision, recall, and F1-score.
6. Examining confusion matrices and sparse feature dimensions.
7. Selecting a model based on evaluation results rather than assumptions.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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


def create_experiments() -> dict[str, Pipeline]:
    """Build pipelines representing different model-feature combinations."""
    return {
        "BoW + Naive Bayes": Pipeline(
            [
                ("vectorizer", CountVectorizer()),
                ("classifier", MultinomialNB(alpha=1.0)),
            ]
        ),
        "TF-IDF + Naive Bayes": Pipeline(
            [
                ("vectorizer", TfidfVectorizer()),
                ("classifier", MultinomialNB(alpha=1.0)),
            ]
        ),
        "BoW + Logistic Regression": Pipeline(
            [
                ("vectorizer", CountVectorizer()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "TF-IDF + Logistic Regression": Pipeline(
            [
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
        ),
    }


def experiment_dataset_split() -> tuple[list[str], list[str], list[str], list[str]]:
    """Create one shared, stratified train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        DOCUMENTS,
        LABELS,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=LABELS,
    )

    print("=== Dataset and Split ===")
    print(f"Total documents: {len(DOCUMENTS)}")
    print(f"Training documents: {len(X_train)}")
    print(f"Test documents: {len(X_test)}")
    print(f"Class labels: {LABEL_ORDER}")

    for label in LABEL_ORDER:
        train_count = sum(item == label for item in y_train)
        test_count = sum(item == label for item in y_test)
        print(f"{label:12} train={train_count}, test={test_count}")

    assert len(X_train) + len(X_test) == len(DOCUMENTS)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    return X_train, X_test, y_train, y_test


def experiment_model_comparison(
    X_train: list[str],
    X_test: list[str],
    y_train: list[str],
    y_test: list[str],
) -> pd.DataFrame:
    """Fit all pipelines and compare their held-out performance."""
    results = []
    fitted_models = create_experiments()

    print("\n=== Model Comparison ===")

    for name, model in fitted_models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=0,
        )

        accuracy = accuracy_score(y_test, predictions)
        vectorizer = model.named_steps["vectorizer"]
        X_train_features = vectorizer.transform(X_train)

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Macro Precision": precision,
                "Macro Recall": recall,
                "Macro F1": f1,
                "Vocabulary Size": len(vectorizer.vocabulary_),
                "Train Matrix Density": (
                    X_train_features.nnz / np.prod(X_train_features.shape)
                ),
            }
        )

        print(f"\n{name}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro precision: {precision:.4f}")
        print(f"  Macro recall: {recall:.4f}")
        print(f"  Macro F1: {f1:.4f}")
        print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
        print(
            "  Training matrix density: "
            f"{X_train_features.nnz / np.prod(X_train_features.shape):.4f}"
        )
        print("  Confusion matrix:")
        print(
            confusion_matrix(
                y_test,
                predictions,
                labels=LABEL_ORDER,
            )
        )

    results_df = pd.DataFrame(results).sort_values(
        by="Macro F1",
        ascending=False,
    )

    print("\n=== Results Ranked by Macro F1 ===")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    return results_df


def experiment_best_model_report(
    X_train: list[str],
    X_test: list[str],
    y_train: list[str],
    y_test: list[str],
    results: pd.DataFrame,
) -> None:
    """Re-fit the top-ranked configuration and inspect its report."""
    best_name = str(results.iloc[0]["Model"])
    model = create_experiments()[best_name]
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n=== Top-Ranked Configuration ===")
    print(f"Selected by macro F1: {best_name}")
    print(
        classification_report(
            y_test,
            predictions,
            labels=LABEL_ORDER,
            zero_division=0,
        )
    )

    print(
        "This is the top result on one small test split, "
        "not evidence of general superiority."
    )


def main() -> None:
    X_train, X_test, y_train, y_test = experiment_dataset_split()

    results = experiment_model_comparison(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    experiment_best_model_report(
        X_train,
        X_test,
        y_train,
        y_test,
        results,
    )


if __name__ == "__main__":
    main()