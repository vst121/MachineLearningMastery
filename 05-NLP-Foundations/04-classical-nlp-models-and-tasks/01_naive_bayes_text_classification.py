"""
Explore Multinomial Naive Bayes for classical text classification.

This experiment covers:
1. Creating a reproducible, self-contained text classification dataset.
2. Converting documents into sparse Bag of Words features.
3. Training a Multinomial Naive Bayes classifier.
4. Evaluating predictions using accuracy, precision, recall, and F1-score.
5. Inspecting class probabilities and prediction confidence.
6. Examining learned word probabilities for each class.
7. Comparing count-based features with binary occurrence features.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42

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


def create_pipeline(binary: bool = False) -> Pipeline:
    """Create a Bag of Words and Multinomial Naive Bayes pipeline."""
    return Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(
                    binary=binary,
                    lowercase=True,
                    stop_words=None,
                ),
            ),
            ("classifier", MultinomialNB(alpha=1.0)),
        ]
    )


def experiment_dataset_overview() -> None:
    """Inspect the dataset and its class distribution."""
    print("=== Dataset Overview ===")
    print(f"Number of documents: {len(DOCUMENTS)}")
    print(f"Number of labels: {len(LABELS)}")
    print(f"Classes: {sorted(set(LABELS))}")

    for label in sorted(set(LABELS)):
        count = sum(item == label for item in LABELS)
        print(f"  {label:12}: {count}")

    assert len(DOCUMENTS) == len(LABELS)
    assert len(DOCUMENTS) == 24


def experiment_train_and_evaluate() -> None:
    """Train Naive Bayes and evaluate it on held-out documents."""
    X_train, X_test, y_train, y_test = train_test_split(
        DOCUMENTS,
        LABELS,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=LABELS,
    )

    model = create_pipeline(binary=False)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n=== Training and Evaluation ===")
    print(f"Training documents: {len(X_train)}")
    print(f"Test documents: {len(X_test)}")
    print(f"Vocabulary size: {len(model.named_steps['vectorizer'].vocabulary_)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["sports", "technology"],
            zero_division=0,
        )
    )

    print("Confusion Matrix (rows=true, columns=predicted):")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["sports", "technology"],
        )
    )

    print("\nIndividual Predictions:")
    for document, actual, predicted in zip(X_test, y_test, predictions):
        status = "correct" if actual == predicted else "incorrect"
        print(f"  Text:      {document}")
        print(f"  Actual:    {actual}")
        print(f"  Predicted: {predicted} ({status})\n")


def experiment_class_probabilities() -> None:
    """Inspect predicted class probabilities for new documents."""
    X_train, _, y_train, _ = train_test_split(
        DOCUMENTS,
        LABELS,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=LABELS,
    )

    model = create_pipeline()
    model.fit(X_train, y_train)

    new_documents = [
        "the football player scored a goal",
        "the developer fixed the computer software",
        "the team released a new application",
        "the coach installed a security system",
    ]

    probabilities = model.predict_proba(new_documents)
    predictions = model.predict(new_documents)
    classes = model.named_steps["classifier"].classes_

    print("\n=== Class Probabilities ===")

    for document, prediction, row in zip(
        new_documents,
        predictions,
        probabilities,
    ):
        print(f"\nText: {document}")
        print(f"Predicted class: {prediction}")

        for label, probability in zip(classes, row):
            print(f"  P({label} | text) = {probability:.4f}")

        assert np.isclose(np.sum(row), 1.0)


def experiment_class_word_probabilities() -> None:
    """Inspect the most probable vocabulary terms for each class."""
    X_train, _, y_train, _ = train_test_split(
        DOCUMENTS,
        LABELS,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=LABELS,
    )

    model = create_pipeline()
    model.fit(X_train, y_train)

    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    vocabulary = np.asarray(vectorizer.get_feature_names_out())

    print("\n=== Most Probable Words by Class ===")

    for class_index, label in enumerate(classifier.classes_):
        log_probabilities = classifier.feature_log_prob_[class_index]
        top_indices = np.argsort(-log_probabilities)[:8]

        print(f"\nClass: {label}")
        for index in top_indices:
            print(
                f"  {vocabulary[index]:15} "
                f"P(word | class)={np.exp(log_probabilities[index]):.4f}"
            )


def experiment_count_vs_binary_features() -> None:
    """Compare raw term counts with binary term occurrence features."""
    X_train, X_test, y_train, y_test = train_test_split(
        DOCUMENTS,
        LABELS,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=LABELS,
    )

    results = {}

    print("\n=== Count Features vs Binary Features ===")

    for binary in [False, True]:
        model = create_pipeline(binary=binary)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        feature_type = "Binary occurrence" if binary else "Term counts"
        results[feature_type] = accuracy

        print(f"{feature_type:20} accuracy={accuracy:.4f}")

    print("\nFeature choice should be evaluated on representative data.")
    print("This small dataset is for learning, not production benchmarking.")


def main() -> None:
    experiment_dataset_overview()
    experiment_train_and_evaluate()
    experiment_class_probabilities()
    experiment_class_word_probabilities()
    experiment_count_vs_binary_features()


if __name__ == "__main__":
    main()