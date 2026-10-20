"""
Explore Logistic Regression for TF-IDF-based text classification.

This experiment covers:
1. Transforming text documents into TF-IDF feature vectors.
2. Training a Logistic Regression classifier using a scikit-learn pipeline.
3. Evaluating predictions with classification metrics.
4. Inspecting predicted class probabilities and decision scores.
5. Identifying the most influential positive and negative word features.
6. Comparing model behavior for familiar, ambiguous, and unseen vocabulary.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
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


def create_pipeline() -> Pipeline:
    """Create a TF-IDF and Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 1),
                    sublinear_tf=False,
                ),
            ),
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


def experiment_dataset_and_features() -> tuple[list[str], list[str], Pipeline]:
    """Split data and fit a pipeline without leaking test information."""
    X_train, X_test, y_train, y_test = train_test_split(
        DOCUMENTS,
        LABELS,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=LABELS,
    )

    model = create_pipeline()
    model.fit(X_train, y_train)

    vectorizer = model.named_steps["vectorizer"]
    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("=== TF-IDF Feature Representation ===")
    print(f"Training documents: {len(X_train)}")
    print(f"Test documents: {len(X_test)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Training matrix shape: {X_train_tfidf.shape}")
    print(f"Test matrix shape: {X_test_tfidf.shape}")
    print(f"Training matrix nonzero values: {X_train_tfidf.nnz}")
    print(f"Training matrix density: {X_train_tfidf.nnz / np.prod(X_train_tfidf.shape):.4f}")

    return X_test, y_test, model


def experiment_model_evaluation(
    X_test: list[str],
    y_test: list[str],
    model: Pipeline,
) -> None:
    """Evaluate the trained classifier on held-out documents."""
    predictions = model.predict(X_test)

    print("\n=== Model Evaluation ===")
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


def experiment_probabilities_and_decision_scores(
    model: Pipeline,
) -> None:
    """Inspect probability estimates and decision scores for sample texts."""
    samples = [
        "the football player scored a goal",
        "the developer fixed the computer software",
        "the team released a new application",
        "the coach installed a security system",
    ]

    classifier = model.named_steps["classifier"]
    probabilities = model.predict_proba(samples)
    decision_scores = model.decision_function(samples)
    predictions = model.predict(samples)

    print("\n=== Predictions and Confidence Signals ===")

    for index, document in enumerate(samples):
        print(f"\nText: {document}")
        print(f"Predicted class: {predictions[index]}")

        for class_index, label in enumerate(classifier.classes_):
            print(
                f"  P({label} | text)="
                f"{probabilities[index, class_index]:.4f}"
            )

        print(f"Decision score: {decision_scores[index]:.4f}")
        assert np.isclose(probabilities[index].sum(), 1.0)


def experiment_influential_features(model: Pipeline) -> None:
    """Inspect features with the strongest learned class associations."""
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()

    if len(classifier.classes_) != 2 or classifier.coef_.shape[0] != 1:
        print("\nFeature inspection in this experiment expects binary classification.")
        return

    weights = classifier.coef_[0]
    negative_indices = np.argsort(weights)[:10]
    positive_indices = np.argsort(weights)[-10:][::-1]

    negative_class = classifier.classes_[0]
    positive_class = classifier.classes_[1]

    print("\n=== Influential TF-IDF Features ===")
    print(f"Features associated with {negative_class}:")

    for index in negative_indices:
        print(f"  {feature_names[index]:15} weight={weights[index]:+.4f}")

    print(f"\nFeatures associated with {positive_class}:")

    for index in positive_indices:
        print(f"  {feature_names[index]:15} weight={weights[index]:+.4f}")

    print("\nFeature weights indicate learned associations, not causation.")


def experiment_unseen_vocabulary(model: Pipeline) -> None:
    """Inspect the representation of texts with unseen terms."""
    vectorizer = model.named_steps["vectorizer"]

    samples = [
        "football goal championship",
        "quantum blockchain spacecraft",
        "the football software application",
    ]

    print("\n=== Unseen Vocabulary and Ambiguous Text ===")

    for document in samples:
        vector = vectorizer.transform([document])
        prediction = model.predict([document])[0]
        probabilities = model.predict_proba([document])[0]
        known_terms = [
            term
            for term in document.lower().split()
            if term in vectorizer.vocabulary_
        ]

        print(f"\nText: {document}")
        print(f"Known terms: {known_terms}")
        print(f"Nonzero TF-IDF features: {vector.nnz}")
        print(f"Predicted class: {prediction}")
        print(
            "Class probabilities: "
            + str(
                {
                    label: round(float(probability), 4)
                    for label, probability in zip(
                        model.named_steps["classifier"].classes_,
                        probabilities,
                    )
                }
            )
        )

        if not known_terms:
            assert vector.nnz == 0


def main() -> None:
    X_test, y_test, model = experiment_dataset_and_features()

    experiment_model_evaluation(X_test, y_test, model)
    experiment_probabilities_and_decision_scores(model)
    experiment_influential_features(model)
    experiment_unseen_vocabulary(model)


if __name__ == "__main__":
    main()