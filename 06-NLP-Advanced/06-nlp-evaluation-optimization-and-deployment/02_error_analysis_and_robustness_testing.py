"""
NLP Error Analysis and Robustness Testing

Experiments:
- Categorize classification errors by type.
- Compare predictions on clean and perturbed text.
- Test robustness to typos, casing, and punctuation changes.
- Measure prediction consistency under text perturbations.
- Identify examples that require manual review.
- Summarize robustness metrics across perturbation types.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class PredictionRecord:
    """Represent one model prediction for analysis."""

    text: str
    true_label: str
    predicted_label: str
    confidence: float


def build_dataset() -> tuple[list[str], list[str]]:
    """Create a small synthetic sentiment dataset."""
    examples = {
        "positive": [
            "The product is excellent and works perfectly",
            "I really love this wonderful service",
            "The experience was great and very enjoyable",
            "This is a fantastic product with excellent quality",
            "The support team was helpful and friendly",
            "I am happy with the results",
            "The application is easy to use and reliable",
            "Delivery was quick and the item was perfect",
            "The quality exceeded my expectations",
            "Everything worked well and I recommend it",
            "A pleasant experience from beginning to end",
            "The service is fast and dependable",
            "This product is impressive and useful",
            "The staff were kind and professional",
            "I would gladly buy this again",
            "The update fixed the problem completely",
        ],
        "negative": [
            "The product is terrible and does not work",
            "I really dislike this awful service",
            "The experience was poor and disappointing",
            "This is a bad product with terrible quality",
            "The support team was rude and unhelpful",
            "I am unhappy with the results",
            "The application is confusing and unreliable",
            "Delivery was late and the item was damaged",
            "The quality was below my expectations",
            "Nothing worked properly and I cannot recommend it",
            "A frustrating experience from beginning to end",
            "The service is slow and unreliable",
            "This product is useless and disappointing",
            "The staff were careless and unprofessional",
            "I would never buy this again",
            "The update made the problem much worse",
        ],
    }

    texts: list[str] = []
    labels: list[str] = []

    for label, sentences in examples.items():
        texts.extend(sentences)
        labels.extend([label] * len(sentences))

    return texts, labels


def build_classifier() -> Pipeline:
    """Build a lightweight baseline text classifier."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def perturb_text(text: str, perturbation: str) -> str:
    """Apply one controlled text perturbation."""
    if perturbation == "uppercase":
        return text.upper()

    if perturbation == "remove_punctuation":
        return "".join(
            character
            for character in text
            if character.isalnum() or character.isspace()
        )

    if perturbation == "extra_whitespace":
        return "   ".join(text.split())

    if perturbation == "character_noise":
        words = text.split()
        for index, word in enumerate(words):
            if len(word) >= 5:
                words[index] = word[:2] + word[3] + word[2] + word[4:]
                break
        return " ".join(words)

    if perturbation == "shortened":
        words = text.split()
        return " ".join(words[: max(1, len(words) // 2)])

    if perturbation == "negation_prefix":
        return "not " + text

    raise ValueError(f"Unknown perturbation: {perturbation}")


def predict_records(
    model: Pipeline,
    texts: list[str],
    labels: list[str],
) -> list[PredictionRecord]:
    """Generate predictions and confidence scores."""
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)
    classes = model.named_steps["classifier"].classes_

    records: list[PredictionRecord] = []

    for text, true_label, prediction, scores in zip(
        texts,
        labels,
        predictions,
        probabilities,
    ):
        predicted_index = int(np.argmax(scores))
        confidence = float(scores[predicted_index])

        records.append(
            PredictionRecord(
                text=text,
                true_label=true_label,
                predicted_label=str(prediction),
                confidence=confidence,
            )
        )

    return records


def classify_errors(
    records: list[PredictionRecord],
) -> dict[str, list[PredictionRecord]]:
    """Group misclassified examples into useful review categories."""
    categories: dict[str, list[PredictionRecord]] = defaultdict(list)

    for record in records:
        if record.true_label == record.predicted_label:
            continue

        if record.confidence < 0.60:
            category = "low-confidence error"
        elif len(record.text.split()) <= 4:
            category = "short-text error"
        else:
            category = "confident misclassification"

        categories[category].append(record)

    return categories


def evaluate_robustness(
    model: Pipeline,
    texts: list[str],
    labels: list[str],
) -> None:
    """Compare clean accuracy with perturbed-text accuracy."""
    clean_predictions = model.predict(texts)
    clean_accuracy = accuracy_score(labels, clean_predictions)

    print("\nRobustness Evaluation")
    print("---------------------")
    print(f"Clean accuracy: {clean_accuracy:.3f}")

    perturbations = [
        "uppercase",
        "remove_punctuation",
        "extra_whitespace",
        "character_noise",
        "shortened",
        "negation_prefix",
    ]

    clean_predictions_array = np.asarray(clean_predictions)

    for perturbation in perturbations:
        changed_texts = [
            perturb_text(text, perturbation)
            for text in texts
        ]
        changed_predictions = model.predict(changed_texts)

        perturbed_accuracy = accuracy_score(
            labels,
            changed_predictions,
        )
        consistency = float(
            np.mean(changed_predictions == clean_predictions_array)
        )

        print(f"\nPerturbation: {perturbation}")
        print(f"  Accuracy:              {perturbed_accuracy:.3f}")
        print(f"  Prediction consistency:{consistency:>7.3f}")
        print(
            "  Accuracy change:       "
            f"{perturbed_accuracy - clean_accuracy:+.3f}"
        )


def main() -> None:
    """Run error analysis and robustness experiments."""
    texts, labels = build_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = build_classifier()
    model.fit(x_train, y_train)

    records = predict_records(model, x_test, y_test)
    predictions = [record.predicted_label for record in records]

    print("NLP Error Analysis and Robustness Testing")
    print("=========================================")
    print(f"Training examples: {len(x_train)}")
    print(f"Test examples:     {len(x_test)}")
    print(f"Test accuracy:     {accuracy_score(y_test, predictions):.3f}")

    print("\nClassification Report")
    print("---------------------")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["negative", "positive"],
            zero_division=0,
        )
    )

    print("\nMisclassification Review")
    print("------------------------")
    errors = classify_errors(records)

    if not errors:
        print("No errors found in this small test split.")
    else:
        for category, examples in sorted(errors.items()):
            print(f"\n{category}: {len(examples)}")
            for example in examples:
                print(f"  Text:       {example.text}")
                print(f"  True label: {example.true_label}")
                print(f"  Predicted:  {example.predicted_label}")
                print(f"  Confidence: {example.confidence:.3f}")

    evaluate_robustness(model, x_test, y_test)

    print("\nInterpretation Notes")
    print("--------------------")
    print(
        "Prediction consistency measures agreement with clean-text "
        "predictions; it does not prove that those predictions are correct."
    )
    print(
        "Some perturbations, especially negation and shortening, can change "
        "meaning. A changed prediction is not automatically a robustness failure."
    )
    print(
        "This dataset is intentionally small and synthetic. Treat results as "
        "a demonstration of the evaluation workflow, not as a benchmark."
    )


if __name__ == "__main__":
    main()
