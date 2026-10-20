"""
Token Classification and Information Extraction

Experiments:
- Load a pretrained token-classification model for named entity recognition.
- Inspect token-level labels, confidence scores, and wordpiece tokens.
- Compare entity aggregation strategies.
- Convert token predictions into structured entity records.
- Evaluate entity extraction against manually annotated examples.
"""

from collections import Counter
from typing import Any

from transformers import pipeline


MODEL_NAME = "dslim/bert-base-NER"

EXAMPLES = [
    {
        "text": (
            "Vahid works at Microsoft in Berlin. "
            "He previously studied at the University of Mannheim."
        ),
        "expected_entities": [
            ("Vahid", "PER"),
            ("Microsoft", "ORG"),
            ("Berlin", "LOC"),
            ("University of Mannheim", "ORG"),
        ],
    },
    {
        "text": (
            "Angela Merkel was born in Hamburg and served as "
            "Chancellor of Germany."
        ),
        "expected_entities": [
            ("Angela Merkel", "PER"),
            ("Hamburg", "LOC"),
            ("Germany", "LOC"),
        ],
    },
]


def load_ner_pipeline():
    """Load a pretrained NER model and return its inference pipeline."""
    print(f"Loading token-classification model: {MODEL_NAME}")

    return pipeline(
        task="token-classification",
        model=MODEL_NAME,
        aggregation_strategy="none",
    )


def display_token_predictions(
    text: str,
    predictions: list[dict[str, Any]],
) -> None:
    """Display raw token-level predictions."""
    print("\nText:")
    print(text)

    print("\nRaw token predictions:")
    print(f"{'Token':25} {'Label':12} {'Confidence':>12}")

    for prediction in predictions:
        token = prediction.get("word", "")
        label = prediction.get("entity", "O")
        score = float(prediction.get("score", 0.0))

        print(f"{token:25} {label:12} {score:12.4f}")


def extract_entities(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge consecutive BIO-style token predictions into entity records.

    This illustrative merger handles B- and I- labels and tolerates
    subword tokens. Production systems should validate the model's label
    scheme and use tokenizer offsets when exact source spans are required.
    """
    entities = []
    current = None

    def flush_current():
        nonlocal current

        if current is not None:
            current["text"] = "".join(current.pop("pieces")).replace(
                " ##", ""
            )
            entities.append(current)
            current = None

    for prediction in predictions:
        label = prediction.get("entity", "O")
        token = prediction.get("word", "")
        score = float(prediction.get("score", 0.0))

        if label == "O" or "-" not in label:
            flush_current()
            continue

        prefix, entity_type = label.split("-", maxsplit=1)

        if prefix == "B":
            flush_current()
            current = {
                "label": entity_type,
                "pieces": [token],
                "scores": [score],
            }
        elif prefix == "I":
            if current is not None and current["label"] == entity_type:
                current["pieces"].append(token)
                current["scores"].append(score)
            else:
                # Recover gracefully from an I-label without a matching
                # preceding entity. Strict BIO validation is a separate task.
                flush_current()
                current = {
                    "label": entity_type,
                    "pieces": [token],
                    "scores": [score],
                }
        else:
            flush_current()

    flush_current()

    for entity in entities:
        entity["confidence"] = sum(entity.pop("scores")) / len(
            entity["scores"]
        )

    return entities


def compare_aggregation_strategies(ner, text: str) -> None:
    """Compare raw token predictions with pipeline aggregation."""
    raw_predictions = ner(text)

    aggregated_ner = pipeline(
        task="token-classification",
        model=MODEL_NAME,
        aggregation_strategy="simple",
    )
    aggregated_predictions = aggregated_ner(text)

    print("\n--- Raw token-level predictions ---")
    for item in raw_predictions:
        print(
            f"{item['word']!r:25} "
            f"{item['entity']:10} "
            f"{float(item['score']):.4f}"
        )

    print("\n--- Pipeline-aggregated entities ---")
    for item in aggregated_predictions:
        print(
            f"{item['word']!r:35} "
            f"{item['entity_group']:8} "
            f"{float(item['score']):.4f}"
        )


def calculate_entity_metrics(
    predicted: list[dict[str, Any]],
    expected: list[tuple[str, str]],
) -> dict[str, float]:
    """Calculate exact-match entity precision, recall, and F1."""
    predicted_set = {
        (item["text"].strip(), item["label"])
        for item in predicted
    }
    expected_set = {
        (text, label)
        for text, label in expected
    }

    true_positives = len(predicted_set & expected_set)
    false_positives = len(predicted_set - expected_set)
    false_negatives = len(expected_set - predicted_set)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": float(true_positives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
    }


def run_entity_extraction(ner) -> list[dict[str, Any]]:
    """Extract entities from each example and evaluate exact matches."""
    all_entities = []
    aggregate_counts = Counter()

    for index, example in enumerate(EXAMPLES, start=1):
        text = example["text"]
        expected = example["expected_entities"]

        print(f"\n{'=' * 72}")
        print(f"Example {index}")

        raw_predictions = ner(text)
        display_token_predictions(text, raw_predictions)

        entities = extract_entities(raw_predictions)

        print("\nExtracted entities:")
        for entity in entities:
            print(
                f"- {entity['text']!r} "
                f"[{entity['label']}] "
                f"confidence={entity['confidence']:.4f}"
            )

        metrics = calculate_entity_metrics(entities, expected)

        print("\nExact-match entity metrics:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1:        {metrics['f1']:.4f}")

        for key in (
            "true_positives",
            "false_positives",
            "false_negatives",
        ):
            aggregate_counts[key] += int(metrics[key])

        all_entities.append(
            {
                "text": text,
                "entities": entities,
                "metrics": metrics,
            }
        )

    tp = aggregate_counts["true_positives"]
    fp = aggregate_counts["false_positives"]
    fn = aggregate_counts["false_negatives"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    print(f"\n{'=' * 72}")
    print("Overall exact-match entity metrics")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    return all_entities


def main():
    ner = load_ner_pipeline()

    first_text = EXAMPLES[0]["text"]
    compare_aggregation_strategies(ner, first_text)

    run_entity_extraction(ner)


if __name__ == "__main__":
    main()
