"""
Document Classification and Hierarchical Processing

Experiments:
- Classify documents using a pretrained sequence-classification model.
- Compare truncation with chunk-based document processing.
- Aggregate chunk predictions into document-level predictions.
- Compare mean-probability and majority-vote aggregation.
- Analyze how chunk count and aggregation affect document predictions.
"""

from collections import Counter
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
CHUNK_WORDS = 100
CHUNK_OVERLAP_WORDS = 20

DOCUMENTS = [
    {
        "id": "positive_review",
        "text": """
        The product arrived quickly and was easy to set up.
        The interface is clear, the performance is reliable, and the
        documentation explains the important features well.
        Customer support answered our questions promptly.
        Overall, the experience has been excellent.
        """
        * 12,
        "expected_label": "POSITIVE",
    },
    {
        "id": "negative_review",
        "text": """
        The application frequently crashes and loses user changes.
        Navigation is confusing, several features are unreliable,
        and the documentation does not explain common problems.
        Support responses were slow and did not solve the issue.
        Overall, this has been a frustrating experience.
        """
        * 12,
        "expected_label": "NEGATIVE",
    },
]


def load_classifier():
    """Load a pretrained sentiment classifier and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    return tokenizer, model, classifier


def split_into_chunks(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Word-based splitting is a simple baseline. Token-based splitting is
    preferable when strict model token limits must be guaranteed.
    """
    words = text.split()

    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero.")

    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError(
            "overlap_words must be non-negative and smaller than chunk_words."
        )

    if not words:
        return []

    step = chunk_words - overlap_words
    chunks = []

    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]

        if not chunk:
            break

        chunks.append(" ".join(chunk))

        if start + chunk_words >= len(words):
            break

    return chunks


def classify_truncated(
    classifier,
    text: str,
    tokenizer,
) -> dict[str, Any]:
    """Classify a document after restricting it to the model input limit."""
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors=None,
    )

    truncated_text = tokenizer.decode(
        encoded["input_ids"],
        skip_special_tokens=True,
    )

    prediction = classifier(truncated_text, truncation=True)[0]

    return {
        "label": prediction["label"],
        "confidence": float(prediction["score"]),
        "input_tokens": len(encoded["input_ids"]),
    }


def classify_chunks(
    classifier,
    tokenizer,
    text: str,
) -> list[dict[str, Any]]:
    """Classify each chunk independently."""
    chunks = split_into_chunks(text)
    results = []

    for index, chunk in enumerate(chunks):
        prediction = classifier(chunk, truncation=True)[0]

        token_count = len(
            tokenizer(
                chunk,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )["input_ids"]
        )

        results.append(
            {
                "chunk_index": index,
                "text": chunk,
                "label": prediction["label"],
                "confidence": float(prediction["score"]),
                "token_count": token_count,
            }
        )

    return results


def aggregate_mean_probability(
    chunk_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate chunk-level labels using confidence-weighted class scores.

    Each chunk contributes its confidence to its predicted class.
    This is a simple baseline, not a substitute for aggregating full
    class-probability vectors from the model.
    """
    if not chunk_predictions:
        return {"label": "UNKNOWN", "confidence": 0.0}

    scores = Counter()
    counts = Counter()

    for prediction in chunk_predictions:
        label = prediction["label"]
        scores[label] += prediction["confidence"]
        counts[label] += 1

    label = max(scores, key=scores.get)

    return {
        "label": label,
        "confidence": scores[label] / counts[label],
        "class_scores": {
            name: scores[name] / counts[name]
            for name in scores
        },
    }


def aggregate_majority_vote(
    chunk_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Choose the document label by majority vote across chunks."""
    if not chunk_predictions:
        return {"label": "UNKNOWN", "vote_fraction": 0.0}

    votes = Counter(
        prediction["label"] for prediction in chunk_predictions
    )
    label, count = votes.most_common(1)[0]

    return {
        "label": label,
        "vote_fraction": count / len(chunk_predictions),
        "votes": dict(votes),
    }


def evaluate_document(
    document: dict[str, str],
    tokenizer,
    classifier,
) -> dict[str, Any]:
    """Compare truncation and chunk aggregation for one document."""
    text = document["text"]
    expected = document["expected_label"]

    truncated = classify_truncated(classifier, text, tokenizer)
    chunks = classify_chunks(classifier, tokenizer, text)
    mean_aggregate = aggregate_mean_probability(chunks)
    majority_aggregate = aggregate_majority_vote(chunks)

    results = {
        "document_id": document["id"],
        "expected_label": expected,
        "truncated": truncated,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "mean_probability_aggregate": mean_aggregate,
        "majority_vote_aggregate": majority_aggregate,
    }

    print(f"\n{'=' * 72}")
    print(f"Document: {document['id']}")
    print(f"Expected label: {expected}")
    print(f"Word count: {len(text.split())}")
    print(f"Chunk count: {len(chunks)}")

    print(
        f"\nTruncation: {truncated['label']} "
        f"(confidence={truncated['confidence']:.4f}, "
        f"tokens={truncated['input_tokens']})"
    )

    print("\nChunk predictions:")
    for chunk in chunks:
        print(
            f"Chunk {chunk['chunk_index']:02d}: "
            f"{chunk['label']:8} "
            f"confidence={chunk['confidence']:.4f} "
            f"tokens={chunk['token_count']}"
        )

    print(
        "\nMean-confidence aggregation: "
        f"{mean_aggregate['label']} "
        f"(confidence={mean_aggregate['confidence']:.4f})"
    )

    print(
        "Majority vote: "
        f"{majority_aggregate['label']} "
        f"(vote fraction={majority_aggregate['vote_fraction']:.4f})"
    )

    for method, prediction in (
        ("truncated", truncated),
        ("mean_probability_aggregate", mean_aggregate),
        ("majority_vote_aggregate", majority_aggregate),
    ):
        results[method]["correct"] = (
            prediction["label"] == expected
        )

    return results


def summarize_results(results: list[dict[str, Any]]) -> None:
    """Summarize document-level accuracy for each strategy."""
    if not results:
        print("No documents to evaluate.")
        return

    strategies = (
        "truncated",
        "mean_probability_aggregate",
        "majority_vote_aggregate",
    )

    print(f"\n{'=' * 72}")
    print("Document-level accuracy")

    for strategy in strategies:
        correct = sum(
            result[strategy]["correct"] for result in results
        )
        accuracy = correct / len(results)

        print(
            f"{strategy:32} "
            f"{correct}/{len(results)} "
            f"accuracy={accuracy:.4f}"
        )


def main():
    tokenizer, _, classifier = load_classifier()

    results = [
        evaluate_document(document, tokenizer, classifier)
        for document in DOCUMENTS
    ]

    summarize_results(results)


if __name__ == "__main__":
    main()
