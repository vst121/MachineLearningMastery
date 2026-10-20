"""
Document Understanding Pipeline

Experiments:
- Normalize and segment documents into manageable text chunks.
- Classify document sentiment and extract named entities.
- Answer questions using the document as the evidence source.
- Validate answer offsets and preserve supporting evidence.
- Produce structured JSON records for downstream applications.
"""

import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import pipeline


OUTPUT_DIR = Path("outputs/document-understanding")
OUTPUT_FILE = OUTPUT_DIR / "document_analysis.json"

CLASSIFICATION_MODEL = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
NER_MODEL = "dslim/bert-base-NER"
QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"

CHUNK_WORDS = 180
CHUNK_OVERLAP_WORDS = 30

DOCUMENTS = [
    {
        "document_id": "doc-001",
        "title": "Company Overview",
        "text": """
        Northstar Analytics was founded in 2018 in Berlin.
        The company develops business intelligence software and employs
        240 people. Chief executive officer Maya Schneider announced
        a new research office in Munich in 2023. The company plans to
        expand its analytics platform for enterprise customers.
        """,
        "questions": [
            "When was Northstar Analytics founded?",
            "Where was Northstar Analytics founded?",
            "Who is the chief executive officer?",
            "How many people does the company employ?",
        ],
    },
    {
        "document_id": "doc-002",
        "title": "Product Feedback",
        "text": """
        The new application is easy to use and performs reliably.
        Customers appreciate its clear interface and helpful support team.
        However, some users report that advanced reports load slowly.
        The product team is improving performance and adding new features.
        Overall, the feedback has been encouraging.
        """,
        "questions": [
            "What problem do some users report?",
            "What is the product team improving?",
        ],
    },
]


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving readable punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def split_into_chunks(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split a document into overlapping word-based chunks."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero.")

    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError(
            "overlap_words must be non-negative and smaller than chunk_words."
        )

    words = text.split()

    if not words:
        return []

    step = chunk_words - overlap_words
    chunks = []

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words])

        if chunk:
            chunks.append(chunk)

        if start + chunk_words >= len(words):
            break

    return chunks


def load_pipelines() -> dict[str, Any]:
    """Load the pretrained models used by the document pipeline."""
    device = 0 if torch.cuda.is_available() else -1

    print("Loading document understanding models...")

    return {
        "classification": pipeline(
            "text-classification",
            model=CLASSIFICATION_MODEL,
            device=device,
        ),
        "ner": pipeline(
            "token-classification",
            model=NER_MODEL,
            aggregation_strategy="simple",
            device=device,
        ),
        "qa": pipeline(
            "question-answering",
            model=QA_MODEL,
            device=device,
        ),
    }


def classify_document(
    classifier,
    chunks: list[str],
) -> dict[str, Any]:
    """Classify each chunk and aggregate predicted class scores."""
    if not chunks:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "chunk_predictions": [],
        }

    predictions = []

    for index, chunk in enumerate(chunks):
        result = classifier(chunk, truncation=True)[0]

        predictions.append(
            {
                "chunk_index": index,
                "label": result["label"],
                "confidence": float(result["score"]),
            }
        )

    # This baseline averages confidence within each predicted class.
    # It does not average the full probability vector across classes.
    class_scores: dict[str, list[float]] = {}

    for prediction in predictions:
        class_scores.setdefault(prediction["label"], []).append(
            prediction["confidence"]
        )

    average_scores = {
        label: sum(scores) / len(scores)
        for label, scores in class_scores.items()
    }

    best_label = max(average_scores, key=average_scores.get)

    return {
        "label": best_label,
        "confidence": average_scores[best_label],
        "class_average_scores": average_scores,
        "chunk_predictions": predictions,
    }


def extract_entities(
    ner,
    chunks: list[str],
) -> list[dict[str, Any]]:
    """Extract entities from chunks and remove duplicate predictions."""
    unique_entities = {}

    for chunk_index, chunk in enumerate(chunks):
        for prediction in ner(chunk):
            entity_text = prediction["word"].strip()
            entity_label = prediction["entity_group"]

            key = (entity_text.casefold(), entity_label)

            entity = {
                "text": entity_text,
                "label": entity_label,
                "confidence": float(prediction["score"]),
                "chunk_index": chunk_index,
            }

            existing = unique_entities.get(key)

            if (
                existing is None
                or entity["confidence"] > existing["confidence"]
            ):
                unique_entities[key] = entity

    return sorted(
        unique_entities.values(),
        key=lambda entity: (entity["label"], entity["text"].casefold()),
    )


def answer_questions(
    qa,
    questions: list[str],
    chunks: list[str],
) -> list[dict[str, Any]]:
    """
    Answer each question using the chunk with the strongest QA score.

    QA scores are model scores, not calibrated probabilities that an
    answer is factually correct. The selected chunk should be reviewed
    when evidence quality is important.
    """
    results = []

    for question in questions:
        candidates = []

        for chunk_index, chunk in enumerate(chunks):
            prediction = qa(
                question=question,
                context=chunk,
                handle_impossible_answer=True,
            )

            answer = prediction.get("answer", "").strip()
            start = int(prediction.get("start", 0))
            end = int(prediction.get("end", 0))
            score = float(prediction.get("score", 0.0))

            valid_offsets = (
                0 <= start <= end <= len(chunk)
                and chunk[start:end] == answer
            )

            candidates.append(
                {
                    "question": question,
                    "answer": answer,
                    "confidence": score,
                    "chunk_index": chunk_index,
                    "evidence": chunk[start:end] if valid_offsets else "",
                    "offsets_valid": valid_offsets,
                    "answer_in_context": (
                        bool(answer)
                        and answer.casefold() in chunk.casefold()
                    ),
                }
            )

        if candidates:
            best = max(candidates, key=lambda item: item["confidence"])
            results.append(best)
        else:
            results.append(
                {
                    "question": question,
                    "answer": "",
                    "confidence": 0.0,
                    "chunk_index": None,
                    "evidence": "",
                    "offsets_valid": False,
                    "answer_in_context": False,
                }
            )

    return results


def process_document(
    document: dict[str, Any],
    models: dict[str, Any],
) -> dict[str, Any]:
    """Run the complete document understanding workflow."""
    text = normalize_text(document["text"])
    chunks = split_into_chunks(text)

    classification = classify_document(
        models["classification"],
        chunks,
    )

    entities = extract_entities(models["ner"], chunks)

    answers = answer_questions(
        models["qa"],
        document.get("questions", []),
        chunks,
    )

    return {
        "document_id": document["document_id"],
        "title": document["title"],
        "text": text,
        "statistics": {
            "character_count": len(text),
            "word_count": len(text.split()),
            "chunk_count": len(chunks),
        },
        "classification": classification,
        "entities": entities,
        "answers": answers,
    }


def save_results(results: list[dict[str, Any]]) -> None:
    """Write the pipeline results to a JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"\nSaved document analysis to: {OUTPUT_FILE}")


def display_summary(results: list[dict[str, Any]]) -> None:
    """Display a concise summary of the pipeline output."""
    print("\n--- Document understanding summary ---")

    for result in results:
        print(f"\n{result['document_id']}: {result['title']}")
        print(f"Words: {result['statistics']['word_count']}")
        print(
            f"Classification: {result['classification']['label']} "
            f"(confidence={result['classification']['confidence']:.4f})"
        )

        print("Entities:")
        for entity in result["entities"]:
            print(
                f"  - {entity['text']} [{entity['label']}] "
                f"confidence={entity['confidence']:.4f}"
            )

        print("Answers:")
        for answer in result["answers"]:
            print(
                f"  - Q: {answer['question']}\n"
                f"    A: {answer['answer']!r}\n"
                f"    Evidence: {answer['evidence']!r}"
            )


def main():
    models = load_pipelines()
    results = []

    for document in DOCUMENTS:
        print(f"\nProcessing {document['document_id']}...")
        results.append(process_document(document, models))

    display_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()
