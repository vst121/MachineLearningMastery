"""
Question Answering and Document Grounding

Experiments:
- Run extractive question answering on supplied documents.
- Inspect predicted answer spans, offsets, and confidence scores.
- Compare answers across different contexts.
- Check whether predicted answers are supported by their source text.
- Evaluate answer quality against manually specified reference answers.
"""

from typing import Any

from transformers import pipeline


MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"

DOCUMENTS = [
    {
        "id": "company_profile",
        "context": (
            "Northstar Analytics was founded in 2018 in Berlin. "
            "The company develops software for business intelligence "
            "and employs 240 people. Its chief executive officer is "
            "Maya Schneider. The company opened a research office in "
            "Munich in 2023."
        ),
        "questions": [
            {
                "question": "When was Northstar Analytics founded?",
                "reference_answer": "2018",
            },
            {
                "question": "Where was Northstar Analytics founded?",
                "reference_answer": "Berlin",
            },
            {
                "question": "How many people does the company employ?",
                "reference_answer": "240",
            },
            {
                "question": "Who is the chief executive officer?",
                "reference_answer": "Maya Schneider",
            },
            {
                "question": "Which city has the research office?",
                "reference_answer": "Munich",
            },
        ],
    },
    {
        "id": "technical_report",
        "context": (
            "The Atlas document processing system was released in 2024. "
            "It uses a Python backend and PostgreSQL for persistent data. "
            "The first version supports PDF extraction and document "
            "classification. The engineering team plans to add table "
            "extraction in a later release."
        ),
        "questions": [
            {
                "question": "When was Atlas released?",
                "reference_answer": "2024",
            },
            {
                "question": "Which programming language does its backend use?",
                "reference_answer": "Python",
            },
            {
                "question": "Which database does the system use?",
                "reference_answer": "PostgreSQL",
            },
            {
                "question": "What document formats does the system process?",
                "reference_answer": "PDF",
            },
        ],
    },
]


def load_qa_pipeline():
    """Load a pretrained extractive question-answering model."""
    print(f"Loading question-answering model: {MODEL_NAME}")

    return pipeline(
        task="question-answering",
        model=MODEL_NAME,
    )


def answer_question(
    qa,
    question: str,
    context: str,
) -> dict[str, Any]:
    """Predict an answer span and verify its location in the context."""
    prediction = qa(
        question=question,
        context=context,
        handle_impossible_answer=True,
    )

    answer = prediction.get("answer", "").strip()
    start = int(prediction.get("start", 0))
    end = int(prediction.get("end", 0))
    score = float(prediction.get("score", 0.0))

    valid_offsets = (
        0 <= start <= end <= len(context)
        and context[start:end] == answer
    )

    return {
        "question": question,
        "answer": answer,
        "confidence": score,
        "start": start,
        "end": end,
        "evidence": context[start:end] if valid_offsets else "",
        "offsets_valid": valid_offsets,
        "answer_in_context": bool(answer) and answer.casefold() in context.casefold(),
    }


def normalize_answer(answer: str) -> str:
    """Normalize basic whitespace and case for simple answer comparison."""
    return " ".join(answer.casefold().split()).strip(" .,!?:;")


def evaluate_answer(
    predicted: str,
    reference: str,
) -> dict[str, float]:
    """
    Calculate exact match and token-overlap F1.

    These simple metrics are useful for the exercise but do not capture
    semantic equivalence or factual correctness in every situation.
    """
    predicted_normalized = normalize_answer(predicted)
    reference_normalized = normalize_answer(reference)

    exact_match = float(predicted_normalized == reference_normalized)

    predicted_tokens = predicted_normalized.split()
    reference_tokens = reference_normalized.split()

    if not predicted_tokens or not reference_tokens:
        return {"exact_match": exact_match, "token_f1": 0.0}

    common = {}
    for token in predicted_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0
    for token in reference_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)

    token_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "exact_match": exact_match,
        "token_f1": token_f1,
    }


def run_document_experiment(qa, document: dict[str, Any]) -> list[dict[str, Any]]:
    """Answer and evaluate all questions associated with one document."""
    context = document["context"]
    results = []

    print(f"\n{'=' * 80}")
    print(f"Document: {document['id']}")
    print(f"Context: {context}")

    for item in document["questions"]:
        question = item["question"]
        reference = item["reference_answer"]

        prediction = answer_question(qa, question, context)
        metrics = evaluate_answer(prediction["answer"], reference)

        result = {
            "document_id": document["id"],
            "reference_answer": reference,
            **prediction,
            **metrics,
        }
        results.append(result)

        print(f"\nQuestion: {question}")
        print(f"Predicted answer: {prediction['answer']!r}")
        print(f"Reference answer: {reference!r}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        print(
            f"Evidence offsets: "
            f"[{prediction['start']}, {prediction['end']})"
        )
        print(f"Evidence: {prediction['evidence']!r}")
        print(f"Offsets valid: {prediction['offsets_valid']}")
        print(f"Answer appears in context: {prediction['answer_in_context']}")
        print(f"Exact match: {metrics['exact_match']:.0f}")
        print(f"Token F1: {metrics['token_f1']:.4f}")

    return results


def summarize_results(results: list[dict[str, Any]]) -> None:
    """Report aggregate answer metrics and basic grounding statistics."""
    if not results:
        print("No questions evaluated.")
        return

    count = len(results)
    exact_match = sum(item["exact_match"] for item in results) / count
    token_f1 = sum(item["token_f1"] for item in results) / count
    valid_offsets = sum(item["offsets_valid"] for item in results) / count
    context_matches = sum(item["answer_in_context"] for item in results) / count

    print(f"\n{'=' * 80}")
    print("Overall evaluation")
    print(f"Questions evaluated: {count}")
    print(f"Exact match: {exact_match:.4f}")
    print(f"Token F1: {token_f1:.4f}")
    print(f"Valid evidence offsets: {valid_offsets:.4f}")
    print(f"Answers appearing in context: {context_matches:.4f}")


def main():
    qa = load_qa_pipeline()
    all_results = []

    for document in DOCUMENTS:
        all_results.extend(run_document_experiment(qa, document))

    summarize_results(all_results)


if __name__ == "__main__":
    main()
