"""
Contextual Embeddings and Model Analysis

Experiments:
- Extract contextual token and sentence representations from a pretrained model.
- Compare representations of the same word in different contexts.
- Compare contextual embeddings with static word embeddings.
- Measure cosine similarity between token representations.
- Inspect parameter counts, tensor dimensions, and inference latency.
"""

from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "google-bert/bert-base-uncased"

SENTENCES = [
    "The bank approved the loan.",
    "They sat on the bank of the river.",
]

STATIC_WORD_VECTORS = {
    "bank": torch.tensor([1.0, 0.0, 0.5, 0.2]),
    "loan": torch.tensor([0.9, 0.1, 0.4, 0.3]),
    "river": torch.tensor([0.1, 0.9, 0.2, 0.5]),
}


def cosine_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    """Calculate cosine similarity between two vectors."""
    return F.cosine_similarity(
        first.reshape(1, -1),
        second.reshape(1, -1),
    ).item()


def extract_token_representation(
    tokenizer,
    model,
    text: str,
    target_token: str,
) -> torch.Tensor:
    """Extract the contextual representation of a single token."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0].tolist()
    )

    matching_positions = [
        index
        for index, token in enumerate(tokens)
        if token == target_token
    ]

    if len(matching_positions) != 1:
        raise ValueError(
            f"Expected exactly one {target_token!r} token, "
            f"found {len(matching_positions)} in {tokens}"
        )

    return outputs.last_hidden_state[
        0, matching_positions[0], :
    ].cpu()


def analyze_contextual_embeddings(tokenizer, model) -> None:
    """Compare a word's contextual representation across sentences."""
    print("\n--- Contextual Embedding Analysis ---")

    representations = []

    for sentence in SENTENCES:
        vector = extract_token_representation(
            tokenizer,
            model,
            sentence,
            target_token="bank",
        )
        representations.append(vector)

        print("Sentence:", sentence)
        print("Representation shape:", tuple(vector.shape))
        print("First 8 dimensions:", vector[:8].tolist())

    similarity = cosine_similarity(
        representations[0],
        representations[1],
    )

    print(
        "\nCosine similarity for 'bank' across contexts:",
        f"{similarity:.4f}",
    )


def compare_static_embeddings() -> None:
    """Demonstrate the context-independent nature of static vectors."""
    print("\n--- Static Embedding Demonstration ---")

    bank_vector = STATIC_WORD_VECTORS["bank"]

    for context in [
        "The bank approved the loan.",
        "They sat on the bank of the river.",
    ]:
        print("Context:", context)
        print("Static vector for 'bank':", bank_vector.tolist())

    loan_similarity = cosine_similarity(
        bank_vector,
        STATIC_WORD_VECTORS["loan"],
    )
    river_similarity = cosine_similarity(
        bank_vector,
        STATIC_WORD_VECTORS["river"],
    )

    print(f"Similarity with 'loan': {loan_similarity:.4f}")
    print(f"Similarity with 'river': {river_similarity:.4f}")
    print(
        "Note: these vectors are synthetic examples, "
        "not embeddings learned from a corpus."
    )


def analyze_model(model) -> None:
    """Inspect model parameters and representation dimensions."""
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\n--- Model Analysis ---")
    print("Model type:", model.config.model_type)
    print("Hidden size:", model.config.hidden_size)
    print("Encoder layers:", model.config.num_hidden_layers)
    print("Attention heads:", model.config.num_attention_heads)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")


def benchmark_inference(tokenizer, model, text: str) -> None:
    """Measure basic inference latency."""
    inputs = tokenizer(text, return_tensors="pt")
    model.eval()

    with torch.inference_mode():
        model(**inputs)  # Warm-up

        start = perf_counter()
        repetitions = 10

        for _ in range(repetitions):
            model(**inputs)

        elapsed = perf_counter() - start

    print("\n--- Inference Benchmark ---")
    print("Device: CPU")
    print("Repetitions:", repetitions)
    print(f"Average latency: {elapsed / repetitions * 1000:.2f} ms")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    analyze_model(model)
    analyze_contextual_embeddings(tokenizer, model)
    compare_static_embeddings()
    benchmark_inference(tokenizer, model, SENTENCES[0])


if __name__ == "__main__":
    main()
