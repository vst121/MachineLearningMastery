"""
Pretrained Model Fundamentals

Experiments:
- Load a pretrained Transformer model and its tokenizer.
- Inspect model configuration and architecture.
- Analyze total and trainable parameter counts.
- Examine embedding, encoder, and hidden-state dimensions.
- Compare contextual representations of words in different contexts.
- Measure basic inference time and memory-independent model statistics.
"""

from time import perf_counter

import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "bert-base-uncased"

TEXTS = [
    "The bank approved the loan.",
    "They sat on the bank of the river.",
]


def inspect_configuration(model) -> None:
    """Display important architecture configuration values."""
    config = model.config

    print("\n--- Model Configuration ---")
    print("Model type:", config.model_type)
    print("Hidden size:", config.hidden_size)
    print("Number of layers:", config.num_hidden_layers)
    print("Attention heads:", config.num_attention_heads)
    print("Vocabulary size:", config.vocab_size)
    print("Maximum position embeddings:", config.max_position_embeddings)


def inspect_parameters(model) -> None:
    """Calculate total and trainable parameter counts."""
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\n--- Parameter Statistics ---")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Parameter storage estimate (FP32): "
          f"{total_parameters * 4 / (1024 ** 2):.2f} MiB")


def inspect_architecture(model) -> None:
    """Print a high-level view of the model's components."""
    print("\n--- Model Architecture ---")
    print("Embedding module:")
    print(model.embeddings)

    print("\nNumber of encoder layers:", len(model.encoder.layer))
    print("\nFirst encoder layer:")
    print(model.encoder.layer[0])


def inspect_input_and_hidden_states(
    tokenizer,
    model,
    texts: list[str],
) -> None:
    """Inspect encoded inputs and contextual hidden-state tensors."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    print("\n--- Tokenized Inputs ---")
    for index, text in enumerate(texts):
        token_ids = inputs["input_ids"][index]
        tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())

        print(f"\nText: {text}")
        print("Tokens:", tokens)
        print("Token IDs:", token_ids.tolist())

    model.eval()

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )

    print("\n--- Hidden States ---")
    print("Input IDs shape:", tuple(inputs["input_ids"].shape))
    print("Last hidden state shape:",
          tuple(outputs.last_hidden_state.shape))
    print("Number of hidden-state tensors:", len(outputs.hidden_states))

    for index, hidden_state in enumerate(outputs.hidden_states):
        print(
            f"Hidden state {index:2}: "
            f"shape={tuple(hidden_state.shape)}"
        )

    print("\n--- Contextual Representation Comparison ---")
    tokenized_words = [
        tokenizer.tokenize(text)
        for text in texts
    ]

    for index, tokens in enumerate(tokenized_words):
        print(f"Text {index + 1} tokens:", tokens)

    # Compare the representation of "bank" in both sentences.
    bank_vectors = []

    for index, text in enumerate(texts):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        )

        with torch.inference_mode():
            result = model(**encoded)

        tokens = tokenizer.convert_ids_to_tokens(
            encoded["input_ids"][0].tolist()
        )

        bank_index = tokens.index("bank")
        bank_vector = result.last_hidden_state[0, bank_index, :]
        bank_vectors.append(bank_vector)

    similarity = torch.nn.functional.cosine_similarity(
        bank_vectors[0].unsqueeze(0),
        bank_vectors[1].unsqueeze(0),
    ).item()

    print(f"Cosine similarity between 'bank' representations: "
          f"{similarity:.4f}")


def benchmark_inference(
    tokenizer,
    model,
    text: str,
    repetitions: int = 10,
) -> None:
    """Measure average CPU inference latency after a warm-up pass."""
    inputs = tokenizer(text, return_tensors="pt")
    model.eval()

    with torch.inference_mode():
        model(**inputs)  # Warm-up

        start = perf_counter()
        for _ in range(repetitions):
            model(**inputs)
        elapsed = perf_counter() - start

    print("\n--- Inference Benchmark ---")
    print("Device: CPU")
    print("Repetitions:", repetitions)
    print(f"Average latency: {elapsed / repetitions * 1000:.2f} ms")
    print("Note: Timing depends on hardware, load, and software versions.")


def main() -> None:
    print("Loading pretrained model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    inspect_configuration(model)
    inspect_parameters(model)
    inspect_architecture(model)

    inspect_input_and_hidden_states(tokenizer, model, TEXTS)
    benchmark_inference(tokenizer, model, TEXTS[0])


if __name__ == "__main__":
    main()
