"""
Language Model Input Preparation

Experiments:
- Inspect special tokens and token IDs.
- Prepare batches of variable-length text sequences.
- Compare padding and attention masks.
- Investigate truncation and maximum sequence lengths.
- Examine the tensors produced for language model inputs.
"""

import torch
from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"

TEXTS = [
    "Subword tokenization helps language models process unfamiliar words.",
    "Attention masks distinguish real tokens from padding.",
    "Transformers operate on numerical representations of text.",
]

MAX_LENGTH = 12


def inspect_special_tokens(tokenizer) -> None:
    """Display the tokenizer's special-token configuration."""
    print("\n--- Special Tokens ---")
    print("Special tokens:", tokenizer.special_tokens_map)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("CLS token:", tokenizer.cls_token, tokenizer.cls_token_id)
    print("SEP token:", tokenizer.sep_token, tokenizer.sep_token_id)
    print("PAD token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("UNK token:", tokenizer.unk_token, tokenizer.unk_token_id)


def inspect_single_sequence(tokenizer, text: str) -> None:
    """Inspect tokenization and special-token insertion."""
    print("\n--- Single Sequence ---")
    print("Original text:", text)

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
    )

    print("Tokens:", tokens)
    print("Token IDs without special tokens:", token_ids)
    print("Tokens with special tokens:", tokenizer.convert_ids_to_tokens(
        encoded["input_ids"]
    ))
    print("Input IDs:", encoded["input_ids"])


def prepare_batch(tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    """Create a padded and truncated batch of model inputs."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )


def inspect_batch(batch: dict[str, torch.Tensor]) -> None:
    """Display model input tensors and their shapes."""
    print("\n--- Prepared Batch ---")

    for name, tensor in batch.items():
        print(f"\n{name}:")
        print(tensor)
        print("Shape:", tuple(tensor.shape))


def compare_padding_masks(
    tokenizer,
    texts: list[str],
) -> None:
    """Compare dynamic padding with fixed-length padding."""
    print("\n--- Padding Comparison ---")

    dynamic_batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    fixed_batch = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    print("Dynamic padding shape:",
          tuple(dynamic_batch["input_ids"].shape))
    print("Fixed padding shape:",
          tuple(fixed_batch["input_ids"].shape))

    print("\nAttention mask:")
    print(dynamic_batch["attention_mask"])

    print("\nPadding mask interpretation:")
    print("1 = real token, 0 = padding token")


def inspect_truncation(tokenizer) -> None:
    """Observe how truncation changes a long sequence."""
    text = (
        "Natural language processing enables computers to analyze "
        "text, represent meaning, identify patterns, and perform "
        "many sophisticated language understanding tasks."
    )

    original = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
    )

    truncated = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    print("\n--- Truncation Analysis ---")
    print("Original token count:", len(original["input_ids"]))
    print("Truncated token count:", len(truncated["input_ids"]))
    print("Maximum length:", MAX_LENGTH)
    print("Truncated tokens:",
          tokenizer.convert_ids_to_tokens(truncated["input_ids"]))


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    inspect_special_tokens(tokenizer)
    inspect_single_sequence(tokenizer, TEXTS[0])

    batch = prepare_batch(tokenizer, TEXTS)
    inspect_batch(batch)

    compare_padding_masks(tokenizer, TEXTS)
    inspect_truncation(tokenizer)


if __name__ == "__main__":
    main()
