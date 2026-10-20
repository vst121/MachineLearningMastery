"""
Explore sequence representation and word embeddings.

This experiment covers:
1. Building a vocabulary from a small text corpus.
2. Converting tokens into integer IDs.
3. Representing documents as ordered token sequences.
4. Padding and truncating variable-length sequences.
5. Creating and inspecting learnable embedding vectors.
6. Understanding the difference between sequence and Bag of Words representations.
"""

from collections import Counter
import random

import numpy as np
import torch
from torch import nn


SEED = 42
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def tokenize(text: str) -> list[str]:
    """Tokenize a simple English sentence into lowercase word tokens."""
    return text.lower().split()


def build_vocabulary(
    texts: list[str],
    min_frequency: int = 1,
) -> dict[str, int]:
    """Build a deterministic vocabulary with padding and unknown tokens."""
    token_counts = Counter(
        token
        for text in texts
        for token in tokenize(text)
    )

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    vocabulary_tokens = sorted(
        token
        for token, count in token_counts.items()
        if count >= min_frequency and token not in vocabulary
    )

    for token in vocabulary_tokens:
        vocabulary[token] = len(vocabulary)

    return vocabulary


def encode_text(
    text: str,
    vocabulary: dict[str, int],
) -> list[int]:
    """Convert a tokenized text into token IDs."""
    unknown_id = vocabulary[UNK_TOKEN]

    return [
        vocabulary.get(token, unknown_id)
        for token in tokenize(text)
    ]


def pad_sequences(
    sequences: list[list[int]],
    max_length: int | None = None,
) -> torch.Tensor:
    """
    Pad or truncate sequences to a common length.

    Padding uses PAD_TOKEN's ID, which is zero.
    Truncation retains the first max_length tokens.
    """
    if not sequences:
        raise ValueError("At least one sequence is required.")

    if any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("Empty token sequences are not supported.")

    if max_length is None:
        max_length = max(len(sequence) for sequence in sequences)

    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    padded = []

    for sequence in sequences:
        truncated = sequence[:max_length]
        padded_sequence = truncated + [0] * (
            max_length - len(truncated)
        )
        padded.append(padded_sequence)

    return torch.tensor(padded, dtype=torch.long)


def display_sequence(
    sequence: list[int],
    vocabulary: dict[str, int],
) -> list[str]:
    """Convert token IDs back into their corresponding tokens."""
    inverse_vocabulary = {
        token_id: token
        for token, token_id in vocabulary.items()
    }

    return [
        inverse_vocabulary.get(token_id, UNK_TOKEN)
        for token_id in sequence
    ]


def main() -> None:
    texts = [
        "the team won the match",
        "the player scored a goal",
        "the computer processed the data",
        "the software improved the system",
    ]

    print("=" * 72)
    print("1. BUILD A VOCABULARY")
    print("=" * 72)

    vocabulary = build_vocabulary(texts)

    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Padding token ID: {vocabulary[PAD_TOKEN]}")
    print(f"Unknown token ID: {vocabulary[UNK_TOKEN]}")
    print("\nVocabulary:")

    for token, token_id in vocabulary.items():
        print(f"  {token:>12}: {token_id}")

    print("\n" + "=" * 72)
    print("2. ENCODE TEXT AS SEQUENCES")
    print("=" * 72)

    sequences = [
        encode_text(text, vocabulary)
        for text in texts
    ]

    for text, sequence in zip(texts, sequences):
        print(f"Text:     {text}")
        print(f"Token IDs: {sequence}")
        print(f"Recovered: {display_sequence(sequence, vocabulary)}")
        print()

    print("=" * 72)
    print("3. HANDLE UNKNOWN TOKENS")
    print("=" * 72)

    new_text = "the player explored robotics"
    encoded_new_text = encode_text(new_text, vocabulary)

    print(f"New text: {new_text}")
    print(f"Encoded:  {encoded_new_text}")
    print(
        "Unknown token IDs:",
        [
            token_id
            for token_id in encoded_new_text
            if token_id == vocabulary[UNK_TOKEN]
        ],
    )

    print("\n" + "=" * 72)
    print("4. PAD VARIABLE-LENGTH SEQUENCES")
    print("=" * 72)

    variable_length_texts = [
        "the team won",
        "the player scored a goal",
        "software improved the system today",
    ]

    variable_sequences = [
        encode_text(text, vocabulary)
        for text in variable_length_texts
    ]

    padded_sequences = pad_sequences(variable_sequences)

    for text, sequence, padded in zip(
        variable_length_texts,
        variable_sequences,
        padded_sequences.tolist(),
    ):
        print(f"Text:       {text}")
        print(f"Original:   {sequence}")
        print(f"Padded:     {padded}")
        print()

    print(f"Padded tensor shape: {tuple(padded_sequences.shape)}")
    print(f"Tensor dtype: {padded_sequences.dtype}")

    print("\n" + "=" * 72)
    print("5. TRUNCATE LONGER SEQUENCES")
    print("=" * 72)

    max_length = 4
    truncated_sequences = pad_sequences(
        variable_sequences,
        max_length=max_length,
    )

    for text, encoded, truncated in zip(
        variable_length_texts,
        variable_sequences,
        truncated_sequences.tolist(),
    ):
        print(f"Text:       {text}")
        print(f"Encoded:    {encoded}")
        print(f"Truncated:  {truncated}")
        print()

    print("=" * 72)
    print("6. CREATE A LEARNABLE EMBEDDING LAYER")
    print("=" * 72)

    embedding_dim = 8

    embedding_layer = nn.Embedding(
        num_embeddings=len(vocabulary),
        embedding_dim=embedding_dim,
        padding_idx=vocabulary[PAD_TOKEN],
    )

    embedded_sequences = embedding_layer(padded_sequences)

    print(f"Embedding vocabulary size: {len(vocabulary)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Token ID tensor shape: {tuple(padded_sequences.shape)}")
    print(
        "Embedding tensor shape:",
        tuple(embedded_sequences.shape),
    )
    print(
        "Expected shape: (batch_size, sequence_length, embedding_dim)"
    )

    first_sequence = padded_sequences[0]
    first_embedding = embedding_layer(first_sequence)

    print("\nFirst document:")
    print(f"Token IDs: {first_sequence.tolist()}")
    print(f"Embedding shape: {tuple(first_embedding.shape)}")
    print("First token embedding:")
    print(first_embedding[0].detach().numpy().round(3))

    padding_embedding = embedding_layer(
        torch.tensor(vocabulary[PAD_TOKEN])
    )

    print("\nPadding embedding:")
    print(padding_embedding.detach().numpy().round(3))

    print("\n" + "=" * 72)
    print("7. COMPARE SEQUENCE AND BAG OF WORDS")
    print("=" * 72)

    ordered_texts = [
        "the team won the match",
        "the match won the team",
    ]

    ordered_sequences = [
        encode_text(text, vocabulary)
        for text in ordered_texts
    ]

    print("Document A:", ordered_texts[0])
    print("Document B:", ordered_texts[1])
    print(f"Sequence A: {ordered_sequences[0]}")
    print(f"Sequence B: {ordered_sequences[1]}")
    print(
        "Sequences equal:",
        ordered_sequences[0] == ordered_sequences[1],
    )

    def bag_of_words(sequence: list[int]) -> Counter[int]:
        """Count token IDs, excluding padding and unknown tokens."""
        return Counter(
            token_id
            for token_id in sequence
            if token_id > vocabulary[UNK_TOKEN]
        )

    bow_a = bag_of_words(ordered_sequences[0])
    bow_b = bag_of_words(ordered_sequences[1])

    print("Bag of Words A:", dict(sorted(bow_a.items())))
    print("Bag of Words B:", dict(sorted(bow_b.items())))
    print("Bag of Words equal:", bow_a == bow_b)

    print("\n" + "=" * 72)
    print("8. IMPORTANT OBSERVATIONS")
    print("=" * 72)

    print("- Token IDs preserve the order of words in each document.")
    print("- Unknown tokens map to a dedicated UNK token ID.")
    print("- Padding makes sequences in a batch the same length.")
    print("- Truncation can remove information from longer documents.")
    print("- Embeddings map token IDs to dense, learnable vectors.")
    print("- Padding positions should not be treated as meaningful words.")
    print("- Bag of Words discards word order; sequence representations retain it.")
    print("- The embeddings are randomly initialized and are not pretrained.")


if __name__ == "__main__":
    main()
