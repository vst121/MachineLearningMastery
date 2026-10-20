"""
Subword Tokenization Fundamentals

Experiments:
- Compare word-level, character-level, and subword tokenization.
- Build a vocabulary and map tokens to integer IDs.
- Examine out-of-vocabulary behavior.
- Investigate how subword units handle unseen words.
"""

import re
from collections import Counter


def word_tokenize(text: str) -> list[str]:
    """Split text into words and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def character_tokenize(text: str) -> list[str]:
    """Represent text as individual characters."""
    return list(text.lower())


def build_vocabulary(
    tokenized_texts: list[list[str]],
    min_frequency: int = 1,
) -> dict[str, int]:
    """Build a token-to-ID vocabulary ordered by frequency."""
    frequencies = Counter(
        token for tokens in tokenized_texts for token in tokens
    )

    vocabulary = {
        token: token_id
        for token_id, (token, frequency) in enumerate(
            sorted(
                (
                    (token, frequency)
                    for token, frequency in frequencies.items()
                    if frequency >= min_frequency
                ),
                key=lambda item: (-item[1], item[0]),
            ),
            start=2,
        )
    }

    return {"<PAD>": 0, "<UNK>": 1, **vocabulary}


def encode_tokens(
    tokens: list[str],
    vocabulary: dict[str, int],
) -> list[int]:
    """Convert tokens into IDs, using <UNK> for unseen tokens."""
    unknown_id = vocabulary["<UNK>"]
    return [vocabulary.get(token, unknown_id) for token in tokens]


def demonstrate_word_level(
    training_texts: list[str],
    test_text: str,
) -> None:
    """Build and evaluate a word-level vocabulary."""
    training_tokens = [word_tokenize(text) for text in training_texts]
    vocabulary = build_vocabulary(training_tokens)

    test_tokens = word_tokenize(test_text)
    encoded = encode_tokens(test_tokens, vocabulary)

    print("\n--- Word-Level Tokenization ---")
    print("Tokens:", test_tokens)
    print("Vocabulary size:", len(vocabulary))
    print("Token IDs:", encoded)
    print(
        "Unknown tokens:",
        [token for token in test_tokens if token not in vocabulary],
    )


def demonstrate_character_level(
    training_texts: list[str],
    test_text: str,
) -> None:
    """Build and evaluate a character-level vocabulary."""
    training_tokens = [character_tokenize(text) for text in training_texts]
    vocabulary = build_vocabulary(training_tokens)

    test_tokens = character_tokenize(test_text)
    encoded = encode_tokens(test_tokens, vocabulary)

    print("\n--- Character-Level Tokenization ---")
    print("Tokens:", test_tokens)
    print("Vocabulary size:", len(vocabulary))
    print("Token IDs:", encoded)
    print(
        "Unknown characters:",
        [token for token in test_tokens if token not in vocabulary],
    )


def demonstrate_subword_segmentation(text: str) -> None:
    """
    Illustrate subword segmentation with a small hand-built vocabulary.

    This is a greedy longest-match demonstration, not a full BPE,
    WordPiece, or Unigram tokenizer.
    """
    subword_vocabulary = {
        "play",
        "##ing",
        "##ed",
        "un",
        "##believ",
        "##able",
        "believ",
        "able",
    }

    def segment_word(word: str) -> list[str]:
        pieces = []
        position = 0

        while position < len(word):
            match = None

            for end in range(len(word), position, -1):
                candidate = word[position:end]
                if position > 0:
                    candidate = "##" + candidate

                if candidate in subword_vocabulary:
                    match = (candidate, end)
                    break

            if match is None:
                return ["<UNK>"]

            piece, position = match
            pieces.append(piece)

        return pieces

    print("\n--- Illustrative Subword Segmentation ---")

    for word in text.lower().split():
        pieces = segment_word(word)
        print(f"{word:15} -> {pieces}")


def main() -> None:
    training_texts = [
        "I enjoy playing games.",
        "She enjoys playing music.",
        "They played together.",
        "This is unbelievable.",
    ]

    unseen_text = "We enjoy replaying games."

    print("Training examples:")
    for text in training_texts:
        print(" ", text)

    print("\nTest example:", unseen_text)

    demonstrate_word_level(training_texts, unseen_text)
    demonstrate_character_level(training_texts, unseen_text)

    demonstrate_subword_segmentation(
        "playing played unbelievable replaying"
    )


if __name__ == "__main__":
    main()
