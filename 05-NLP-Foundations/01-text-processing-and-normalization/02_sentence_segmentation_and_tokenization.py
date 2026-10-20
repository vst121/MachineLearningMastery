"""
Sentence Segmentation and Tokenization

Experiments:
1. Sentence segmentation using simple rules.
2. Sentence segmentation using NLTK.
3. Word tokenization using different strategies.
4. Examine punctuation, contractions, and abbreviations.
5. Compare token counts and information preservation.

Dependencies:
    uv add nltk
"""

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def ensure_nltk_resources() -> None:
    """Check required NLTK resources without downloading automatically."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    missing = []

    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(package_name)

    if missing:
        packages = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            f"Missing NLTK resources: {packages}\n"
            "Download them explicitly in your environment with:\n"
            f"    uv run python -c \"import nltk; "
            f"nltk.download('{missing[0]}')\""
            + (
                f"\nRepeat for: {', '.join(missing[1:])}"
                if len(missing) > 1
                else ""
            )
        )


def simple_sentence_split(text: str) -> list[str]:
    """Split sentences using punctuation followed by whitespace.

    This is a baseline, not a linguistically complete solution.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part]


def whitespace_tokenize(text: str) -> list[str]:
    """Split text on whitespace only."""
    return text.split()


def regex_tokenize(text: str) -> list[str]:
    """Extract word-like sequences and individual punctuation marks."""
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text, flags=re.UNICODE)


def print_tokens(label: str, tokens: list[str]) -> None:
    print(f"{label} ({len(tokens)} tokens):")
    print(tokens)


def experiment_sentence_segmentation() -> None:
    print("\n=== Experiment 1: Sentence Segmentation ===")

    text = (
        "Dr. Smith studies NLP. "
        "The models work well! "
        "Do they handle abbreviations, e.g. correctly? "
        "Yes, they can."
    )

    print(f"\nOriginal text:\n{text}")

    print("\nSimple rule-based segmentation:")
    for index, sentence in enumerate(simple_sentence_split(text), start=1):
        print(f"{index}. {sentence}")

    ensure_nltk_resources()

    print("\nNLTK sentence segmentation:")
    for index, sentence in enumerate(sent_tokenize(text), start=1):
        print(f"{index}. {sentence}")


def experiment_tokenization() -> None:
    print("\n=== Experiment 2: Word Tokenization ===")

    text = "NLP isn't just keywords: it's about meaning, context, and language!"

    print(f"\nOriginal text:\n{text}")

    print_tokens("Whitespace tokenization", whitespace_tokenize(text))
    print_tokens("Regex tokenization", regex_tokenize(text))

    ensure_nltk_resources()
    print_tokens("NLTK word tokenization", word_tokenize(text))


def experiment_edge_cases() -> None:
    print("\n=== Experiment 3: Tokenization Edge Cases ===")

    examples = [
        "The model achieved 95.7% accuracy.",
        "E-mail and email may be treated differently.",
        "Don't ignore negation!",
        "Natural-language-processing is a compound expression.",
        "Das ist ein deutscher Satz.",
    ]

    for text in examples:
        print(f"\nText: {text}")
        print(f"Whitespace: {whitespace_tokenize(text)}")
        print(f"Regex:      {regex_tokenize(text)}")


def experiment_token_counts() -> None:
    print("\n=== Experiment 4: Token Count Comparison ===")

    text = "Text processing, tokenization, and NLP."

    whitespace_tokens = whitespace_tokenize(text)
    regex_tokens = regex_tokenize(text)

    print(f"Text: {text}")
    print(f"Whitespace token count: {len(whitespace_tokens)}")
    print(f"Regex token count:       {len(regex_tokens)}")

    print("\nToken counts depend on the tokenization strategy.")
    print("A larger token count does not automatically mean better tokenization.")


def main() -> None:
    experiment_sentence_segmentation()
    experiment_tokenization()
    experiment_edge_cases()
    experiment_token_counts()


if __name__ == "__main__":
    main()
