"""
Tokenizer Algorithms Comparison

Experiments:
- Train BPE, WordPiece, and Unigram tokenizers on the same corpus.
- Compare subword segmentation for familiar and unseen words.
- Examine vocabulary size and token sequence length.
- Understand the different principles behind the three algorithms.
"""

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


CORPUS = [
    "I enjoy playing games.",
    "She enjoys playing music.",
    "They played together.",
    "This is unbelievable.",
    "We are learning natural language processing.",
    "Language models learn patterns from text.",
    "Playing and replaying can reveal word structure.",
    "The model processes words into smaller pieces.",
]

TEST_WORDS = [
    "playing",
    "replaying",
    "unbelievable",
    "tokenization",
    "misunderstanding",
]

VOCAB_SIZE = 100
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]


def train_bpe() -> Tokenizer:
    """Train a Byte Pair Encoding tokenizer."""
    tokenizer = Tokenizer(
        models.BPE(unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(CORPUS, trainer=trainer)
    return tokenizer


def train_wordpiece() -> Tokenizer:
    """Train a WordPiece tokenizer."""
    tokenizer = Tokenizer(
        models.WordPiece(
            unk_token="[UNK]",
            continuing_subword_prefix="##",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordPieceTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix="##",
    )
    tokenizer.train_from_iterator(CORPUS, trainer=trainer)
    return tokenizer


def train_unigram() -> Tokenizer:
    """Train a Unigram tokenizer."""
    tokenizer = Tokenizer(
        models.Unigram()
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )

    trainer = trainers.UnigramTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        unk_token="[UNK]",
    )
    tokenizer.train_from_iterator(CORPUS, trainer=trainer)
    return tokenizer


def inspect_tokenizer(name: str, tokenizer: Tokenizer) -> None:
    """Display segmentation and vocabulary statistics."""
    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")
    print("Vocabulary size:", tokenizer.get_vocab_size())

    for word in TEST_WORDS:
        encoding = tokenizer.encode(word)
        print(
            f"{word:22} -> {encoding.tokens!s:40} "
            f"IDs={encoding.ids}"
        )


def compare_sequence_lengths(
    tokenizers: dict[str, Tokenizer],
    text: str,
) -> None:
    """Compare token counts for the same sentence."""
    print(f"\nSequence-length comparison:")
    print("Text:", text)

    for name, tokenizer in tokenizers.items():
        encoding = tokenizer.encode(text)
        print(f"{name:12}: {len(encoding.tokens):2} tokens")


def main() -> None:
    tokenizers = {
        "BPE": train_bpe(),
        "WordPiece": train_wordpiece(),
        "Unigram": train_unigram(),
    }

    for name, tokenizer in tokenizers.items():
        inspect_tokenizer(name, tokenizer)

    compare_sequence_lengths(
        tokenizers,
        "We are learning tokenization and language modeling.",
    )


if __name__ == "__main__":
    main()
