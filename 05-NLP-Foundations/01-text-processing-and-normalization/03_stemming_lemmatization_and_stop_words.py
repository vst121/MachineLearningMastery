"""
Stemming, Lemmatization, and Stop Words

Experiments:
1. Compare Porter and Snowball stemming.
2. Compare stemming with spaCy lemmatization.
3. Inspect stop-word filtering.
4. Demonstrate why negation should often be preserved.
5. Compare token sequences before and after filtering.

Dependencies:
    uv add nltk spacy
    uv run python -m spacy download en_core_web_sm
"""

import nltk
import spacy
from nltk.stem import PorterStemmer, SnowballStemmer
from spacy.language import Language
from spacy.tokens import Doc


def ensure_nltk_resources() -> None:
    """Check that the NLTK stop-word corpus is available."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError as exc:
        raise RuntimeError(
            "Missing NLTK stopwords corpus. Download it explicitly with:\n"
            "    uv run python -c "
            "\"import nltk; nltk.download('stopwords')\""
        ) from exc


def load_spacy_model() -> Language:
    """Load the English model and provide a useful setup message."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "Missing spaCy English model. Install it with:\n"
            "    uv run python -m spacy download en_core_web_sm"
        ) from exc


def experiment_stemming() -> None:
    print("\n=== Experiment 1: Stemming ===")

    words = [
        "connect",
        "connected",
        "connecting",
        "connection",
        "studies",
        "studying",
        "relational",
    ]

    stemmers = {
        "Porter": PorterStemmer(),
        "Snowball": SnowballStemmer("english"),
    }

    for word in words:
        results = {
            name: stemmer.stem(word)
            for name, stemmer in stemmers.items()
        }
        print(f"{word:15} -> {results}")


def experiment_lemmatization(nlp: Language) -> None:
    print("\n=== Experiment 2: Lemmatization ===")

    text = "The students were studying better methods and running experiments."

    doc = nlp(text)

    print(f"Text: {text}\n")
    print(f"{'Token':15} {'POS':10} {'Lemma':15}")

    for token in doc:
        if not token.is_space and not token.is_punct:
            print(f"{token.text:15} {token.pos_:10} {token.lemma_:15}")


def experiment_stop_words() -> None:
    print("\n=== Experiment 3: Stop-Word Filtering ===")

    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
    text = "The model is learning from the data and improving its predictions."

    tokens = text.split()
    filtered = [
        token for token in tokens
        if token.lower().strip(".,!?") not in stop_words
    ]

    print(f"Original: {tokens}")
    print(f"Filtered: {filtered}")
    print(f"Stop-word count: {sum(
        token.lower().strip('.,!?') in stop_words
        for token in tokens
    )}")


def experiment_negation(nlp: Language) -> None:
    print("\n=== Experiment 4: Negation Preservation ===")

    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))

    examples = [
        "The movie was good.",
        "The movie was not good.",
        "The movie was never good.",
    ]

    for text in examples:
        tokens = [token.text for token in nlp(text) if not token.is_space]

        naive_filtered = [
            token for token in tokens
            if token.lower() not in stop_words
        ]

        protected_filtered = [
            token for token in tokens
            if token.lower() not in stop_words
            or token.lower() in {"no", "not", "nor", "never"}
        ]

        print(f"\nOriginal:          {text}")
        print(f"Naive filtering:   {naive_filtered}")
        print(f"Protected filtering: {protected_filtered}")


def experiment_sequence_lengths(nlp: Language) -> None:
    print("\n=== Experiment 5: Sequence Lengths ===")

    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
    protected_negations = {"no", "not", "nor", "never"}

    text = "The product is not useful for the intended purpose."
    doc: Doc = nlp(text)

    original = [
        token.text for token in doc
        if not token.is_space and not token.is_punct
    ]

    filtered = [
        token.text for token in doc
        if not token.is_space
        and not token.is_punct
        and (
            token.lower_ not in stop_words
            or token.lower_ in protected_negations
        )
    ]

    print(f"Original tokens: {original}")
    print(f"Filtered tokens: {filtered}")
    print(f"Original length: {len(original)}")
    print(f"Filtered length: {len(filtered)}")


def main() -> None:
    ensure_nltk_resources()
    nlp = load_spacy_model()

    experiment_stemming()
    experiment_lemmatization(nlp)
    experiment_stop_words()
    experiment_negation(nlp)
    experiment_sequence_lengths(nlp)


if __name__ == "__main__":
    main()
