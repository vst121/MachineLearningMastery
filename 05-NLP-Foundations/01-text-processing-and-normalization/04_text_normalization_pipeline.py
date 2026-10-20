"""
Text Normalization Pipeline

Experiments:
1. Build a reusable text preprocessing pipeline.
2. Configure Unicode normalization and HTML cleaning.
3. Apply sentence segmentation and tokenization.
4. Optionally apply lemmatization and stop-word filtering.
5. Compare pipeline configurations.
6. Test reproducibility and important invariants.

Dependencies:
    uv add nltk spacy
    uv run python -m spacy download en_core_web_sm

NLTK resources required:
    punkt
    punkt_tab
    stopwords
"""

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import re
import unicodedata

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from spacy.language import Language


class HTMLTextExtractor(HTMLParser):
    """Extract text content from HTML markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.parts)


@dataclass(frozen=True)
class NormalizationConfig:
    """Controls the transformations applied by the pipeline."""

    unicode_form: str = "NFC"
    clean_html: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = False
    lemmatize: bool = False
    remove_stop_words: bool = False
    remove_punctuation: bool = False
    preserve_negation: bool = True


class TextNormalizationPipeline:
    """Configurable preprocessing pipeline for English text."""

    NEGATIONS = {"no", "not", "nor", "never"}

    def __init__(
        self,
        config: NormalizationConfig | None = None,
        nlp: Language | None = None,
    ) -> None:
        self.config = config or NormalizationConfig()
        self.nlp = nlp
        self.stop_words = set(stopwords.words("english"))

        if self.config.lemmatize and self.nlp is None:
            raise ValueError(
                "An initialized spaCy model is required for lemmatization."
            )

        if self.config.unicode_form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError(
                f"Unsupported Unicode form: {self.config.unicode_form}"
            )

    def clean(self, text: str) -> str:
        """Apply character-level cleaning."""
        text = unicodedata.normalize(self.config.unicode_form, text)

        if self.config.clean_html:
            text = unescape(text)
            parser = HTMLTextExtractor()
            parser.feed(text)
            parser.close()
            text = parser.get_text()

        if self.config.normalize_whitespace:
            text = " ".join(text.split())

        return text

    def process_tokens(self, text: str) -> list[str]:
        """Tokenize text and apply configured token-level transformations."""
        if self.nlp is not None:
            doc = self.nlp(text)
            tokens = []

            for token in doc:
                if token.is_space:
                    continue

                if self.config.remove_punctuation and token.is_punct:
                    continue

                value = token.lemma_ if self.config.lemmatize else token.text

                if self.config.lowercase:
                    value = value.lower()

                if self.config.remove_stop_words:
                    stop_word = value.lower()

                    if stop_word in self.stop_words:
                        if not (
                            self.config.preserve_negation
                            and stop_word in self.NEGATIONS
                        ):
                            continue

                tokens.append(value)

            return tokens

        # Standard-library fallback when lemmatization is disabled.
        tokens = re.findall(
            r"\w+(?:['’]\w+)*|[^\w\s]",
            text,
            flags=re.UNICODE,
        )

        results = []

        for token in tokens:
            if self.config.remove_punctuation and re.fullmatch(
                r"[^\w\s]+", token
            ):
                continue

            value = token.lower() if self.config.lowercase else token

            if self.config.remove_stop_words:
                stop_word = value.lower()

                if stop_word in self.stop_words and not (
                    self.config.preserve_negation
                    and stop_word in self.NEGATIONS
                ):
                    continue

            results.append(value)

        return results

    def transform(self, text: str) -> list[str]:
        """Return normalized tokens for a document."""
        cleaned = self.clean(text)
        return self.process_tokens(cleaned)

    def transform_sentences(self, text: str) -> list[list[str]]:
        """Normalize each sentence independently."""
        cleaned = self.clean(text)
        sentences = sent_tokenize(cleaned)

        return [
            self.process_tokens(sentence)
            for sentence in sentences
        ]


def experiment_basic_pipeline(nlp: Language) -> None:
    print("\n=== Experiment 1: Basic Pipeline ===")

    config = NormalizationConfig()
    pipeline = TextNormalizationPipeline(config, nlp)

    text = (
        "<p>Cafe\u0301   is useful &amp; practical.</p> "
        "<p>Text preprocessing matters.</p>"
    )

    print(f"Original: {text}")
    print(f"Cleaned:  {pipeline.clean(text)}")
    print(f"Tokens:   {pipeline.transform(text)}")


def experiment_configurations(nlp: Language) -> None:
    print("\n=== Experiment 2: Configuration Comparison ===")

    text = "The product is NOT useful for the intended purpose."

    configurations = {
        "Basic": NormalizationConfig(),
        "Lowercase": NormalizationConfig(lowercase=True),
        "Lemmatized": NormalizationConfig(
            lowercase=True,
            lemmatize=True,
        ),
        "Filtered": NormalizationConfig(
            lowercase=True,
            lemmatize=True,
            remove_stop_words=True,
            remove_punctuation=True,
        ),
    }

    for name, config in configurations.items():
        pipeline = TextNormalizationPipeline(config, nlp)
        print(f"{name:12}: {pipeline.transform(text)}")


def experiment_negation(nlp: Language) -> None:
    print("\n=== Experiment 3: Negation Preservation ===")

    text = "The service is not good."

    configurations = {
        "Preserve negation": NormalizationConfig(
            lowercase=True,
            remove_stop_words=True,
            preserve_negation=True,
        ),
        "Remove negation": NormalizationConfig(
            lowercase=True,
            remove_stop_words=True,
            preserve_negation=False,
        ),
    }

    for name, config in configurations.items():
        pipeline = TextNormalizationPipeline(config, nlp)
        print(f"{name:20}: {pipeline.transform(text)}")


def experiment_sentences(nlp: Language) -> None:
    print("\n=== Experiment 4: Sentence-Level Processing ===")

    text = (
        "NLP is useful. "
        "It supports search and classification! "
        "Careful preprocessing matters."
    )

    pipeline = TextNormalizationPipeline(
        NormalizationConfig(lowercase=True),
        nlp,
    )

    for index, tokens in enumerate(
        pipeline.transform_sentences(text),
        start=1,
    ):
        print(f"Sentence {index}: {tokens}")


def experiment_invariants(nlp: Language) -> None:
    print("\n=== Experiment 5: Reproducibility and Invariants ===")

    pipeline = TextNormalizationPipeline(
        NormalizationConfig(
            lowercase=True,
            remove_stop_words=True,
            preserve_negation=True,
        ),
        nlp,
    )

    text = "The product is not useful."

    first = pipeline.transform(text)
    second = pipeline.transform(text)

    assert first == second, "Pipeline output must be reproducible."
    assert "not" in first, "Negation must be preserved."
    assert all(token == token.lower() for token in first)

    print(f"Tokens: {first}")
    print("All invariant checks passed.")


def main() -> None:
    required_resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    for resource_path, package_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError as exc:
            raise RuntimeError(
                f"Missing NLTK resource '{package_name}'. "
                "Download it explicitly in your environment."
            ) from exc

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "Missing spaCy English model. Install it with:\n"
            "uv run python -m spacy download en_core_web_sm"
        ) from exc

    experiment_basic_pipeline(nlp)
    experiment_configurations(nlp)
    experiment_negation(nlp)
    experiment_sentences(nlp)
    experiment_invariants(nlp)


if __name__ == "__main__":
    main()
