"""
Text Cleaning and Unicode

Experiments:
1. Inspect Unicode representations.
2. Compare NFC and NFKC normalization.
3. Remove HTML markup and decode entities.
4. Normalize whitespace.
5. Combine the operations into a basic cleaning function.
"""

import html
import re
import unicodedata
from html.parser import HTMLParser


class HTMLTextExtractor(HTMLParser):
    """Extract visible text from HTML markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.parts)


def extract_html_text(text: str) -> str:
    """Extract text content from HTML."""
    parser = HTMLTextExtractor()
    parser.feed(text)
    parser.close()
    return " ".join(parser.get_text().split())


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Normalize Unicode using a supported normalization form."""
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"Unsupported Unicode normalization form: {form}")

    return unicodedata.normalize(form, text)


def normalize_whitespace(text: str) -> str:
    """Replace consecutive whitespace with a single space."""
    return " ".join(text.split())


def clean_text(text: str, unicode_form: str = "NFC") -> str:
    """Apply basic, configurable text-cleaning operations."""
    text = normalize_unicode(text, unicode_form)
    text = html.unescape(text)
    text = extract_html_text(text)
    text = normalize_whitespace(text)
    return text


def experiment_unicode() -> None:
    print("\n=== Experiment 1: Unicode Normalization ===")

    composed = "\u00e9"
    decomposed = "e\u0301"

    print(f"Original A: {composed!r}")
    print(f"Original B: {decomposed!r}")
    print(f"Equal before normalization: {composed == decomposed}")

    normalized_a = normalize_unicode(composed, "NFC")
    normalized_b = normalize_unicode(decomposed, "NFC")

    print(f"Equal after NFC: {normalized_a == normalized_b}")
    print(f"Code points: {[hex(ord(c)) for c in normalized_b]}")

    compatibility_character = "\u2460"
    print(f"Compatibility character: {compatibility_character}")
    print(f"NFC: {normalize_unicode(compatibility_character, 'NFC')}")
    print(f"NFKC: {normalize_unicode(compatibility_character, 'NFKC')}")


def experiment_html() -> None:
    print("\n=== Experiment 2: HTML Cleaning ===")

    raw_html = "<p>Natural Language &amp; AI</p>"
    print(f"Original: {raw_html}")
    print(f"Cleaned:  {extract_html_text(html.unescape(raw_html))}")


def experiment_whitespace() -> None:
    print("\n=== Experiment 3: Whitespace Normalization ===")

    raw_text = "NLP   is\tuseful.\n\nText   needs cleaning."
    cleaned = normalize_whitespace(raw_text)

    print(f"Original: {raw_text!r}")
    print(f"Cleaned:  {cleaned!r}")

    # Verify that whitespace is standardized.
    assert cleaned == "NLP is useful. Text needs cleaning."


def experiment_pipeline() -> None:
    print("\n=== Experiment 4: Basic Cleaning Function ===")

    raw_text = (
        "<p>Cafe\u0301   is useful &amp; practical.</p>\n"
        "<p>Text\tpreprocessing matters.</p>"
    )

    cleaned = clean_text(raw_text, unicode_form="NFC")

    print(f"Original: {raw_text!r}")
    print(f"Cleaned:  {cleaned!r}")

    assert "\u0301" not in cleaned
    assert "  " not in cleaned
    assert "&amp;" not in cleaned
    assert "<p>" not in cleaned


def main() -> None:
    experiment_unicode()
    experiment_html()
    experiment_whitespace()
    experiment_pipeline()


if __name__ == "__main__":
    main()
