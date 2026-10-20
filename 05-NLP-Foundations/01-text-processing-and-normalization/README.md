# Text Processing and Normalization

This experiment explores how to clean noisy text, normalize Unicode representations, and standardize whitespace while preserving meaningful content.

## Learning Objectives

- Identify common text-quality issues.
- Apply Unicode normalization using NFC and NFKC.
- Remove HTML tags and decode HTML entities.
- Normalize whitespace using regular expressions.
- Understand the trade-offs between normalization and information preservation.

## Core Concepts

### 1. Unicode Normalization

Unicode allows visually identical text to have different underlying character representations. For example, an accented character may be represented as a single code point or as a base character followed by a combining mark.

Unicode normalization provides standardized representations:

- **NFC:** Canonical decomposition followed by composition.
- **NFD:** Canonical decomposition.
- **NFKC:** Compatibility decomposition followed by composition.
- **NFKD:** Compatibility decomposition.

For example, NFC can make canonically equivalent representations of an accented character identical.

Let \(D\) be the original document and \(T_{\text{NFC}}\) the NFC transformation:

$$
D' = T_{\text{NFC}}(D)
$$

NFKC additionally normalizes certain compatibility characters. It can change text representation more aggressively, so it should be chosen according to the application.

### 2. HTML Cleaning

Web documents may contain HTML tags and encoded entities.

For example:

- Input: `<p>Natural Language &amp; AI</p>`
- Output: `Natural Language & AI`

HTML entity decoding and HTML tag removal are distinct operations. A robust implementation should decode entities and use an HTML parser when processing complex markup.

### 3. Whitespace Normalization

Text can contain repeated spaces, tabs, and newline characters.

A regular expression can replace consecutive whitespace characters with one space:

$$
D' = \operatorname{join}(\operatorname{split}(D))
$$

This expression represents splitting on whitespace and joining the resulting parts with single spaces.

### 4. Cleaning as a Transformation

Text cleaning applies selected transformations to the original document:

$$
D' = T_{\text{space}}(T_{\text{HTML}}(D))
$$

The order and choice of transformations matter. Aggressive cleaning may remove meaningful punctuation, formatting, or other information needed by downstream tasks.

## Implementation and Experiments

Run the Python script to compare original and normalized text, inspect Unicode code points, clean HTML, and normalize whitespace.

## Expected Outcome

A practical understanding of Unicode normalization and common text-cleaning operations, including the ability to implement them while minimizing unintended information loss.