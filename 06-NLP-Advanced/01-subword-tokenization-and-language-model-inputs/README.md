# Subword Tokenization and Language Model Inputs

This module explores how raw text is transformed into numerical representations suitable for pretrained language models. It covers subword tokenization algorithms, vocabulary construction, special tokens, attention masks, padding, truncation, and end-to-end input preparation.

## Learning Objectives

- Understand word-level, character-level, and subword tokenization.
- Compare Byte Pair Encoding (BPE), WordPiece, and Unigram tokenization.
- Understand vocabulary construction, token IDs, and unknown-token handling.
- Prepare model inputs using special tokens, padding, attention masks, and truncation.
- Analyze sequence lengths and padding efficiency.
- Build a reusable preprocessing pipeline for pretrained language models.
- Validate input tensors and inspect contextual representations produced by a Transformer encoder.

## Core Concepts

### 1. Tokenization Strategies

Tokenization transforms text into a sequence of smaller units called tokens.

**Word-level tokenization** splits text into words and punctuation. It provides relatively short sequences but can produce unknown tokens for words outside the vocabulary.

**Character-level tokenization** represents text as individual characters. It reduces dependence on a fixed word vocabulary but produces longer sequences.

**Subword tokenization** divides words into reusable units, balancing vocabulary size and sequence length.

For a text sequence containing tokens, the tokenization function is:

$$
T(x) = (t_1, t_2, \ldots, t_n)
$$

Each token is mapped to an integer ID using a vocabulary function:

$$
E(t_i) = v_i
$$

The encoded input becomes:

$$
\mathbf{x} = (v_1, v_2, \ldots, v_n)
$$

### 2. Subword Tokenization Algorithms

**Byte Pair Encoding (BPE)** iteratively merges frequent adjacent token pairs.

$$
(a,b) \rightarrow ab
$$

**WordPiece** constructs a subword vocabulary using a scoring strategy associated with language-model likelihood. At inference time, it typically applies a longest-match-first segmentation strategy.

**Unigram** models tokenization probabilistically using a vocabulary of candidate subwords.

$$
T^* = \arg\max_T P(T \mid x)
$$

Here, the selected token sequence is the most probable segmentation under the tokenizer's model.

The algorithms differ in vocabulary construction and segmentation. Their practical behavior also depends on normalization, pre-tokenization, vocabulary size, and training data.

### 3. Vocabulary and Unknown Tokens

A vocabulary maps tokens to unique integer IDs. Special entries commonly include padding and unknown-token identifiers.

$$
V = \{t_1, t_2, \ldots, t_{|V|}\}
$$

The vocabulary size is:

$$
|V| = \text{number of distinct vocabulary entries}
$$

An unknown token may be used when the tokenizer cannot represent an input using its available vocabulary. Subword tokenizers can often represent unfamiliar words by combining known pieces, although this depends on the tokenizer's design.

### 4. Special Tokens and Sequence Structure

Pretrained models may use special tokens to mark sequence boundaries, classification positions, separators, or other task-specific information.

For example, BERT commonly uses:

$$
[\mathrm{CLS}], \quad [\mathrm{SEP}], \quad [\mathrm{PAD}], \quad [\mathrm{UNK}]
$$

The exact tokens and their meanings depend on the model and tokenizer.

Special tokens contribute to the final sequence length and must be considered when preparing model inputs.

### 5. Padding and Attention Masks

Batches often contain sequences of different lengths. Padding makes their tensor dimensions compatible.

For a batch with sequence lengths \(n_1, n_2, \ldots, n_B\), dynamic padding uses:

$$
L = \max(n_1, n_2, \ldots, n_B)
$$

Each sequence is padded to length \(L\).

The attention mask identifies real tokens and padding positions:

$$
m_i =
\begin{cases}
1 & \text{if position } i \text{ contains a real token} \\
0 & \text{if position } i \text{ contains padding}
\end{cases}
$$

The attention mask helps the model avoid attending to padding positions. It does not automatically exclude padding from every possible downstream operation.

### 6. Truncation and Sequence-Length Constraints

When an input exceeds the permitted sequence length, truncation removes tokens according to the configured strategy.

$$
L_{\text{final}} = \min(L_{\text{original}}, L_{\max})
$$

Here:

- \(L\_{\text{original}}\) is the tokenized sequence length, including any special tokens added during encoding.
- \(L\_{\max}\) is the configured maximum sequence length.
- \(L\_{\text{final}}\) is the resulting sequence length.

Truncation controls memory consumption and computation but may remove useful context. The appropriate maximum length depends on the model's supported context length, task requirements, and resource constraints.

### 7. Padding Efficiency

Padding efficiency measures the proportion of positions in a batch that contain real tokens.

$$
\text{Efficiency} =
\frac{\text{Number of real tokens}}
{\text{Total number of sequence positions}}
$$

Equivalently:

$$
\eta =
\frac{\sum_{i=1}^{B} n_i}{B L}
$$

where \(B\) is the batch size, \(n_i\) is the number of non-padding tokens in sequence \(i\), and \(L\) is the padded sequence length.

Higher padding efficiency means fewer positions are allocated to padding, although actual runtime also depends on hardware, model architecture, and implementation.

### 8. Transformer Input Representations

Token IDs are converted into learned vector representations by the model's embedding layers. A Transformer encoder then produces contextual representations.

For a batch of tokenized inputs, the last hidden-state tensor has shape:

$$
\mathbf{H} \in \mathbb{R}^{B \times L \times d}
$$

where:

- \(B\) is the batch size.
- \(L\) is the sequence length.
- \(d\) is the hidden dimension.

For BERT, the representation at the first token position can be extracted as:

$$
\mathbf{h}_{\mathrm{CLS}} = \mathbf{H}_{:,0,:}
$$

This produces a tensor of shape:

$$
\mathbf{h}_{\mathrm{CLS}} \in \mathbb{R}^{B \times d}
$$

The resulting vector is a contextual representation. Using it for classification or another prediction task requires an appropriate downstream head or task-specific procedure.

## Implementation and Experiments

| File                                      | Experiment                                                                                                                                                                                               |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_subword_tokenization_fundamentals.py` | Compare word-level and character-level tokenization, construct vocabularies, encode tokens into IDs, and examine unknown-token behavior. Includes a simplified greedy subword-segmentation illustration. |
| `02_tokenizer_algorithms_comparison.py`   | Train and compare BPE, WordPiece, and Unigram tokenizers on a shared corpus. Inspect token segmentation, vocabulary sizes, and sequence lengths.                                                         |
| `03_language_model_input_preparation.py`  | Inspect pretrained BERT tokenizer special tokens, token IDs, dynamic and fixed-length padding, attention masks, and truncation.                                                                          |
| `04_language_model_input_pipeline.py`     | Build a reusable input pipeline with batch processing, tensor validation, padding-efficiency analysis, and a forward pass through a pretrained BERT encoder.                                             |

### Experimental Workflow

1. Tokenize raw text and inspect the resulting tokens.
2. Construct or train vocabularies and map tokens to IDs.
3. Compare subword segmentation across algorithms.
4. Prepare batches with special tokens, padding, attention masks, and truncation.
5. Validate tensor shapes and attention-mask values.
6. Evaluate padding efficiency and inspect contextual representations.

The experiments use Python, PyTorch, Hugging Face Transformers, and the Hugging Face Tokenizers library. Pretrained model experiments may require downloading tokenizer files and model weights.

## Expected Outcome

After completing this module, you should be able to:

- Explain the differences between word-level, character-level, BPE, WordPiece, and Unigram tokenization.
- Interpret token IDs, vocabulary sizes, special tokens, and unknown-token behavior.
- Prepare batches of model-compatible input tensors.
- Explain how padding, attention masks, and truncation affect sequence preparation.
- Measure padding efficiency and understand its impact on computational resources.
- Implement and validate a reusable preprocessing pipeline for pretrained language models.
- Interpret the dimensions of Transformer contextual representations and understand their role in downstream NLP tasks.
