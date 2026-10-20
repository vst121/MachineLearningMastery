# Attention and Transformers

Explore attention mechanisms and the Transformer architecture, which form the foundation of modern language models and many state-of-the-art AI systems.

## 🎯 Learning Objectives

- Understand how attention overcomes limitations of traditional sequence models.
- Implement scaled dot-product attention and self-attention from first principles.
- Understand query, key, and value representations.
- Implement multi-head attention and positional encoding.
- Build Transformer encoder and decoder components.
- Use PyTorch's built-in attention and Transformer modules.
- Understand masking, residual connections, layer normalization, and feed-forward networks.

## 🧠 Core Concepts

### 1. Attention Fundamentals

Attention allows a model to determine which parts of an input are most relevant when processing a particular element.

- Queries, keys, and values
- Similarity scores and attention weights
- Softmax normalization
- Context vectors
- Weighted aggregation of information

### 2. Scaled Dot-Product Attention

Scaled dot-product attention computes query-key similarities, normalizes the scores, and combines the value vectors.

$$
\text{Attention}(Q,K,V) = \text{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

- Query-key similarity and scaling
- Softmax and weighted aggregation
- Attention masks
- Numerical stability and tensor dimensions

### 3. Self-Attention

Self-attention enables each token to incorporate contextual information from other tokens in the same sequence.

- Learnable query, key, and value projections
- Token-to-token relationships
- Attention score matrices
- Padding and causal masks
- Tensor shape transformations

### 4. Multi-Head Attention

Multi-head attention performs several attention operations in parallel, allowing the model to capture different relationships between tokens.

- Attention heads and head dimensions
- Parallel attention computation
- Concatenation and output projection
- Computational cost and tensor dimensions

### 5. Positional Encoding

Attention does not inherently encode token order. Positional information helps the model distinguish tokens according to their positions.

- Sinusoidal positional encoding
- Learned positional embeddings
- Combining token embeddings with positional information
- Sequence length and embedding dimensions

### 6. Transformer Architecture

Transformers combine attention, feed-forward networks, residual connections, and normalization to build contextual representations and support sequence generation.

- Encoder blocks and bidirectional self-attention
- Decoder blocks and causal self-attention
- Encoder-decoder cross-attention
- Position-wise feed-forward networks
- Residual connections and layer normalization
- Padding and causal masks
- Encoder-only, decoder-only, and encoder-decoder architectures
- PyTorch attention and Transformer modules

## 🛠️ Implementation and Experiments

| File                                 | Experiment                                                                                                                                                          |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_attention_fundamentals.py`       | Explore queries, keys, values, attention weights, and context vectors using a numerical example.                                                                    |
| `02_scaled_dot_product_attention.py` | Implement scaled dot-product attention, including softmax, masking, and numerical verification.                                                                     |
| `03_self_attention.py`               | Build self-attention with learnable query, key, and value projections.                                                                                              |
| `04_multi_head_attention.py`         | Implement multiple attention heads, combine their outputs, and verify tensor dimensions.                                                                            |
| `05_positional_encoding.py`          | Generate and inspect sinusoidal positional encodings and their effect on token representations.                                                                     |
| `06_transformer_architecture.py`     | Build and validate Transformer encoder and decoder components, including residual connections, normalization, causal masking, cross-attention, and PyTorch modules. |

## ✅ Expected Outcome

By completing this module, you will be able to:

- Explain the mathematical foundations of attention and Transformers.
- Implement core attention mechanisms from scratch.
- Track and validate tensor dimensions throughout attention computations.
- Explain positional encoding, multi-head attention, and masking.
- Build and test Transformer encoder and decoder components in PyTorch.
- Distinguish encoder-only, decoder-only, and encoder-decoder architectures.
- Understand the architectural foundations of modern LLMs and prepare for advanced topics such as generative deep learning and model fine-tuning.
