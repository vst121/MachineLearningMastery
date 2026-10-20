# Information Retrieval and Text Similarity

## 🎯 Learning Objectives

- Represent text as ordered sequences of token IDs.
- Build vocabularies, embeddings, and padded batches.
- Understand recurrent neural networks and their hidden states.
- Explain the differences between vanilla RNNs, LSTMs, and GRUs.
- Implement neural text classifiers using PyTorch.
- Handle variable-length sequences without treating padding as meaningful input.
- Compare recurrent architectures using consistent evaluation procedures.
- Analyze errors caused by unknown vocabulary, truncation, and limited training data.

## 🧠 Core Concepts

### 1. Text as a Sequence

Sequence models preserve token order, unlike Bag of Words representations.

$$
\mathbf{x}=(x_1,x_2,\ldots,x_T)
$$

Here, \(T\) is the sequence length and each \(x_t\) is a token ID.

A vocabulary maps tokens to integer IDs. Special tokens such as `<PAD>` and `<UNK>` represent padding and unknown words.

### 2. Word Embeddings

An embedding layer maps each token ID to a dense, learnable vector.

$$
\mathbf{e}_t=E[x_t]
$$

The embedding matrix \(E\) is learned during training unless pretrained embeddings are used.

For a batch, the embedding tensor has shape:

$$
(B,T,d)
$$

Where \(B\) is the batch size, \(T\) is the padded sequence length, and \(d\) is the embedding dimension.

### 3. Vanilla Recurrent Neural Networks

A vanilla RNN processes tokens sequentially, updating its hidden state at each step.

$$
\mathbf{h}_t=
\tanh(W_x\mathbf{e}_t+W_h\mathbf{h}_{t-1}+\mathbf{b})
$$

The hidden state carries information from previously processed tokens. For sequence classification, the final valid hidden state can serve as the document representation.

$$
\mathbf{z}=W_o\mathbf{h}_T+\mathbf{b}_o
$$

The output layer produces class logits, which can be converted into probabilities using softmax.

Vanilla RNNs can struggle with long-range dependencies because gradients may vanish or explode during training.

### 4. Long Short-Term Memory

LSTMs use gates and a separate cell state to control information flow.

$$
\mathbf{c}_t=
\mathbf{f}_t\odot\mathbf{c}_{t-1}
+
\mathbf{i}_t\odot\tilde{\mathbf{c}}_t
$$

The forget gate controls retained information, the input gate controls new information, and the output gate controls the exposed hidden state.

These mechanisms help LSTMs learn longer dependencies than vanilla RNNs in many settings.

### 5. Gated Recurrent Units

GRUs use update and reset gates but do not maintain a separate cell state.

$$
\mathbf{h}_t=
(1-\mathbf{z}_t)\odot\mathbf{h}_{t-1}
+
\mathbf{z}_t\odot\tilde{\mathbf{h}}_t
$$

Gate conventions differ between formulations and implementations. In this equation, the update gate weights the candidate hidden state.

GRUs often have fewer parameters than comparable LSTMs with the same input and hidden dimensions. However, parameter count alone does not determine predictive performance or runtime.

### 6. Padding and Variable-Length Sequences

Sequences in a batch are commonly padded to a common length.

Padding positions must not be processed as ordinary tokens. PyTorch supports packed sequences, which let recurrent layers process the valid sequence lengths.

Truncation limits computational cost but may remove important information.

### 7. Training Neural Classifiers

A typical training loop performs the following operations:

1. Convert token IDs into embeddings.
2. Process embeddings through a recurrent layer.
3. Generate class logits from the sequence representation.
4. Calculate cross-entropy loss.
5. Backpropagate gradients and update model parameters.
6. Evaluate on held-out examples.

For a single labeled example, cross-entropy loss is:

$$
L=-\log P(y\mid\mathbf{x})
$$

### 8. Classification Metrics

Accuracy measures the proportion of correct predictions.

$$
\operatorname{Accuracy}
=
\frac{\text{Correct predictions}}
{\text{Total predictions}}
$$

Precision, recall, and F1-score help identify class-specific weaknesses. Macro F1 gives equal weight to each class's F1-score.

$$
F_{1,\text{macro}}
=
\frac{1}{C}\sum_{c=1}^{C}F_{1,c}
$$

A confusion matrix and inspection of individual mistakes provide additional diagnostic information.

### 9. Error Analysis

Neural models can make errors for several reasons:

- **Unknown vocabulary:** unfamiliar tokens map to the same unknown-token representation.
- **Truncation:** useful words may be removed when sequences exceed the length limit.
- **Padding:** incorrect handling can make artificial padding affect the recurrent state.
- **Limited data:** a model may memorize training examples instead of learning general patterns.
- **Order sensitivity:** changing token order can alter hidden states and predictions.
- **Confidence:** high predicted probability does not guarantee a correct or well-calibrated prediction.

Error analysis should combine aggregate metrics with representative examples and, when possible, larger evaluation datasets.

## 🧪 Implementation and Experiments

| File                                           | Experiment                                                                                                                                                                        |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_sequence_representation_and_embeddings.py` | Build a vocabulary, encode token sequences, handle unknown tokens, pad and truncate sequences, inspect embedding tensors, and compare sequence representations with Bag of Words. |
| `02_vanilla_rnn_sequence_classification.py`    | Train a vanilla RNN classifier, inspect its hidden states, and evaluate predictions using classification metrics.                                                                 |
| `03_lstm_and_gru_comparison.py`                | Compare LSTM and GRU classifiers using a shared dataset split, reporting loss, accuracy, macro F1, parameter counts, and inference time.                                          |
| `04_sequence_model_error_analysis.py`          | Inspect misclassifications, confidence, unknown vocabulary, truncation sensitivity, padding, token order, and hidden-state statistics.                                            |

### Tools and Libraries

- **PyTorch:** embedding layers, recurrent networks, automatic differentiation, and optimization.
- **NumPy:** numerical analysis and reproducibility.
- **scikit-learn:** train-test splitting and classification metrics.

The experiments use a small, self-contained sports-versus-technology dataset. No external corpus or pretrained model is required.

## ✅ Expected Outcome

After completing this module, you should be able to:

- Convert text into sequences suitable for neural NLP models.
- Explain embedding lookups and recurrent hidden states.
- Describe the key differences between RNN, LSTM, and GRU architectures.
- Train and evaluate recurrent text classifiers in PyTorch.
- Handle padding and variable-length sequences correctly.
- Compare models using consistent splits and multiple evaluation metrics.
- Diagnose errors associated with vocabulary coverage and sequence length.
- Explain why recurrent models are useful learning tools while recognizing the role of Transformers in modern NLP.

**Important:** these experiments are educational demonstrations, not reliable benchmarks. The dataset is tiny, and the held-out test set contains only six documents. Use larger representative datasets, cross-validation, and a separate test set before drawing conclusions about generalization.
