# Recurrent Neural Networks (RNNs)

## Overview

Recurrent Neural Networks (RNNs) are neural networks designed to process sequential data. Unlike feedforward networks, RNNs maintain a hidden state that carries information across time steps, allowing the model to use previous inputs when processing the current input.

This section explores the mathematical foundations, forward propagation, training dynamics, and practical implementation of RNNs using NumPy and PyTorch.

## Learning Objectives

- Understand sequential data and recurrent connections.
- Explain hidden states and parameter sharing across time steps.
- Implement the RNN forward pass from scratch using NumPy.
- Understand Backpropagation Through Time (BPTT).
- Analyze vanishing and exploding gradients.
- Build and train RNN models using PyTorch.
- Apply RNNs to sequence prediction tasks.

## Core Concepts

### 1. Recurrent Computation

At each time step, an RNN combines the current input with the previous hidden state to compute a new hidden state.

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

Where:

- $$x_t$$ is the input at time step $$t$$.
- $$h_t$$ is the current hidden state.
- $$h_{t-1}$$ is the previous hidden state.
- $$W_{xh}$$ and $$W_{hh}$$ are the input-to-hidden and hidden-to-hidden weight matrices.
- $$b_h$$ is the hidden-state bias.

The same parameters are reused at every time step, enabling the network to process sequences of different lengths.

### 2. Output Computation

An RNN can generate an output at each time step or use the final hidden state for a sequence-level prediction.

$$
y_t = W_{hy}h_t + b_y
$$

For classification, the output layer typically produces logits that are passed to an appropriate loss function.

### 3. Backpropagation Through Time

Backpropagation Through Time (BPTT) computes gradients by unfolding the recurrent network across its time steps and propagating errors backward through the sequence.

Long sequences can cause gradients to vanish or explode, making it difficult to learn long-range dependencies.

### 4. Sequence Modeling

RNNs can be applied to tasks such as:

- Time-series forecasting
- Sequence classification
- Language modeling
- Sequential pattern recognition

## Implementation and Experiments

| File                                  | Description                                                               |
| ------------------------------------- | ------------------------------------------------------------------------- |
| `01_rnn_fundamentals.py`              | Explore recurrent connections, hidden states, and parameter sharing.      |
| `02_rnn_forward_pass.py`              | Implement and inspect hidden states and outputs across a sequence.        |
| `03_backpropagation_through_time.py`  | Study temporal gradient propagation and BPTT.                             |
| `04_rnn_pytorch.py`                   | Implement an RNN using PyTorch and inspect its tensor dimensions.         |
| `05_vanishing_exploding_gradients.py` | Experiment with gradient norms, sequence length, and gradient clipping.   |
| `06_rnn_sequence_prediction.py`       | Train an RNN for a sequence-prediction task and evaluate its performance. |

### Experimental Focus

- Compare hidden-state dimensions across time steps.
- Verify recurrent computations against a manual implementation.
- Observe how gradients propagate through recurrent connections.
- Investigate training stability on longer sequences.
- Evaluate predictions on unseen sequential data.

## Expected Outcome

After completing this section, you should be able to explain how RNNs process sequential information, implement their core computations, understand the limitations of recurrent training, and build a basic sequence model using PyTorch.

These foundations prepare you to study more advanced recurrent architectures, including Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU).
