# Activation Functions

Activation functions introduce nonlinearity into neural networks, enabling them to learn complex relationships beyond linear transformations.

This module explores the mathematical foundations, practical implementations, and behavioral differences of common activation functions used in deep learning.

## Learning Objectives

- Understand why neural networks require nonlinear activation functions.
- Explain the mathematical properties and output ranges of common activations.
- Understand saturation, vanishing-gradient intuition, and the dying ReLU problem.
- Select appropriate activation functions for hidden layers and output layers.
- Compare activation functions through Python implementations and experiments.

## Topics

| Section                   | Topics                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------- |
| 01. Step and Sigmoid      | Threshold-based activation, sigmoid function, output range, binary classification     |
| 02. Tanh                  | Hyperbolic tangent, zero-centered outputs, comparison with sigmoid                    |
| 03. ReLU and Variants     | ReLU, Leaky ReLU, PReLU, GELU, dying ReLU                                             |
| 04. Softmax               | Multiclass classification, probability distributions, numerical stability             |
| 05. Activation Comparison | Function visualization, output behavior, comparative experiments, practical selection |

## Mathematical Foundations

### Nonlinearity

A neural network without nonlinear activation functions reduces to a single linear transformation, regardless of how many linear layers it contains.

Activation functions allow neural networks to represent more complex nonlinear relationships.

### Common Activation Functions

| Function   | Output Range             | Typical Use                                            |
| ---------- | ------------------------ | ------------------------------------------------------ |
| Step       | Binary threshold output  | Perceptron fundamentals                                |
| Sigmoid    | (0, 1)                   | Binary classification output                           |
| Tanh       | (-1, 1)                  | Selected hidden layers and recurrent architectures     |
| ReLU       | [0, ∞)                   | Hidden layers in many neural networks                  |
| Leaky ReLU | Unbounded                | Hidden layers where a nonzero negative slope is useful |
| GELU       | Unbounded                | Transformer and modern neural network architectures    |
| Softmax    | (0, 1), outputs sum to 1 | Multiclass classification output                       |

## Implementation and Experiments

Implementations use Python, NumPy, Matplotlib, and scikit-learn where appropriate.

Experiments focus on:

- Implementing activation functions and visualizing their curves.
- Comparing output ranges and nonlinear behavior.
- Investigating saturation and the intuition behind vanishing gradients.
- Demonstrating the dying ReLU problem and comparing ReLU variants.
- Understanding softmax outputs and numerical stability.
- Selecting activation functions based on the network architecture and learning task.

## Key Takeaways

- Activation functions introduce nonlinearity into neural networks.
- Different activation functions have different output ranges and numerical properties.
- Saturation can contribute to vanishing gradients during training.
- ReLU and its variants are widely used in hidden layers, while sigmoid and softmax are common output-layer choices for classification.
- Activation functions must be selected according to the task, architecture, and training behavior.

## Scope and Prerequisites

This module builds on the neural network fundamentals covered in `01-neural-network-fundamentals`.

Gradient derivation, backpropagation, and optimization algorithms are covered in dedicated modules later in the course. Here, gradient-related concepts are introduced only as needed to explain activation-function behavior.
