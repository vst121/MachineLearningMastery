# Activation Functions

Activation functions introduce nonlinearity into neural networks, enabling them to learn complex relationships beyond linear transformations.

This module explores the mathematical foundations, practical implementations, and behavioral differences of common activation functions used in deep learning.

## Learning Objectives

- Understand why neural networks require nonlinear activation functions.
- Implement and visualize common activation functions using NumPy.
- Compare activation functions by output range, symmetry, and behavior for negative and positive inputs.
- Understand saturation, the intuition behind vanishing gradients, and the dying ReLU problem.
- Explain Softmax and its role in multiclass classification.
- Select appropriate activation functions for hidden layers and output layers.

## Implementations and Experiments

| File                          | Topics                                                                                   |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| `01_step_and_sigmoid.py`      | Step and sigmoid functions, threshold-based decisions, output ranges, saturation         |
| `02_tanh.py`                  | Hyperbolic tangent, zero-centered outputs, sigmoid relationship, saturation              |
| `03_relu_and_variants.py`     | ReLU, Leaky ReLU, PReLU, GELU, and dying ReLU                                            |
| `04_softmax.py`               | Multiclass probability distributions, logits, prediction confidence, numerical stability |
| `05_activation_comparison.py` | Comparative visualizations, output behavior, and typical use cases                       |

## Mathematical Foundations

### Nonlinearity

A composition of linear transformations remains a linear transformation. Without nonlinear activation functions, stacking linear layers cannot represent general nonlinear relationships.

Activation functions enable neural networks to learn more complex patterns.

### Activation Functions

| Function   | Output Range     | Typical Use                                        |
| ---------- | ---------------- | -------------------------------------------------- |
| Step       | {0, 1}           | Perceptron fundamentals                            |
| Sigmoid    | (0, 1)           | Binary classification output                       |
| Tanh       | (-1, 1)          | Selected hidden layers and recurrent architectures |
| ReLU       | [0, ∞)           | Hidden layers in many neural networks              |
| Leaky ReLU | Unbounded        | Hidden layers with a nonzero negative slope        |
| PReLU      | Unbounded        | Hidden layers with a learnable negative slope      |
| GELU       | Unbounded        | Transformers and modern neural networks            |
| Softmax    | (0, 1) per class | Mutually exclusive multiclass classification       |

### Important Properties

- **Nonlinearity:** Enables neural networks to model nonlinear relationships.
- **Saturation:** Sigmoid and tanh approach limiting values for extreme inputs.
- **Zero-centered output:** Tanh produces outputs around zero, unlike sigmoid.
- **Dying ReLU:** ReLU neurons can remain inactive when their pre-activation values stay negative.
- **Numerical stability:** Softmax implementations should avoid overflow during exponentiation.
- **Task-dependent selection:** Hidden-layer and output-layer activation functions serve different purposes.

## Implementation and Experiments

The implementations use Python, NumPy, and Matplotlib.

Experiments include:

- Plotting and comparing activation curves.
- Inspecting function outputs for representative inputs.
- Examining sigmoid and tanh saturation.
- Comparing ReLU variants for negative inputs.
- Illustrating the dying ReLU phenomenon conceptually.
- Converting logits into Softmax probability distributions.
- Verifying that Softmax probabilities sum to one.
- Examining numerical stability with large logits.

## Key Takeaways

- Nonlinear activation functions are essential for learning general nonlinear relationships.
- ReLU and its variants are widely used in hidden layers.
- Sigmoid and Softmax are common output functions for binary and mutually exclusive multiclass classification, respectively.
- Saturation can contribute to vanishing gradients during training.
- Softmax probabilities describe a model's relative output distribution but do not guarantee calibrated confidence.
- Activation functions should be selected according to the task, architecture, and training behavior.

## Scope and Prerequisites

This module builds on `01-neural-network-fundamentals`.

Gradient derivation, backpropagation, and optimization algorithms are covered in dedicated modules later in the course. Gradient-related behavior is discussed here only to explain activation-function properties.
