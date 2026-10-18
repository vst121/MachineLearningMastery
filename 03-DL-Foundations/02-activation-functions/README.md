# 02. Activation Functions

Activation functions introduce nonlinearity into neural networks, enabling them to learn complex relationships beyond linear transformations.

This module explores the mathematical foundations, practical implementations, and behavioral differences of common activation functions used in deep learning.

## Learning Objectives

- Understand why neural networks require nonlinear activation functions.
- Explain the mathematical definitions and properties of common activation functions.
- Compare output ranges, saturation, and behavior for negative and positive inputs.
- Understand the intuition behind vanishing gradients and the dying ReLU problem.
- Explain Softmax and its role in multiclass classification.
- Select appropriate activation functions for hidden layers and output layers.

## Implementations and Experiments

| File | Topics |
|---|---|
| `01_step_and_sigmoid.py` | Step and sigmoid functions, threshold-based decisions, saturation |
| `02_tanh.py` | Hyperbolic tangent, zero-centered outputs, sigmoid relationship |
| `03_relu_and_variants.py` | ReLU, Leaky ReLU, PReLU, GELU, dying ReLU |
| `04_softmax.py` | Logits, multiclass probabilities, numerical stability |
| `05_activation_comparison.py` | Comparative visualizations, output behavior, practical use cases |

## 1. Why Activation Functions Matter

A neural network layer applies a linear transformation followed by an activation function:

$$
h = f(Wx + b)
$$

where:

- \(x\) is the input vector.
- \(W\) is the weight matrix.
- \(b\) is the bias vector.
- \(f\) is the activation function.
- \(h\) is the resulting output.

Without nonlinear activation functions, stacking linear layers is mathematically equivalent to a single linear transformation. Nonlinear activations allow neural networks to learn more complex relationships.

## 2. Step Function

The step function converts an input into a binary output based on a threshold.

$$
f(x) =
\begin{cases}
1 & \text{if } x \geq 0 \\
0 & \text{if } x < 0
\end{cases}
$$

**Properties**

- Output range: {0, 1}.
- Produces a discrete threshold-based decision.
- Used in classical perceptron explanations.
- Not suitable for standard gradient-based training of multilayer networks because its derivative is zero almost everywhere and undefined at the threshold.

## 3. Sigmoid Function

The sigmoid function maps real-valued inputs to outputs between zero and one.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Its derivative is:

$$
\sigma'(x) = \sigma(x)(1-\sigma(x))
$$

**Properties**

- Output range: (0, 1).
- \(\sigma(0) = 0.5\).
- Smooth and differentiable.
- Often used in binary classification output layers.
- Saturates for large positive and negative inputs, where its derivative approaches zero.

The sigmoid function can be interpreted as a predicted probability when used in an appropriate binary classification model.

## 4. Hyperbolic Tangent (tanh)

The hyperbolic tangent function maps real-valued inputs to outputs between negative one and one.

$$
\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

Its derivative is:

$$
\frac{d}{dx}\tanh(x) = 1-\tanh^2(x)
$$

The relationship to sigmoid is:

$$
\tanh(x) = 2\sigma(2x)-1
$$

**Properties**

- Output range: (-1, 1).
- Zero-centered: \(\tanh(0)=0\).
- Useful in selected hidden layers and recurrent architectures.
- Can saturate at extreme inputs, contributing to vanishing gradients.

## 5. ReLU and Its Variants

### 5.1 ReLU

The Rectified Linear Unit is defined as:

$$
\text{ReLU}(x)=\max(0,x)
$$

Its derivative, away from zero, is:

$$
f'(x)=
\begin{cases}
1 & \text{if } x>0 \\
0 & \text{if } x<0
\end{cases}
$$

At zero, the function is not differentiable; implementations use a chosen subgradient convention.

**Properties**

- Output range: [0, ∞).
- Computationally simple.
- Widely used in hidden layers.
- Can produce inactive neurons when pre-activation values remain negative.

### 5.2 Leaky ReLU

Leaky ReLU introduces a small negative slope.

$$
f(x)=
\begin{cases}
x & \text{if } x\geq 0 \\
\alpha x & \text{if } x<0
\end{cases}
$$

Here, \(\alpha\) is a small positive constant, often 0.01.

**Properties**

- Allows a nonzero slope for negative inputs.
- Can reduce the risk of dying ReLU neurons.
- The negative slope is normally fixed.

### 5.3 Parametric ReLU (PReLU)

PReLU uses a learnable negative slope.

$$
f(x)=
\begin{cases}
x & \text{if } x\geq 0 \\
\alpha x & \text{if } x<0
\end{cases}
$$

The difference from Leaky ReLU is that \(\alpha\) is learned during training rather than fixed.

**Properties**

- Retains a nonzero negative slope.
- Adds trainable parameters.
- Requires gradient-based optimization to learn \(\alpha\).

### 5.4 Gaussian Error Linear Unit (GELU)

GELU is defined using the standard normal cumulative distribution function:

$$
\text{GELU}(x)=x\Phi(x)
$$

where \(\Phi(x)\) is the cumulative distribution function of a standard normal distribution.

A commonly used approximation is:

$$
\text{GELU}(x)\approx
\frac{x}{2}
\left[
1+\tanh\left(
\sqrt{\frac{2}{\pi}}
(x+0.044715x^3)
\right)
\right]
$$

**Properties**

- Smooth activation function.
- Allows small negative outputs.
- Widely used in Transformer architectures.
- Unlike ReLU, does not simply set all negative inputs to zero.

### 5.5 The Dying ReLU Problem

A ReLU neuron outputs zero whenever its pre-activation is negative. If it remains in this region for all relevant training examples, its gradient through the ReLU activation is zero, potentially preventing it from recovering through ordinary gradient updates.

Leaky ReLU and PReLU preserve a negative slope, which can mitigate this issue.

## 6. Softmax Function

Softmax converts a vector of logits into a probability distribution over \(K\) classes.

$$
\text{softmax}(z_i)=
\frac{e^{z_i}}
{\sum_{j=1}^{K}e^{z_j}}
$$

where:

- \(z_i\) is the logit for class \(i\).
- \(K\) is the number of classes.
- The denominator sums exponentials across all classes.

For every class \(i\):

$$
0 < \text{softmax}(z_i) < 1
$$

and:

$$
\sum_{i=1}^{K}\text{softmax}(z_i)=1
$$

**Properties**

- Produces normalized class probabilities.
- The largest logit corresponds to the largest Softmax probability.
- Commonly used for mutually exclusive multiclass classification.
- Should be implemented with numerical stability in mind.

A common stable formulation subtracts the largest logit before exponentiation:

$$
\text{softmax}(z_i)=
\frac{e^{z_i-m}}
{\sum_{j=1}^{K}e^{z_j-m}}
$$

where:

$$
m=\max_j z_j
$$

Subtracting the same constant from every logit does not change the resulting probabilities, but helps prevent exponential overflow.

For multilabel classification, where several classes may independently be present, sigmoid is generally more appropriate than Softmax.

## 7. Comparison of Activation Functions

| Function | Output Range | Zero-Centered | Typical Use |
|---|---|---|---|
| Step | {0, 1} | No | Perceptron fundamentals |
| Sigmoid | (0, 1) | No | Binary classification output |
| Tanh | (-1, 1) | Yes | Selected hidden and recurrent layers |
| ReLU | [0, ∞) | No | Hidden layers |
| Leaky ReLU | Unbounded | No | Hidden layers |
| PReLU | Unbounded | No | Hidden layers with learnable negative slope |
| GELU | Unbounded | No | Transformers and modern neural networks |
| Softmax | (0, 1) per class | No | Multiclass classification output |

## 8. Implementation and Experiments

The Python implementations use NumPy and Matplotlib.

Experiments include:

- Plotting activation curves and comparing their behavior.
- Examining sigmoid and tanh saturation.
- Comparing ReLU, Leaky ReLU, PReLU, and GELU for negative inputs.
- Illustrating the dying ReLU problem conceptually.
- Converting logits into Softmax probability distributions.
- Verifying that Softmax probabilities sum to one.
- Investigating numerical stability with large logits.

## 9. Key Takeaways

- Nonlinear activations allow neural networks to learn nonlinear relationships.
- Sigmoid and tanh can saturate, contributing to vanishing gradients.
- ReLU is simple and effective, while its variants offer alternative negative-input behavior.
- GELU is widely used in Transformer architectures.
- Softmax converts multiclass logits into a normalized probability distribution.
- Activation functions should be selected according to the task, architecture, and training behavior.

## 10. Scope and Prerequisites

This module builds on `01-neural-network-fundamentals`.

The mathematical intuition behind gradient behavior is introduced here, but detailed gradient derivations, backpropagation, and optimization algorithms are covered in dedicated modules later in the course.