# Backpropagation and Autograd

Backpropagation is the algorithm used to calculate how each model parameter contributes to the final loss. It applies the chain rule of calculus to propagate gradients backward through a neural network.

Automatic differentiation frameworks such as PyTorch automate these calculations, allowing neural networks to be trained efficiently.

## Learning Objectives

- Understand derivatives, partial derivatives, and the chain rule.
- Represent neural networks as computational graphs.
- Understand forward and backward passes.
- Calculate gradients manually for simple neural networks.
- Implement backpropagation using NumPy.
- Use PyTorch autograd to calculate gradients automatically.
- Understand gradient accumulation and computational graph lifecycle.
- Verify gradients using numerical approximations.

## 1. Derivatives and the Chain Rule

A derivative measures how a function changes when its input changes.

For a function \(y=f(x)\), the derivative is:

$$
\frac{dy}{dx}=f'(x)
$$

For a composition of functions, the chain rule calculates how changes propagate through each function:

$$
y=f(g(x))
$$

$$
\frac{dy}{dx}=
\frac{df}{dg}\frac{dg}{dx}
$$

Neural networks contain many composed functions. The chain rule makes it possible to calculate derivatives throughout the network.

## 2. Computational Graphs

A computational graph represents mathematical operations as connected nodes.

For a simple neural network:

$$
z=wx+b
$$

$$
a=\sigma(z)
$$

$$
L=(a-y)^2
$$

The graph describes how the input, parameters, activation, and loss depend on one another.

During the forward pass, values are calculated from inputs to outputs. During the backward pass, gradients are propagated from the loss toward the parameters.

## 3. Forward and Backward Passes

### Forward pass

The network calculates predictions and evaluates the loss.

$$
x \rightarrow z \rightarrow a \rightarrow L
$$

### Backward pass

The network calculates derivatives of the loss with respect to intermediate values and parameters.

$$
\frac{\partial L}{\partial L}
\rightarrow
\frac{\partial L}{\partial a}
\rightarrow
\frac{\partial L}{\partial z}
\rightarrow
\frac{\partial L}{\partial w}
$$

These gradients describe how the loss changes with each parameter. An optimizer can subsequently use them to update the parameters.

## 4. Gradients in a Simple Neural Network

Consider a neuron:

$$
z=wx+b
$$

$$
a=\sigma(z)
$$

Using squared error for a single sample:

$$
L=(a-y)^2
$$

The gradient with respect to the weight is obtained through the chain rule:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
$$

The individual derivatives are:

$$
\frac{\partial L}{\partial a}=2(a-y)
$$

$$
\frac{\partial a}{\partial z}=\sigma'(z)
$$

$$
\frac{\partial z}{\partial w}=x
$$

Therefore:

$$
\frac{\partial L}{\partial w}
=
2(a-y)\sigma'(z)x
$$

This demonstrates how the gradient depends on the loss, activation function, and input.

## 5. Manual Backpropagation with NumPy

Implement a small neural network and calculate gradients explicitly.

Focus:
- Forward computation.
- Derivatives of each operation.
- Chain rule application.
- Gradients for weights and biases.
- Comparison against finite-difference approximations.

The purpose is to understand the mathematics rather than build a production training framework.

## 6. Automatic Differentiation with PyTorch

PyTorch autograd records operations performed on tensors and calculates gradients automatically.

Core concepts:
- `requires_grad=True`
- `loss.backward()`
- `tensor.grad`
- `torch.no_grad()`
- `detach()`
- Gradient accumulation
- Computational graph lifecycle

Autograd automates differentiation, but it does not automatically choose an optimization algorithm or update model parameters.

## 7. Gradient Verification

Numerical gradient checking provides an independent approximation of a derivative.

For a scalar function:

$$
f'(x)\approx
\frac{f(x+h)-f(x-h)}{2h}
$$

Compare numerical approximations against analytically calculated gradients and PyTorch autograd results.

This helps identify errors in manually implemented backward passes.

## Implementation and Experiments

| File | Focus |
|---|---|
| `01_derivatives_and_chain_rule.py` | Derivatives, partial derivatives, and chain rule |
| `02_computational_graph.py` | Forward computation and dependency graphs |
| `03_manual_backpropagation.py` | Explicit gradients for a simple neural network using NumPy |
| `04_gradient_checking.py` | Numerical gradient approximation and verification |
| `05_pytorch_autograd.py` | Automatic differentiation using PyTorch |
| `06_gradient_accumulation.py` | Gradient accumulation, clearing gradients, and graph lifecycle |

## Important Distinctions

### Backpropagation versus autograd

Backpropagation describes how gradients are propagated through a computational graph. Automatic differentiation is a broader technique for calculating derivatives programmatically. PyTorch autograd implements automatic differentiation and uses reverse-mode differentiation for typical neural network training.

### Gradients versus optimization

A gradient describes how a loss changes with respect to a parameter. An optimizer uses gradients and an update rule to change parameters.

Gradient-based optimization, learning rates, SGD, momentum, and Adam belong to a separate section.

### Loss versus gradient

The loss quantifies prediction error. Backpropagation calculates how that loss depends on the network's parameters. Both are necessary for gradient-based neural network training.

## Scope

This section focuses on the mathematics and implementation of gradient computation.

It does not cover optimization algorithms in depth, advanced initialization strategies, or training techniques such as learning-rate scheduling. Those topics are addressed separately.
