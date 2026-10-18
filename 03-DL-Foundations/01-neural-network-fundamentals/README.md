# 🧠 Neural Network Fundamentals

## 1. Overview

This module introduces the fundamental building blocks of artificial neural networks and explains how neural networks transform input data into predictions.

It covers artificial neurons, weights, biases, layers, parameters, forward propagation, and multilayer perceptrons (MLPs). The practical exercises use NumPy to make the underlying computations explicit and PyTorch to introduce modern neural network implementation.

The focus is on understanding how a neural network computes its output. Gradient calculation, backpropagation, and parameter optimization are covered in subsequent modules.

## 2. Learning Objectives

By completing this module, you will be able to:

- Explain the structure and purpose of an artificial neuron.
- Understand weights, biases, parameters, inputs, and outputs.
- Calculate the weighted sum and output of a neuron.
- Explain forward propagation through one or more layers.
- Understand the structure of a multilayer perceptron.
- Distinguish linear transformations from nonlinear neural network representations.
- Implement basic neural network computations using NumPy.
- Build equivalent neural network structures using PyTorch.

## 3. Topics and Practical Exercises

| #   | Topic                 | Key Concepts                                                                       |
| --- | --------------------- | ---------------------------------------------------------------------------------- |
| 01  | Artificial Neuron     | Inputs, weights, bias, weighted sum, activation, prediction                        |
| 02  | Forward Propagation   | Layer computations, matrix multiplication, bias broadcasting, tensor shapes        |
| 03  | Multilayer Perceptron | Input layer, hidden layers, output layer, parameters, nonlinear representations    |
| 04  | NumPy vs. PyTorch     | Manual computations, tensors, `nn.Linear`, activation functions, output comparison |

## 4. Key Concepts

### Artificial Neuron

An artificial neuron computes a weighted sum of its inputs, adds a bias, and applies an activation function.

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

$$
a = f(z)
$$

Here, x represents the inputs, w represents the weights, b is the bias, z is the pre-activation value, and f is the activation function.

### Forward Propagation

Forward propagation passes input data through the network layer by layer to produce predictions.

For a layer:

$$
Z = XW + b
$$

$$
A = f(Z)
$$

The exact matrix dimensions depend on the chosen convention. In this module, batches of input samples are represented by rows of X.

### Multilayer Perceptron

An MLP consists of an input representation, one or more hidden layers, and an output layer. Hidden-layer activation functions allow the network to represent nonlinear relationships.

Without nonlinear activation functions between layers, stacking ordinary linear transformations still produces a linear transformation.

### Parameters

Weights and biases are learnable parameters. The network architecture determines how these parameters are organized, while training methods determine how they are updated.

## 5. Best Practices

- Check tensor shapes and matrix dimensions carefully.
- Distinguish pre-activation values from activated outputs.
- Use simple numerical examples to verify calculations.
- Compare NumPy and PyTorch outputs when implementing equivalent computations.
- Introduce nonlinear activation functions when learning nonlinear relationships.
- Keep forward propagation separate from gradient calculation and optimization.
- Use clear, reproducible examples with descriptive variable names.

## 6. References

- [Dive into Deep Learning: Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/index.html)
- [PyTorch: `torch.nn`](https://pytorch.org/docs/stable/nn.html)
- [PyTorch: Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
