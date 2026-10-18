# 🧠 Deep Learning Foundations

## 1. Overview

This module introduces the fundamental concepts, mathematical principles, and practical techniques required to understand and build neural networks using Python and PyTorch.

It focuses on how neural networks represent functions, calculate predictions, measure errors, compute gradients, and optimize their parameters during training.

The module combines mathematical explanations, NumPy implementations, PyTorch experiments, and practical neural network projects to establish a strong foundation for advanced deep learning, NLP, Transformers, and AI Engineering.

## 2. Learning Objectives

By completing this module, you will be able to:

- Understand neurons, layers, weights, biases, and neural network architectures.
- Explain forward propagation, activation functions, and loss functions.
- Understand backpropagation, computational graphs, and automatic differentiation.
- Apply gradient-based optimization techniques to train neural networks.
- Work with PyTorch tensors, autograd, modules, optimizers, and data loaders.
- Diagnose overfitting, underfitting, gradient problems, and poor convergence.
- Apply regularization and weight initialization techniques.
- Build, train, evaluate, save, and load neural networks for regression and classification.

## 3. Topics and Practical Exercises

| #   | Topic                                | Key Concepts                                                                                                           |
| --- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| 01  | Neural Network Fundamentals          | Neurons, weights, biases, layers, parameters, forward propagation, multilayer perceptrons                              |
| 02  | Activation Functions                 | Sigmoid, tanh, ReLU, Leaky ReLU, softmax, nonlinear transformations                                                    |
| 03  | Loss Functions                       | Mean squared error, binary cross-entropy, multiclass cross-entropy, logits, prediction errors                          |
| 04  | Backpropagation and Autograd         | Derivatives, chain rule, computational graphs, gradient calculation, automatic differentiation                         |
| 05  | Gradient-Based Optimization          | Gradient descent, stochastic gradient descent, mini-batches, learning rates, momentum, Adam                            |
| 06  | PyTorch Fundamentals                 | Tensors, tensor operations, `nn.Module`, autograd, optimizers, datasets, data loaders, training loops                  |
| 07  | Training Dynamics and Generalization | Epochs, batches, learning curves, overfitting, underfitting, convergence, vanishing and exploding gradients            |
| 08  | Regularization and Initialization    | Xavier initialization, He initialization, L1/L2 regularization, weight decay, dropout, batch normalization             |
| 09  | Practical Neural Network Projects    | MLP regression, binary classification, multiclass classification, model persistence, NumPy and PyTorch implementations |

## 4. Key Concepts

### Neural Network Computation

A neural network transforms input data through successive layers using learnable weights, biases, and activation functions.

For a single neuron:

$$
z = wx + b
$$

$$
a = f(z)
$$

where x is the input, w is the weight, b is the bias, and f is the activation function.

### Learning and Optimization

Training consists of computing predictions, evaluating a loss function, calculating gradients, and updating model parameters.

The basic gradient descent update is:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

where theta represents the model parameters, eta is the learning rate, and L is the loss function.

### Backpropagation and Automatic Differentiation

Backpropagation applies the chain rule to calculate gradients throughout a neural network. PyTorch's automatic differentiation system computes these gradients efficiently, allowing us to focus on model design and training.

### Generalization

A well-trained neural network should perform effectively on unseen data, not merely memorize its training examples. Validation, regularization, appropriate initialization, and careful monitoring help achieve this goal.

## 5. Best Practices

- Understand the mathematical principles behind each technique before relying on framework abstractions.
- Use NumPy for small implementations when it helps explain neural network computations.
- Use PyTorch as the primary framework for practical model development.
- Keep training, validation, and test data separate.
- Establish simple baselines before increasing model complexity.
- Monitor training and validation losses to identify overfitting and convergence problems.
- Use reproducible experiments with controlled random seeds where appropriate.
- Choose activation functions, output layers, and loss functions that match the task.
- Save model parameters and the configuration needed to reproduce inference.
- Prefer clear, modular, testable code over unnecessarily complex training abstractions.

## 6. References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Dive into Deep Learning](https://d2l.ai/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/)
