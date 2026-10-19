# Optimization Algorithms

## Overview

Optimization algorithms update neural network parameters to minimize a loss function. They use gradients calculated through backpropagation to determine how weights and biases should change during training.

This module covers gradient descent, stochastic optimization, momentum, adaptive optimizers, learning-rate scheduling, and practical optimizer comparisons.

## Learning Objectives

- Understand gradient descent and parameter updates.
- Compare batch, stochastic, and mini-batch gradient descent.
- Understand learning rates and their effects on convergence.
- Implement momentum and adaptive optimization algorithms.
- Use PyTorch optimizers to train neural networks.
- Apply learning-rate schedules.
- Compare optimizers using loss curves and convergence behavior.

## Core Concepts

### 1. Gradient Descent

Gradient descent updates parameters in the direction opposite to the gradient.

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

Where:

- theta represents model parameters.
- eta represents the learning rate.
- The gradient indicates the direction of greatest local increase in loss.

### 2. Batch, Stochastic, and Mini-Batch Gradient Descent

- **Batch Gradient Descent:** Calculates gradients using the entire training dataset.
- **Stochastic Gradient Descent:** Calculates gradients using one training example at a time.
- **Mini-Batch Gradient Descent:** Calculates gradients using a small subset of training examples.

Mini-batch training balances computational efficiency and gradient stability.

### 3. Learning Rate

The learning rate controls the size of each parameter update.

- Too small: training may converge slowly.
- Too large: training may oscillate or diverge.
- Appropriate: training can converge efficiently.

### 4. Momentum

Momentum incorporates previous gradients to smooth optimization and accelerate progress in consistent directions.

$$
v_t = \beta v_{t-1} + \nabla_\theta L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \eta v_t
$$

### 5. Adaptive Optimizers

Adaptive optimizers adjust parameter updates using gradient statistics.

- **RMSprop:** Scales updates using an exponential moving average of squared gradients.
- **Adam:** Combines first-moment and second-moment estimates with bias correction.
- **AdamW:** Decouples weight decay from the gradient-based update.

No optimizer is universally best. Performance depends on the model, data, learning rate, and training objective.

### 6. Learning-Rate Scheduling

A learning-rate scheduler changes the learning rate during training.

Common strategies include:

- Step decay
- Exponential decay
- Cosine annealing
- Reduce learning rate when a monitored metric stops improving

### 7. Optimizer Evaluation

Optimization algorithms should be compared using consistent datasets, model architectures, initialization, and evaluation conditions.

Useful measurements include:

- Training and validation loss
- Convergence speed
- Training stability
- Final validation performance
- Computational cost

## Implementation and Experiments

| File                              | Description                                                                |
| --------------------------------- | -------------------------------------------------------------------------- |
| `01_gradient_descent.py`          | Implement gradient descent on a mathematical function                      |
| `02_batch_sgd_minibatch.py`       | Compare batch, stochastic, and mini-batch gradient descent                 |
| `03_learning_rate_experiments.py` | Examine how different learning rates affect convergence                    |
| `04_momentum.py`                  | Implement momentum and compare it with standard gradient descent           |
| `05_adaptive_optimizers.py`       | Compare SGD, RMSprop, Adam, and AdamW using PyTorch                        |
| `06_learning_rate_schedulers.py`  | Experiment with step, exponential, and cosine learning-rate schedules      |
| `07_optimizer_comparison.py`      | Compare optimizer convergence using a consistent neural network experiment |

## Key Takeaways

- Backpropagation calculates gradients; optimizers use them to update parameters.
- The learning rate strongly influences convergence and training stability.
- Momentum can accelerate progress and reduce oscillation.
- Adaptive optimizers adjust updates using gradient statistics.
- Learning-rate schedules can improve convergence during training.
- Optimizer selection should be based on experiments and validation performance.
