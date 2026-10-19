# 08 - Regularization and Initialization

## Learning Objectives

- Understand how weight initialization affects activation variance, gradient propagation, and training stability.
- Apply Xavier (Glorot) and He (Kaiming) initialization based on activation functions.
- Understand how L1 and L2 regularization influence model complexity and learned parameters.
- Distinguish L2 regularization from decoupled weight decay in optimizers such as AdamW.
- Apply dropout and batch normalization correctly in PyTorch.
- Evaluate regularization techniques using training loss, validation loss, and the generalization gap.

## Core Concepts

### 1. Weight Initialization

Neural networks learn by updating their weights through gradient-based optimization. However, unsuitable initial weights can cause activations or gradients to become excessively small or large as they propagate through deep networks.

- **Default initialization:** PyTorch layers use their own default initialization strategies. Understanding alternatives helps when designing deeper networks.
- **Xavier (Glorot) initialization:** Designed to help maintain activation and gradient variance across layers. It is commonly suitable for tanh and linear activations.
- **He (Kaiming) initialization:** Designed for ReLU-family activations, accounting for the fact that ReLU sets negative inputs to zero.

For a layer with $$n_{\text{in}}$$ input units and $$n_{\text{out}}$$ output units, Xavier uniform initialization samples weights from:

$$
W \sim U\left(
-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},
\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}
\right)
$$

He initialization for ReLU commonly uses a normal distribution with:

$$
W \sim \mathcal{N}\left(0,\frac{2}{n_{\text{in}}}\right)
$$

The goal is to preserve useful signal variance through the network, making optimization more stable. No initialization strategy is universally optimal for every architecture and activation function.

### 2. L1 and L2 Regularization

Regularization adds a penalty to the training objective to discourage overly complex models or excessively large parameters. It can help reduce **overfitting**, which occurs when a model performs well on training data but poorly on unseen data.

Let:

- $$L_{\text{data}}$$ represent the original data loss.
- $$W$$ represent the model's trainable weights.
- $$\lambda$$ control the regularization strength.

**L1 regularization** penalizes the absolute values of weights:

$$
L_{\text{total}} =
L_{\text{data}}+\lambda\sum_i |W_i|
$$

Its main characteristic is that it can encourage **sparsity**, pushing some weights toward zero. This can make a model more compact or easier to interpret, although sparsity is not guaranteed in every situation.

**L2 regularization** penalizes the squared values of weights:

$$
L_{\text{total}} =
L_{\text{data}}+\lambda\sum_i W_i^2
$$

L2 regularization discourages large weights and often distributes learning across many parameters rather than explicitly encouraging sparse solutions.

The regularization coefficient $$\lambda$$ controls the trade-off between fitting the training data and limiting parameter magnitudes.

- **Too little regularization:** The model may memorize training patterns and perform poorly on unseen data.
- **Too much regularization:** The model may be unable to learn important patterns, resulting in underfitting.
- **Appropriate regularization:** The model learns useful patterns while maintaining good generalization.

In practice, regularization penalties are often applied to weights rather than biases, depending on the model and implementation.

### 3. Weight Decay: L2 Regularization vs. AdamW

Weight decay reduces the magnitude of model parameters during optimization. Although the terms _L2 regularization_ and _weight decay_ are sometimes used interchangeably, their implementations can behave differently.

- **L2 regularization:** Adds a penalty to the loss function. The penalty's gradient contributes to the optimizer's update.
- **Decoupled weight decay:** Applies weight shrinkage separately from the loss-gradient update.
- **AdamW:** Implements decoupled weight decay and is widely used to train neural networks and transformer models.

For standard SGD, L2 regularization and weight decay can produce equivalent updates when their coefficients are matched. With adaptive optimizers such as Adam, adding an L2 penalty to the loss is generally not equivalent to decoupled weight decay.

A simplified decoupled weight-decay update is:

$$
W_{t+1} =
(1-\eta\lambda)W_t-\eta U_t
$$

Here:

- $$W_t$$ is the current parameter value.
- $$W_{t+1}$$ is the updated parameter value.
- $$\eta$$ is the learning rate.
- $$\lambda$$ is the weight-decay coefficient.
- $$U_t$$ represents the optimizer's update direction derived from the data gradient.

The exact update depends on the optimizer and its configuration.

The practical lesson is to understand what an optimizer's weight-decay parameter actually does rather than assuming that every implementation of L2 regularization behaves identically.

### 4. Dropout

Dropout randomly sets a proportion of activations to zero during training. It helps prevent the network from relying too heavily on particular neurons and can improve generalization.

For an activation vector $$h$$, inverted dropout can be expressed as:

$$
\tilde{h}=\frac{m\odot h}{1-p}
$$

Here:

- $$h$$ represents the original activations.
- $$\tilde{h}$$ represents the activations after dropout.
- $$m$$ is a random mask whose elements are one with probability $$1-p$$ and zero with probability $$p$$.
- $$p$$ is the dropout probability.
- $$\odot$$ denotes element-wise multiplication.

The scaling factor preserves the expected activation magnitude during training.

In PyTorch:

- `model.train()` enables dropout.
- `model.eval()` disables dropout.

A high dropout probability can make learning difficult, so its value should be selected through experimentation and validation.

Dropout is not automatically beneficial for every architecture. Its effectiveness depends on the model, dataset, and other regularization techniques being used.

### 5. Batch Normalization

Batch normalization normalizes intermediate activations using statistics calculated from a mini-batch during training. This can make optimization more stable and often enables effective training with deeper networks.

For an activation $$x$$, the normalization step is:

$$
\hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

Here:

- $$\mu_B$$ is the mean of the current mini-batch.
- $$\sigma_B^2$$ is the variance of the current mini-batch.
- $$\epsilon$$ is a small constant that improves numerical stability.
- $$\hat{x}$$ is the normalized activation.

Batch normalization then applies a learnable scale and shift:

$$
y=\gamma\hat{x}+\beta
$$

Here:

- $$\gamma$$ is a learnable scaling parameter.
- $$\beta$$ is a learnable shifting parameter.
- $$y$$ is the resulting activation.

The learnable parameters allow the network to adjust the normalized values when needed.

In PyTorch:

- `model.train()` uses batch statistics and updates running estimates.
- `model.eval()` uses stored running statistics for inference.

**Important considerations:**

- Batch normalization is not simply another form of weight penalty. Its primary purpose is to normalize activations.
- Very small batch sizes can make batch statistics unreliable.
- With `BatchNorm1d` applied to a batch of feature vectors, training generally requires more than one value per channel.
- Batch normalization and dropout solve different problems, and combining them is not always necessary or beneficial.

### 6. Understanding Generalization

Regularization techniques should be evaluated using both training and validation performance.

The generalization gap can be expressed as:

$$
G=L_{\text{validation}}-L_{\text{training}}
$$

A large positive gap often indicates that the model fits the training data substantially better than the validation data.

However, the gap should not be interpreted in isolation. Both losses matter:

- **Low training loss and high validation loss:** Often indicates overfitting.
- **High training loss and high validation loss:** Often indicates underfitting or optimization problems.
- **Low training loss and low validation loss:** Usually indicates a good fit, assuming the validation data represents the intended use case.

Regularization should improve performance on unseen data, not merely reduce the training loss.

## Implementation and Experiments

1. **Weight Initialization** (`01_weight_initialization.py`)
   - Compare PyTorch's default initialization, Xavier initialization, and He initialization.
   - Measure activation variance and gradient norms across networks of different depths.
   - Examine the interaction between initialization strategies and activation functions such as tanh and ReLU.

2. **L1, L2 Regularization, and Weight Decay** (`02_l1_l2_regularization_and_weight_decay.py`)
   - Implement L1 and L2 penalties in the training loss.
   - Compare parameter magnitudes and training versus validation performance.
   - Compare Adam with AdamW and investigate the differences between loss-based penalties and decoupled weight decay.

3. **Dropout and Batch Normalization** (`03_dropout_and_batch_normalization.py`)
   - Build PyTorch models using dropout and batch normalization.
   - Observe their effects on training dynamics and validation performance.
   - Verify training and evaluation behavior and investigate batch-size considerations.

4. **Regularization Comparison** (`04_regularization_comparison.py`)
   - Compare a baseline model with selected regularization strategies.
   - Track training loss, validation loss, and the generalization gap.
   - Identify overfitting and underfitting, and evaluate the trade-offs of each technique.

## Expected Outcome

After completing this module, you should be able to:

- Select suitable weight initialization strategies for common activation functions.
- Explain the mathematical and practical differences between L1 and L2 regularization.
- Choose between explicit loss penalties and decoupled weight decay.
- Apply dropout and batch normalization correctly in PyTorch.
- Interpret learning curves and validation metrics to assess generalization.
- Make evidence-based decisions about regularization rather than assuming one technique works best for every neural network.
