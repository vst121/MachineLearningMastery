# Regularization and Initialization

## Learning Objectives

- Understand how weight initialization affects activation variance, gradient propagation, and training stability.
- Apply Xavier (Glorot) and He (Kaiming) initialization based on the activation function.
- Understand how L1 and L2 regularization influence model complexity and learned parameters.
- Distinguish L2 regularization from decoupled weight decay in optimizers such as AdamW.
- Apply dropout and batch normalization correctly in PyTorch.
- Evaluate regularization techniques using training loss, validation loss, and the generalization gap.

## Core Concepts

### 1. Weight Initialization

Neural networks learn by updating their weights through gradient-based optimization. However, training can become difficult if weights are initialized with unsuitable values.

- **Default initialization:** PyTorch layers use their own default initialization strategies. These are convenient, but understanding alternatives helps when designing deeper networks.
- **Xavier (Glorot) initialization:** Designed to help maintain activation and gradient variance across layers. It is commonly suitable for tanh and linear activations.
- **He (Kaiming) initialization:** Designed for ReLU-family activations, accounting for the fact that ReLU sets negative inputs to zero.

For a layer with \(n*{\text{in}}\) input units and \(n*{\text{out}}\) output units, Xavier uniform initialization samples weights from:

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

The objective is not to find universally optimal initial weights, but to avoid activations or gradients becoming excessively small or large as they propagate through a network.

### 2. L1 and L2 Regularization

Regularization adds a penalty to the training objective to discourage overly complex models or excessively large parameters. It can help reduce overfitting, where a model performs well on training data but poorly on unseen data.

Let:

- \(L\_{\text{data}}\) be the original data loss.
- \(W\) represent the model's trainable weights.
- \(\lambda\) control the regularization strength.

**L1 regularization** penalizes the absolute values of weights:

$$
L_{\text{total}} =
L_{\text{data}}+\lambda\sum_i |W_i|
$$

Its key characteristic is that it can encourage sparsity, pushing some weights toward exactly zero. Sparse models can sometimes be easier to interpret, although L1 does not guarantee that a model will become sparse.

**L2 regularization** penalizes the squared values of weights:

$$
L_{\text{total}} =
L_{\text{data}}+\lambda\sum_i W_i^2
$$

L2 regularization discourages large weights and generally distributes learning across many parameters rather than explicitly encouraging sparsity.

The parameter \(\lambda\) controls the trade-off:

- Too little regularization may leave the model prone to overfitting.
- Too much regularization may prevent the model from learning important patterns, causing underfitting.

In practice, regularization penalties are often applied to weights rather than biases, depending on the model and implementation.

### 3. Weight Decay: L2 Regularization vs. AdamW

Weight decay reduces the magnitude of model parameters during optimization.

With standard SGD, L2 regularization and weight decay can produce equivalent parameter updates when their coefficients are matched. With adaptive optimizers such as Adam, adding an L2 penalty to the loss is generally not equivalent to decoupled weight decay.

- **L2 regularization:** Adds a penalty to the loss, so its gradient contributes to the optimizer's update.
- **Decoupled weight decay:** Applies weight shrinkage separately from the loss-gradient update.
- **AdamW:** Implements decoupled weight decay and is widely used for training neural networks and transformer models.

A simplified decoupled weight-decay update is:

$$
W_{t+1} =
(1-\eta\lambda)W_t-\eta\,U_t
$$

Here, \(\eta\) is the learning rate and \(U_t\) is the optimizer's update direction derived from the data gradient. The exact update depends on the optimizer.

The practical lesson is to understand what an optimizer's weight-decay parameter actually does rather than assuming that every implementation of L2 regularization behaves identically.

### 4. Dropout

Dropout randomly sets a proportion of activations to zero during training.

If \(h\) represents a layer's activations, inverted dropout can be expressed as:

$$
\tilde{h}=\frac{m\odot h}{1-p}
$$

where:

- \(m\) is a random mask that keeps each activation with probability \(1-p\).
- \(p\) is the dropout probability.
- \(\odot\) denotes element-wise multiplication.

The scaling keeps the expected activation magnitude consistent between training and inference.

Dropout discourages a network from relying too heavily on particular neurons. It can improve generalization, but excessive dropout can make learning difficult.

In PyTorch:

- `model.train()` enables dropout.
- `model.eval()` disables dropout.

Dropout should be evaluated experimentally. It is not automatically beneficial for every architecture, especially when other forms of regularization are already effective.

### 5. Batch Normalization

Batch normalization normalizes intermediate activations using statistics calculated from a mini-batch during training.

For an activation \(x\), the normalization step is:

$$
\hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

A learnable scale and shift are then applied:

$$
y=\gamma\hat{x}+\beta
$$

Here, \(\mu_B\) and \(\sigma_B^2\) are the batch mean and variance, \(\epsilon\) improves numerical stability, and \(\gamma\) and \(\beta\) are trainable parameters.

Batch normalization can make optimization more stable and often permits more effective training. Its behavior depends on the batch size and data distribution.

In PyTorch:

- `model.train()` uses batch statistics and updates running estimates.
- `model.eval()` uses the stored running statistics.

**Important:** Batch normalization is not simply another form of weight penalty. Its primary purpose is to normalize activations, although its interaction with mini-batches can sometimes influence generalization.

## Implementation and Experiments

1. **Weight Initialization** (`01_weight_initialization.py`)
   - Compare PyTorch's default initialization, Xavier, and He initialization.
   - Measure activation variance and gradient norms across networks of different depths.
   - Examine the interaction between initialization and activation functions such as tanh and ReLU.

2. **L1, L2 Regularization, and Weight Decay** (`02_l1_l2_regularization_and_weight_decay.py`)
   - Implement L1 and L2 penalties in the training loss.
   - Compare parameter magnitudes and training versus validation performance.
   - Compare Adam with AdamW and discuss why their weight-decay behavior differs.

3. **Dropout and Batch Normalization** (`03_dropout_and_batch_normalization.py`)
   - Build PyTorch models with dropout and batch normalization.
   - Observe their impact on training dynamics and validation performance.
   - Verify train/evaluation behavior and examine batch-size considerations.

4. **Regularization Comparison** (`04_regularization_comparison.py`)
   - Compare a baseline model with selected regularization strategies.
   - Track training loss, validation loss, and the generalization gap.
   - Identify cases of overfitting and underfitting, and assess the trade-offs of each technique.

## Expected Outcome

After completing this module, you should be able to:

- Select a suitable weight initialization strategy for common activation functions.
- Explain the mathematical and practical differences between L1 and L2 regularization.
- Choose between explicit loss penalties and decoupled weight decay.
- Apply dropout and batch normalization correctly in PyTorch.
- Interpret learning curves and validation metrics to determine whether regularization improves generalization.
