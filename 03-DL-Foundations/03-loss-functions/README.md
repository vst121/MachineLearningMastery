# Loss Functions

Loss functions quantify the difference between a model's predictions and the target values. They define the objective that a neural network aims to minimize during training.

Different tasks require different loss functions. Regression commonly uses error-based losses, while classification often uses cross-entropy losses.

## Learning Objectives

- Understand the role of loss functions in neural network training.
- Implement common regression and classification losses.
- Explain the mathematical foundations of MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy, and Huber Loss.
- Understand how outliers affect different regression losses.
- Distinguish logits, predicted probabilities, target labels, and loss values.
- Select appropriate loss functions for different machine learning tasks.

## Implementations and Experiments

| File                              | Topics                                                               |
| --------------------------------- | -------------------------------------------------------------------- |
| `01_mean_squared_error.py`        | Mean Squared Error, squared prediction errors, regression            |
| `02_mean_absolute_error.py`       | Mean Absolute Error, absolute errors, outlier sensitivity            |
| `03_binary_cross_entropy.py`      | Binary classification, predicted probabilities, log loss             |
| `04_categorical_cross_entropy.py` | Multiclass classification, class probabilities, target distributions |
| `05_huber_loss.py`                | Huber Loss, transition between squared and absolute errors           |
| `06_loss_comparison.py`           | Comparative experiments, outlier effects, task-dependent selection   |

## 1. The Role of a Loss Function

A loss function measures how well a model's predictions match the target values.

For a dataset containing \(N\) samples, the average loss is commonly expressed as:

$$
L = \frac{1}{N}\sum_{i=1}^{N}\ell(y_i,\hat{y}_i)
$$

where:

- \(y_i\) is the true target.
- \(\hat{y}\_i\) is the model prediction.
- \(\ell\) is the loss for an individual sample.
- \(N\) is the number of samples.
- \(L\) is the average loss.

The goal of training is generally to find model parameters that minimize the chosen objective.

**Important distinction:** The loss function defines what the model should minimize. An optimizer determines how model parameters are updated to reduce that loss.

## 2. Mean Squared Error (MSE)

MSE calculates the average squared difference between predictions and target values.

$$
L_{\text{MSE}} =
\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

**Properties**

- Commonly used for regression.
- Penalizes large errors more heavily than small errors.
- Is differentiable with respect to predictions.
- Is sensitive to outliers because errors are squared.
- Expresses the loss in squared target units.

**Typical use cases:** Continuous-value prediction, such as estimating prices, temperatures, or demand.

## 3. Mean Absolute Error (MAE)

MAE calculates the average absolute difference between predictions and target values.

$$
L_{\text{MAE}} =
\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|
$$

**Properties**

- Commonly used for regression.
- Measures errors in the original target units.
- Is generally less sensitive to outliers than MSE.
- Is not differentiable at zero error, although suitable subgradients can be used.
- Penalizes errors linearly rather than quadratically.

**Typical use cases:** Regression tasks where a large individual error should not dominate the entire loss.

## 4. Binary Cross-Entropy (BCE)

Binary Cross-Entropy measures the difference between binary target labels and predicted probabilities.

$$
L_{\text{BCE}} =
-\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
$$

where:

- \(y_i \in \{0,1\}\) is the target label.
- \(p_i \in (0,1)\) is the predicted probability of class 1.

**Properties**

- Used for binary classification.
- Strongly penalizes confident predictions that are incorrect.
- Requires probabilities when using the probability-based formula.
- Is commonly paired with a sigmoid output.
- Can be computed directly from logits using numerically stable implementations.

**Typical use cases:** Spam detection, fraud classification, and binary image classification.

For numerical stability, implementations should avoid directly evaluating logarithms at zero. In practice, a logits-based BCE implementation is often preferable when the model outputs raw logits.

## 5. Categorical Cross-Entropy

Categorical Cross-Entropy measures the difference between a target class distribution and a predicted probability distribution.

$$
L_{\text{CCE}} =
-\frac{1}{N}\sum_{i=1}^{N}
\sum_{k=1}^{K}y_{ik}\log(p_{ik})
$$

where:

- \(N\) is the number of samples.
- \(K\) is the number of classes.
- \(y\_{ik}\) is the target value for class \(k\) of sample \(i\).
- \(p\_{ik}\) is the predicted probability for that class.

For one-hot targets, only the term corresponding to the correct class contributes to the sample's loss:

$$
\ell_i = -\log(p_{i,\text{true class}})
$$

**Properties**

- Used for mutually exclusive multiclass classification.
- Commonly paired with a Softmax output.
- Penalizes assigning low probability to the correct class.
- Can support probability distributions as targets, including label smoothing.

**Typical use cases:** Multiclass image classification, document categorization, and language modeling.

Numerically stable implementations often calculate cross-entropy directly from logits rather than applying Softmax and logarithms separately.

## 6. Huber Loss

Huber Loss combines squared-error behavior for small errors with absolute-error behavior for large errors.

Let the residual be:

$$
r = y-\hat{y}
$$

The Huber Loss for one sample is:

$$
L_\delta(r)=
\begin{cases}
\frac{1}{2}r^2 & \text{if } |r|\leq\delta \\
\delta\left(|r|-\frac{1}{2}\delta\right)
& \text{if } |r|>\delta
\end{cases}
$$

where \(\delta>0\) controls the transition between the two regions.

The mean Huber Loss is:

$$
L_{\text{Huber}} =
\frac{1}{N}\sum_{i=1}^{N}L_\delta(y_i-\hat{y}_i)
$$

**Properties**

- Quadratic for small errors.
- Linear for large errors.
- Less sensitive to extreme errors than MSE.
- Differentiable at the transition points.
- Requires choosing a suitable \(\delta\).

**Typical use cases:** Regression problems where small errors should be treated smoothly but outliers should have limited influence.

## 7. Comparison of Loss Functions

| Loss Function             | Task                      | Main Advantage                                   | Main Limitation                                             |
| ------------------------- | ------------------------- | ------------------------------------------------ | ----------------------------------------------------------- |
| MSE                       | Regression                | Strong penalty for large errors                  | Sensitive to outliers                                       |
| MAE                       | Regression                | More robust to outliers                          | Not differentiable at zero residual                         |
| Binary Cross-Entropy      | Binary classification     | Evaluates predicted binary probabilities         | Requires numerically careful probability or logits handling |
| Categorical Cross-Entropy | Multiclass classification | Evaluates probability assigned to target classes | Must match the target and output representation             |
| Huber Loss                | Regression                | Balances quadratic and linear penalties          | Requires a transition parameter                             |

## 8. Implementation and Experiments

The Python implementations use NumPy and Matplotlib where appropriate.

Experiments include:

- Calculating regression losses for sample predictions.
- Comparing MSE and MAE for the same residuals.
- Demonstrating how outliers affect regression losses.
- Calculating Binary Cross-Entropy for different predicted probabilities.
- Comparing multiclass predictions using Categorical Cross-Entropy.
- Investigating Huber Loss at different residual magnitudes.
- Comparing loss functions and selecting them according to the task.

## 9. Key Takeaways

- A loss function defines the objective that a model attempts to minimize.
- MSE penalizes large errors quadratically, while MAE penalizes them linearly.
- Huber Loss combines both behaviors.
- Cross-Entropy evaluates predictions in terms of their assigned probabilities for target classes.
- Output activations, target encoding, and loss functions must be compatible.
- A lower training loss does not automatically guarantee better generalization to unseen data.

## 10. Scope and Prerequisites

This module builds on `01-neural-network-fundamentals` and `02-activation-functions`.

The focus is on loss definitions, mathematical properties, implementation, and empirical comparisons. Gradient derivations, backpropagation, and optimization algorithms remain dedicated topics later in the course.
