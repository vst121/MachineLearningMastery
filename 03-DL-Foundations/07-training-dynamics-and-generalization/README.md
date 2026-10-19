# Training Dynamics and Generalization

This module explores how neural networks learn over time, how to interpret training behavior, and how to determine whether a model generalizes beyond its training data.

## Learning Objectives

- Understand epochs, batches, iterations, and parameter updates.
- Interpret training and validation learning curves.
- Diagnose overfitting, underfitting, and generalization gaps.
- Analyze convergence, optimization plateaus, and unstable training.
- Understand vanishing and exploding gradients.
- Apply early stopping, gradient clipping, and validation-based model selection.

## Core Concepts

| Concept             | Description                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| Epoch               | One complete pass through the training dataset.                                     |
| Batch               | A subset of training samples processed together.                                    |
| Iteration           | One batch-processing step; typically one optimizer update during standard training. |
| Learning Curve      | Training or validation metrics tracked over training time.                          |
| Generalization      | The ability to perform well on unseen data drawn from the target distribution.      |
| Overfitting         | Learning training-specific patterns that do not generalize well.                    |
| Underfitting        | Failing to learn patterns sufficiently from the training data.                      |
| Convergence         | The optimization process approaches a stable solution or region.                    |
| Vanishing Gradients | Gradients become too small for effective parameter updates.                         |
| Exploding Gradients | Gradients become excessively large, causing unstable updates.                       |
| Early Stopping      | Stopping training when validation performance no longer improves sufficiently.      |
| Gradient Clipping   | Limiting gradient magnitude to reduce excessively large updates.                    |

## Implementation and Experiments

### 1. Epochs, Batches, and Learning Curves

**File:** `01_epochs_batches_and_learning_curves.py`

- Compare batch sizes and their effect on optimizer updates.
- Calculate iterations per epoch.
- Track training and validation loss and accuracy.
- Visualize learning curves and interpret changes over time.

### 2. Overfitting, Underfitting, and Generalization

**File:** `02_overfitting_underfitting_and_generalization.py`

- Demonstrate underfitting and overfitting.
- Compare training and validation performance.
- Analyze the generalization gap.
- Implement early stopping and restore the best validation checkpoint.
- Understand validation-based model selection.

### 3. Convergence and Learning Dynamics

**File:** `03_convergence_and_learning_dynamics.py`

- Compare convergence under different learning rates.
- Identify slow convergence, plateaus, and unstable optimization.
- Observe loss oscillations and divergence.
- Relate learning dynamics to optimizer behavior and learning-rate scheduling.

### 4. Vanishing and Exploding Gradients

**File:** `04_vanishing_and_exploding_gradients.py`

- Measure gradient norms during backpropagation.
- Investigate how network depth and activation functions affect gradient flow.
- Demonstrate vanishing and exploding gradient behavior.
- Apply gradient clipping and observe its effects.

## Expected Outcome

After completing this module, you should be able to interpret neural network training behavior, diagnose common optimization and generalization problems, and implement practical techniques to improve training stability and model selection.
