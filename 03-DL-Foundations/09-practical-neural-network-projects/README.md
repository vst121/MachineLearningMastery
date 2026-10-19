# Practical Neural Network Projects

## Learning Objectives

- Build complete neural network projects using PyTorch.
- Prepare datasets, define models, train them, and evaluate their performance.
- Apply appropriate loss functions, optimizers, and evaluation metrics.
- Use validation data for model selection and test data for final evaluation.
- Apply regularization, learning-rate scheduling, and early stopping when appropriate.
- Save trained models and build reproducible inference pipelines.
- Diagnose common training problems using learning curves and evaluation metrics.

## Core Concepts

### 1. End-to-End Neural Network Workflow

A practical neural network project involves more than defining a model. It requires a complete workflow:

1. Define the problem and select evaluation metrics.
2. Prepare the dataset and split it into training, validation, and test sets.
3. Build a model suitable for the task.
4. Select an appropriate loss function and optimizer.
5. Train the model and monitor learning curves.
6. Tune hyperparameters using validation performance.
7. Evaluate the selected model on the held-out test set.
8. Save the model and implement inference.

Keeping these steps separate makes experiments easier to reproduce, test, and maintain.

### 2. Regression

Regression predicts continuous numerical values, such as prices, demand, or energy consumption.

Common components include:

- **Output layer:** Usually produces one or more continuous values without a classification activation.
- **Loss function:** Mean Squared Error (MSE) or Mean Absolute Error (MAE).
- **Evaluation metrics:** MAE, RMSE, and the coefficient of determination, $$R^2$$.
- **Common challenges:** Outliers, feature scaling, nonlinear relationships, and overfitting.

The Mean Squared Error is:

$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

Here, $$y_i$$ is the actual target, $$\hat{y}_i$$ is the predicted value, and $$N$$ is the number of evaluated samples.

### 3. Binary Classification

Binary classification predicts one of two classes, such as fraud versus legitimate transactions.

Common components include:

- **Output layer:** One logit representing the positive class.
- **Loss function:** `BCEWithLogitsLoss`, which combines sigmoid and binary cross-entropy in a numerically stable implementation.
- **Evaluation metrics:** Precision, recall, F1-score, confusion matrix, and ROC-AUC.
- **Common challenges:** Class imbalance, threshold selection, and distinguishing probability estimation from class prediction.

For a binary target, binary cross-entropy is:

$$
L = -\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i\log(p_i)+(1-y_i)\log(1-p_i)
\right]
$$

Here, $$y_i$$ is either zero or one, and $$p_i$$ is the predicted probability of the positive class.

When using `BCEWithLogitsLoss`, the model should return logits rather than applying sigmoid inside the model. Convert logits into probabilities for inference and threshold them to obtain class predictions.

### 4. Multiclass Classification

Multiclass classification assigns each input to one of several mutually exclusive classes.

Common components include:

- **Output layer:** One logit per class.
- **Loss function:** `CrossEntropyLoss`.
- **Evaluation metrics:** Accuracy, per-class precision and recall, macro F1-score, and a confusion matrix.
- **Common challenges:** Class imbalance, misclassification between similar classes, and interpreting per-class performance.

For a sample with the correct class indexed by $$y$$, cross-entropy is:

$$
L = -\log(p_y)
$$

Here, $$p_y$$ is the predicted probability assigned to the correct class. PyTorch's `CrossEntropyLoss` expects raw logits and performs the required log-softmax calculation internally.

### 5. Reproducibility and Model Evaluation

A reliable experiment must separate model training from model selection and final evaluation.

- **Training set:** Used to optimize model parameters.
- **Validation set:** Used to compare configurations, tune hyperparameters, and select checkpoints.
- **Test set:** Used for the final performance estimate after model selection.

Additional practices include setting random seeds, recording metrics, saving model parameters, preserving preprocessing steps, and checking for data leakage.

The best model is not necessarily the one with the lowest training loss. The objective is to learn patterns that generalize to unseen data.

## Implementation and Experiments

1. **Regression Project** (`01_neural_network_regression.py`)
   - Generate or prepare a nonlinear regression dataset.
   - Build and train a multilayer perceptron.
   - Compare a baseline model with a regularized model.
   - Evaluate MAE, RMSE, and $$R^2$$.
   - Plot predictions against actual targets and inspect residuals.

2. **Binary Classification Project** (`02_neural_network_binary_classification.py`)
   - Create a binary classification dataset.
   - Build a classifier using `BCEWithLogitsLoss`.
   - Evaluate precision, recall, F1-score, and ROC-AUC.
   - Inspect a confusion matrix and experiment with classification thresholds.
   - Investigate how class imbalance affects evaluation.

3. **Multiclass Classification Project** (`03_neural_network_multiclass_classification.py`)
   - Build a classifier for multiple mutually exclusive classes.
   - Train the model using `CrossEntropyLoss`.
   - Evaluate accuracy, macro F1-score, and per-class performance.
   - Inspect a confusion matrix to identify frequently confused classes.
   - Save the trained model and implement reproducible inference.

## Expected Outcome

After completing this module, you should be able to build complete neural network solutions for regression, binary classification, and multiclass classification. You should also be able to justify model and metric choices, diagnose training behavior, prevent evaluation leakage, and deliver a trained PyTorch model with a reliable inference workflow.
