# Logistic Regression 📈

Logistic Regression is a supervised machine learning algorithm primarily used for classification. Despite its name, it predicts class probabilities rather than continuous numerical values.

It estimates the probability that an observation belongs to a particular class and uses a decision threshold to assign a predicted label.

Logistic Regression is an important baseline for classification problems because it is computationally efficient, interpretable, and provides a foundation for understanding more advanced classification algorithms.

## 1. 🎯 Learning Objectives

By completing this topic, you will be able to:

- Explain the difference between regression and classification.
- Understand the sigmoid function and its role in probability estimation.
- Derive the Logistic Regression prediction equation.
- Understand log loss and maximum likelihood estimation.
- Explain how model parameters are learned.
- Interpret coefficients, odds, and odds ratios.
- Understand decision thresholds and classification boundaries.
- Implement binary and multiclass classification using scikit-learn.
- Evaluate classification models using appropriate metrics.
- Understand regularization, its benefits, and its limitations.

## 2. 🧠 Fundamentals

### 2.1. What Problem Does Logistic Regression Solve?

Logistic Regression estimates the probability that an input belongs to a class.

For binary classification, the target variable has two possible values:

$$
y \in \{0,1\}
$$

Examples include:

- Email classification: spam or not spam.
- Fraud detection: fraudulent or legitimate transaction.
- Customer churn: churn or remain.
- Medical screening: positive or negative outcome.

Unlike Linear Regression, Logistic Regression constrains its predicted probabilities to values between 0 and 1.

### 2.2. From Linear Regression to Logistic Regression

A linear model produces a score:

$$
z = \boldsymbol{\theta}^{T}\mathbf{x} + b
$$

where:

- x is the input feature vector.
- theta is the coefficient vector.
- b is the intercept.
- z is the raw model score, also called the logit in Logistic Regression.

The raw score can take any real value. To convert it into a probability, Logistic Regression applies the sigmoid function.

### 2.3. The Sigmoid Function

The sigmoid function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

It maps any real-valued input to a value strictly between 0 and 1.

The predicted probability of the positive class is:

$$
\hat{p} = P(y=1 \mid \mathbf{x}) = \sigma(z)
$$

Therefore, the complete prediction equation is:

$$
\hat{p} =
\frac{1}{1 + e^{-(\boldsymbol{\theta}^{T}\mathbf{x}+b)}}
$$

Interpretation:

- A large positive z produces a probability close to 1.
- A large negative z produces a probability close to 0.
- When z = 0, the predicted probability is 0.5.

**Key note:** A predicted probability is not the same as a predicted class label.

## 3. 📐 Mathematical Foundations

### 3.1. Probability and Decision Threshold

For binary classification, the predicted class is commonly assigned using a threshold of 0.5:

$$
\hat{y} =
\begin{cases}
1 & \text{if } \hat{p} \geq 0.5 \\
0 & \text{if } \hat{p} < 0.5
\end{cases}
$$

Since the sigmoid function equals 0.5 when its input is zero, a threshold of 0.5 corresponds to the decision boundary:

$$
\boldsymbol{\theta}^{T}\mathbf{x}+b=0
$$

The decision boundary is linear in the original feature space.

The threshold can be adjusted according to the problem. For example, fraud detection may require a lower threshold to identify more potentially fraudulent transactions, at the cost of additional false positives.

### 3.2. Odds and Log-Odds

The odds of the positive class are:

$$
\text{Odds} = \frac{p}{1-p}
$$

Taking the natural logarithm gives the log-odds:

$$
\log\left(\frac{p}{1-p}\right)
= \boldsymbol{\theta}^{T}\mathbf{x}+b
$$

This equation explains the name Logistic Regression: the model assumes that the log-odds are a linear function of the input features.

### 3.3. Interpreting Coefficients

For feature x with coefficient theta, a one-unit increase in that feature multiplies the odds by:

$$
\text{Odds Ratio} = e^{\theta}
$$

This interpretation assumes that the other features remain constant.

- If theta = 0, the odds ratio is 1.
- If theta is positive, the odds increase.
- If theta is negative, the odds decrease.

**Important:** Coefficients describe changes in log-odds, not direct changes in probability. The probability change depends on the current feature values and predicted probability.

Feature scaling also affects coefficient magnitudes and their interpretation.

### 3.4. Log Loss

Logistic Regression commonly learns its parameters by minimizing binary cross-entropy, also known as log loss.

For one observation:

$$
L(y,\hat{p}) =
-\left[y\log(\hat{p})+
(1-y)\log(1-\hat{p})\right]
$$

For m observations:

$$
J(\boldsymbol{\theta},b) =
-\frac{1}{m}\sum_{i=1}^{m}
\left[
y_i\log(\hat{p}_i)
+(1-y_i)\log(1-\hat{p}_i)
\right]
$$

where:

- m is the number of training observations.
- y is the true class label for an observation.
- p-hat is the predicted probability of the positive class.

Log loss strongly penalizes confident incorrect predictions. For example, assigning a probability of 0.99 to the wrong class incurs a much larger penalty than assigning a probability of 0.55 to that class.

Minimizing log loss is equivalent to maximizing the likelihood of the observed labels under the model, or minimizing the negative log-likelihood.

### 3.5. Learning the Parameters

The parameters are learned from training data by optimizing the objective function.

For unregularized binary Logistic Regression, the gradients of the average log loss are:

$$
\frac{\partial J}{\partial \boldsymbol{\theta}} = \frac{1}{m}\mathbf{X}^{T}(\hat{\mathbf{p}}-\mathbf{y})
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(\hat{p}_i-y_i)
$$

A gradient-based optimizer can update the parameters as follows:

$$
\boldsymbol{\theta}
\leftarrow
\boldsymbol{\theta}
-\alpha\frac{\partial J}{\partial\boldsymbol{\theta}}
$$

$$
b \leftarrow b-\alpha\frac{\partial J}{\partial b}
$$

where alpha is the learning rate.

In practice, scikit-learn's LogisticRegression uses a solver to optimize the objective. It does not use the simple gradient-descent procedure shown above as its default implementation.

## 4. ⚙️ Model Parameters and Hyperparameters

| Parameter     | Description                                                                         |
| ------------- | ----------------------------------------------------------------------------------- |
| C             | Inverse regularization strength. Smaller values apply stronger regularization.      |
| penalty       | Specifies the regularization type supported by the chosen solver, such as L1 or L2. |
| solver        | Optimization algorithm used to fit the model.                                       |
| max_iter      | Maximum number of solver iterations.                                                |
| class_weight  | Adjusts the relative weight of classes during training.                             |
| fit_intercept | Determines whether the model includes an intercept.                                 |
| random_state  | Controls randomness where supported by the solver and configuration.                |

**Important:** Not every penalty is supported by every solver. Check the scikit-learn documentation for supported combinations.

### 4.1. Regularization

Regularization discourages overly large coefficients and can improve generalization.

A regularized objective can be written as:

$$
J_{\text{regularized}}
=
J_{\text{data}}+\lambda R(\boldsymbol{\theta})
$$

where R is the penalty function and lambda controls its strength.

In scikit-learn, C is inversely related to regularization strength. A smaller C means stronger regularization; a larger C means weaker regularization.

Common penalties include:

- **L2:** discourages large coefficients.
- **L1:** encourages sparsity and can drive coefficients to zero.
- **Elastic Net:** combines L1 and L2 penalties when supported by the selected solver.

## 5. 💻 Implementation and Experiments

The following scripts are a proposed learning sequence. Create them incrementally as you work through the topic.

| File                                   | Experiment                                                                          |
| -------------------------------------- | ----------------------------------------------------------------------------------- |
| `01_logistic_regression.py`            | Train a binary classifier and inspect predicted probabilities.                      |
| `02_decision_thresholds.py`            | Change the classification threshold and examine the resulting predictions.          |
| `03_multiclass_classification.py`      | Explore multiclass classification and probability estimates.                        |
| `04_regularization_experiment.py`      | Compare regularization strengths and their effects on coefficients and performance. |
| `05_logistic_regression_evaluation.py` | Evaluate predictions using classification metrics and visualizations.               |

Use `# %%` cell markers to divide scripts into interactive sections in VS Code.

Recommended progression:

1. Start with a simple dataset, such as Iris using two classes.
2. Split the data into training and test sets.
3. Train Logistic Regression using scikit-learn.
4. Inspect predicted probabilities using the `predict_proba()` method.
5. Compare probabilities with predicted labels from the `predict()` method.
6. Visualize the decision boundary when using two features.
7. Experiment with thresholds and regularization.
8. Evaluate the model on held-out data.

## 6. 📊 Model Evaluation

The choice of evaluation metric depends on the classification problem.

| Metric    | What It Measures                                                 |
| --------- | ---------------------------------------------------------------- |
| Accuracy  | Proportion of predictions that are correct.                      |
| Precision | Proportion of predicted positives that are truly positive.       |
| Recall    | Proportion of actual positives correctly identified.             |
| F1-score  | Harmonic mean of precision and recall.                           |
| ROC-AUC   | How well scores rank positive cases above negative cases.        |
| Log loss  | Quality of predicted probabilities, penalizing confident errors. |

Accuracy alone can be misleading when classes are imbalanced. For example, a model that predicts the majority class for every observation can achieve high accuracy while failing to identify the minority class.

Use a confusion matrix and select metrics that reflect the cost of false positives and false negatives.

## 7. ⚠️ Limitations and Common Pitfalls

- **Linear decision boundary:** standard Logistic Regression is linear in the supplied features. Nonlinear relationships may require feature transformations or another model.
- **Feature scaling:** scaling is especially useful when features have different magnitudes and when regularization is used.
- **Multicollinearity:** highly correlated features can make individual coefficients unstable.
- **Class imbalance:** accuracy may hide poor minority-class performance.
- **Threshold assumptions:** 0.5 is a common default, not an optimal threshold for every problem.
- **Probability calibration:** good classification accuracy does not automatically imply well-calibrated probabilities.
- **Data leakage:** preprocessing and feature selection must be fitted using training data only.
- **Overfitting:** excessive model complexity or weak regularization can reduce performance on unseen data.

## 8. 📝 Exercises

1. Train a Logistic Regression model on a binary classification dataset and explain its prediction pipeline.
2. Plot the sigmoid function and explain how the logit affects predicted probability.
3. Compare predicted probabilities with predicted class labels.
4. Change the decision threshold and observe its effects on precision and recall.
5. Interpret model coefficients and calculate odds ratios.
6. Compare models with different values of C.
7. Evaluate a model using accuracy, precision, recall, F1-score, ROC-AUC, and log loss.
8. Investigate the effects of feature scaling.
9. Test the model on an imbalanced dataset and explain why accuracy may be insufficient.
10. Compare Logistic Regression with a simple baseline classifier.

## 9. 🔑 Key Takeaways

- Logistic Regression is a classification algorithm that estimates class probabilities.
- The sigmoid function maps the linear score to a probability.
- The model learns coefficients by optimizing an objective commonly based on log loss.
- Coefficients describe effects on log-odds, and exponentiated coefficients represent odds ratios.
- A decision threshold converts probabilities into class labels.
- Regularization can improve generalization by controlling coefficient magnitudes.
- Model evaluation must reflect the problem's requirements and the costs of classification errors.

## 10. 📚 References

- [scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [scikit-learn: LogisticRegression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn: Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [scikit-learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
