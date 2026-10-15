# Regularization

## 1. 🎯 Overview

Regularization is a technique used to reduce overfitting by penalizing model complexity. It helps models generalize better to unseen data by discouraging excessively large parameter values.

Regularization is especially useful when a model learns patterns that fit the training data well but do not generalize reliably to new observations.

Two widely used approaches are **L1 regularization (Lasso)** and **L2 regularization (Ridge)**. They modify the model's optimization objective by adding a penalty based on the model coefficients.

## 2. 🧠 Core Concepts

- **Overfitting:** When a model fits training data too closely and performs poorly on unseen data.
- **Regularization Penalty:** An additional term in the optimization objective that discourages model complexity.
- **L1 Regularization (Lasso):** Penalizes the absolute values of coefficients and can drive some coefficients to zero.
- **L2 Regularization (Ridge):** Penalizes squared coefficient values and generally shrinks coefficients toward zero.
- **Regularization Strength:** Controls how strongly the penalty influences model training.
- **Coefficient Shrinkage:** Reducing coefficient magnitudes to limit model sensitivity.
- **Feature Selection:** L1 regularization can produce sparse models by setting some coefficients to zero.
- **Feature Scaling:** Important because regularization penalizes coefficient magnitudes.
- **Bias-Variance Trade-off:** Stronger regularization can reduce variance but may increase bias.
- **Cross-Validation:** Helps select an appropriate regularization strength without using the test set for tuning.

## 3. 📐 Mathematical Foundation

### 3.1. Ordinary Least Squares

Linear regression minimizes the sum of squared prediction errors:

$$
J(\theta)=\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

### 3.2. L2 Regularization (Ridge)

Ridge regression adds a penalty based on the squared coefficient values:

$$
J(\theta)=\sum_{i=1}^{m}(y_i-\hat{y}_i)^2+\alpha\sum_{j=1}^{n}\theta_j^2
$$

The penalty discourages large coefficients. In standard implementations, the intercept is typically not penalized.

### 3.3. L1 Regularization (Lasso)

Lasso regression adds a penalty based on the absolute coefficient values:

$$
J(\theta)=\sum_{i=1}^{m}(y_i-\hat{y}_i)^2+\alpha\sum_{j=1}^{n}|\theta_j|
$$

L1 regularization can set some coefficients exactly to zero, effectively removing the corresponding features from the model.

### 3.4. Regularization Strength

The parameter alpha controls the strength of the penalty.

- **Small alpha:** Weaker regularization, allowing larger coefficient values.
- **Large alpha:** Stronger regularization, shrinking coefficients more aggressively.
- **Alpha equal to zero:** No regularization penalty.

The appropriate value depends on the dataset and should be selected using validation data or cross-validation.

### 3.5. Logistic Regression Regularization

Logistic regression can also use L1 or L2 regularization. In scikit-learn, its parameter C represents the inverse of regularization strength:

$$
C\propto\frac{1}{\text{regularization strength}}
$$

A smaller C means stronger regularization, while a larger C means weaker regularization.

## 4. 💻 Implementation and Experiments

The following scripts are planned for this module:

| File                                         | Focus                                                                 |
| -------------------------------------------- | --------------------------------------------------------------------- |
| `01_ridge_regression.py`                     | Ridge regression, coefficient shrinkage, and regression metrics       |
| `02_lasso_regression.py`                     | Lasso regression, coefficient shrinkage, and zero-valued coefficients |
| `03_ridge_vs_lasso.py`                       | Comparing L1 and L2 regularization                                    |
| `04_logistic_regression_regularization.py`   | L1 and L2 regularization in logistic regression                       |
| `05_regularization_strength.py`              | Investigating how alpha and C affect coefficients and performance     |
| `06_regularization_with_cross_validation.py` | Selecting regularization strength using cross-validation              |

## 5. 🧪 Datasets and Evaluation

The module uses regression and classification datasets to examine regularization under different conditions.

- **Regression:** Used to compare Ridge, Lasso, and ordinary linear regression.
- **Classification:** Used to investigate regularization in logistic regression.

Regression experiments use MAE, MSE, RMSE, and R². Classification experiments use accuracy, precision, recall, F1-score, and appropriate probability-based metrics.

Experiments examine coefficient magnitudes, model performance, and the effect of regularization strength. Cross-validation is used for model selection, while a separate test set is reserved for final evaluation.

Feature scaling is applied where appropriate, particularly when coefficient penalties are used.

## 6. 🔑 Key Takeaways

- Regularization helps control overfitting by penalizing model coefficients.
- Ridge uses an L2 penalty and generally shrinks coefficients without eliminating features.
- Lasso uses an L1 penalty and can set some coefficients to zero.
- Stronger regularization can simplify a model but may lead to underfitting.
- Feature scaling matters when coefficient magnitudes determine the penalty.
- Alpha controls regularization strength in Ridge and Lasso, while C controls inverse regularization strength in scikit-learn's logistic regression.
- Cross-validation provides a systematic way to select regularization parameters.
