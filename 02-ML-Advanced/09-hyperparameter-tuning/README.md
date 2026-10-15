# Hyperparameter Tuning

## 1. Overview

Hyperparameter tuning is the process of finding suitable configurations for a machine learning algorithm to improve its predictive performance and generalization.

Unlike model parameters, which are learned during training, hyperparameters are configured before or during the training process. Examples include the number of neighbors in KNN, the regularization strength in logistic regression, and the maximum depth of a decision tree.

This module explores systematic search strategies, pipeline optimization, and rigorous model evaluation using scikit-learn.

## 2. Learning Objectives

By completing this module, you will be able to:

- Distinguish model parameters from hyperparameters.
- Implement exhaustive search with `GridSearchCV`.
- Use `RandomizedSearchCV` to explore large search spaces efficiently.
- Design appropriate hyperparameter ranges and probability distributions.
- Tune preprocessing and model hyperparameters together using `Pipeline`.
- Prevent data leakage during hyperparameter optimization.
- Explain the difference between cross-validation for tuning and nested cross-validation for performance estimation.
- Evaluate tuned models using appropriate metrics and an independent test set.

## 3. Topics and Practical Exercises

| File                                   | Topic                   | Key Concepts                                                                         |
| -------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------ |
| `01_grid_search.py`                    | Grid Search             | `GridSearchCV`, parameter grids, cross-validation, scoring, best parameters          |
| `02_randomized_search.py`              | Randomized Search       | `RandomizedSearchCV`, parameter distributions, search budgets, efficient exploration |
| `03_pipeline_hyperparameter_tuning.py` | Pipeline Tuning         | `Pipeline`, `ColumnTransformer`, preprocessing, model parameters, leakage prevention |
| `04_nested_cross_validation.py`        | Nested Cross-Validation | Inner and outer folds, model selection bias, performance estimation, generalization  |

## 4. Key Concepts

### 4.1. Parameters vs. Hyperparameters

- **Model parameters** are learned from training data, such as regression coefficients and decision-tree split thresholds.
- **Hyperparameters** control the learning process or model configuration, such as regularization strength, tree depth, and the number of neighbors.

### 4.2. Grid Search

Grid search evaluates every combination of explicitly specified hyperparameter values. It is systematic and easy to interpret, but computational cost grows rapidly as the number of parameters and candidate values increases.

### 4.3. Randomized Search

Randomized search samples a fixed number of candidate configurations from specified lists or probability distributions. It can explore broad search spaces with a controlled computational budget.

### 4.4. Cross-Validation and Scoring

Cross-validation estimates model performance across multiple training and validation splits. The scoring metric should match the problem, such as accuracy, F1-score, ROC AUC, or negative root mean squared error.

The best cross-validation score is useful for model selection, but it can be optimistic when many configurations are evaluated.

### 4.5. Pipeline-Based Tuning

A scikit-learn `Pipeline` combines preprocessing and model training into one estimator. During cross-validation, each preprocessing step is fitted only on the training portion of the corresponding fold.

This prevents leakage from validation data and enables preprocessing and model hyperparameters to be optimized together.

### 4.6. Nested Cross-Validation

Nested cross-validation separates hyperparameter selection from performance estimation:

- **Inner cross-validation:** Selects the best hyperparameters.
- **Outer cross-validation:** Estimates the performance of the complete model-selection process on held-out folds.

Nested cross-validation is particularly useful when a reliable performance estimate is required and the available dataset is limited.

## 5. Best Practices

- Define a meaningful baseline before tuning.
- Choose hyperparameter ranges based on model behavior and domain knowledge.
- Use logarithmic distributions for parameters that span several orders of magnitude, such as regularization strength.
- Prefer randomized search when the search space is large.
- Select scoring metrics that reflect the actual objective.
- Keep preprocessing and feature selection inside the cross-validation pipeline.
- Never use the final test set to choose hyperparameters.
- Consider computational cost when defining the search budget.
- Inspect cross-validation results and variability, not only the best score.
- Use nested cross-validation when a more rigorous estimate of the complete tuning process is needed.
- Refit the selected estimator on the available training data before final evaluation.

## 6. References

- [Scikit-learn: Tuning the Hyper-Parameters of an Estimator](https://scikit-learn.org/stable/modules/grid_search.html)
- [Scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Scikit-learn: RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
- [Scikit-learn: Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Scikit-learn: Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
