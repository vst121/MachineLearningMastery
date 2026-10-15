# 🌲 Ensemble Learning

Ensemble Learning combines predictions from multiple models to improve predictive performance, stability, or generalization. Common approaches include voting, Bagging, Boosting, and Stacking.

This module focuses on the ensemble methods most relevant to practical machine learning and Senior AI Engineer interviews.

## 🎯 Learning Objectives

By completing this module, you will learn how to:

- Understand why ensembles can outperform individual models.
- Apply voting, averaging, and Bagging.
- Train Random Forest classifiers and regressors.
- Understand AdaBoost and Gradient Boosting.
- Explore efficient histogram-based boosting.
- Combine different models using Stacking.
- Prevent data leakage during model training and evaluation.
- Tune hyperparameters using cross-validation.
- Compare models using appropriate evaluation metrics.

## 📁 Module Structure

| File                                   | Focus                                                                            |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| `01_voting_and_bagging.py`             | Voting, averaging, bootstrap sampling, and Bagging                               |
| `02_random_forest.py`                  | Random Forest classification and regression, Extra Trees, and feature importance |
| `03_boosting.py`                       | AdaBoost, Gradient Boosting, and sequential learning                             |
| `04_histogram_gradient_boosting.py`    | Efficient histogram-based boosting and early stopping                            |
| `05_stacking.py`                       | Combining different models with a meta-model                                     |
| `06_ensemble_evaluation_and_tuning.py` | Cross-validation, hyperparameter tuning, and model comparison                    |

## 💡 Key Concepts

- Ensemble diversity
- Hard and soft voting
- Bootstrap sampling
- Bagging and variance reduction
- Random feature selection
- Random Forests and Extra Trees
- Adaptive Boosting
- Gradient Boosting and learning rates
- Histogram-based splitting
- Stacking and out-of-fold predictions
- Bias-variance trade-off
- Feature importance
- Cross-validation and data leakage

## 🧰 Tools and Libraries

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn

## 📝 Practical Exercises

1. Compare individual classifiers with voting and Bagging.
2. Investigate how Random Forests differ from individual Decision Trees.
3. Compare AdaBoost with Gradient Boosting.
4. Evaluate histogram-based boosting for tabular data.
5. Build a Stacking ensemble and understand out-of-fold predictions.
6. Tune ensemble hyperparameters using cross-validation.
7. Compare performance, model complexity, and generalization using a held-out test set.

## 📌 Evaluation Principles

- Split data before fitting models or preprocessing transformations.
- Use cross-validation on training data for model selection.
- Keep the test set separate from hyperparameter tuning.
- Choose evaluation metrics based on the problem's requirements.
- Interpret feature importance as a model diagnostic, not proof of causation.
