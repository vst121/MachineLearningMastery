# Machine Learning Advanced 🚀

This module builds on the concepts introduced in [Machine Learning Foundations](../01-ML-Foundations/README.md) and explores more advanced algorithms, model evaluation techniques, and practical machine learning workflows.

The focus is on understanding how models work, why they behave differently, how to evaluate them reliably, and how to select and optimize them for real-world problems.

## 1. 🎯 Learning Objectives

By completing this module, you will be able to:

- Understand the mathematical foundations of advanced machine learning algorithms.
- Implement classification and regression models using Python and scikit-learn.
- Evaluate model performance using appropriate metrics and validation strategies.
- Understand overfitting, underfitting, regularization, and the bias-variance trade-off.
- Apply ensemble learning techniques to improve predictive performance.
- Reduce dimensionality and analyze high-dimensional datasets.
- Perform feature engineering and build preprocessing pipelines.
- Optimize model hyperparameters without introducing data leakage.
- Interpret model behavior and explain model selection decisions.
- Design reproducible machine learning experiments.

## 2. 🗺️ Learning Roadmap

| # | Topic | Key Concepts |
|---|---|---|
| 1 | Logistic Regression | Sigmoid function, probabilities, log loss, decision thresholds, multiclass classification |
| 2 | Model Evaluation | Cross-validation, confusion matrix, precision, recall, F1-score, ROC-AUC |
| 3 | K-Nearest Neighbors (KNN) | Distance metrics, selecting K, feature scaling, curse of dimensionality |
| 4 | Regularization | Ridge, Lasso, Elastic Net, coefficient shrinkage, bias-variance trade-off |
| 5 | Decision Trees | Gini impurity, entropy, information gain, pruning, feature importance |
| 6 | Ensemble Learning | Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM |
| 7 | Dimensionality Reduction | Principal Component Analysis (PCA), explained variance, component selection |
| 8 | Feature Engineering | Missing values, categorical encoding, scaling, transformations, feature selection |
| 9 | Hyperparameter Tuning | GridSearchCV, RandomizedSearchCV, parameter selection, cross-validation |

## 3. 📂 Repository Structure

```text
02-ML-Advanced/
├── 01-logistic-regression/
├── 02-model-evaluation/
├── 03-knn/
├── 04-regularization/
├── 05-decision-trees/
├── 06-ensemble-learning/
├── 07-dimensionality-reduction/
├── 08-feature-engineering/
├── 09-hyperparameter-tuning/
└── README.md
```

Each topic folder contains its own documentation, mathematical foundations, practical examples, experiments, and exercises.

## 4. 📚 Topics and Key Concepts

### 4.1. Logistic Regression

- Logistic function and probability estimation
- Log loss and maximum likelihood
- Decision thresholds and classification boundaries
- Binary and multiclass classification
- Coefficient interpretation and regularization

### 4.2. Model Evaluation

- Training, validation, and test sets
- Cross-validation and stratified sampling
- Confusion matrix and classification metrics
- ROC curves, ROC-AUC, and precision-recall curves
- Overfitting, underfitting, and data leakage

### 4.3. K-Nearest Neighbors (KNN)

- Euclidean and Manhattan distances
- Choosing the number of neighbors
- Uniform and distance-weighted voting
- Feature scaling and decision boundaries
- Computational complexity and the curse of dimensionality

### 4.4. Regularization

- Ridge Regression (L2)
- Lasso Regression (L1)
- Elastic Net
- Penalty terms and coefficient shrinkage
- Feature selection and the bias-variance trade-off

### 4.5. Decision Trees

- Gini impurity and entropy
- Information gain and split selection
- Tree depth and stopping criteria
- Overfitting and pruning
- Feature importance and interpretability

### 4.6. Ensemble Learning

- Bagging and Random Forest
- Boosting and AdaBoost
- Gradient Boosting
- XGBoost and LightGBM
- Bias and variance in ensemble models

### 4.7. Dimensionality Reduction

- Principal Component Analysis (PCA)
- Covariance and principal components
- Explained variance and component selection
- Feature scaling before dimensionality reduction
- Visualization and information loss

### 4.8. Feature Engineering

- Missing-value imputation
- Numerical and categorical feature transformations
- One-hot and ordinal encoding
- Feature selection and transformation
- Pipeline and ColumnTransformer

### 4.9. Hyperparameter Tuning

- Hyperparameters versus model parameters
- GridSearchCV and RandomizedSearchCV
- Search spaces and scoring strategies
- Cross-validation during model selection
- Reproducibility and avoiding test-set leakage

## 5. 🧪 Practical Learning Workflow

Each topic follows a consistent learning process:

1. **Understand:** study the intuition, assumptions, and mathematical foundations.
2. **Implement:** build a small example using Python and scikit-learn.
3. **Visualize:** investigate decision boundaries, model behavior, or parameter effects where applicable.
4. **Experiment:** change important parameters and compare outcomes.
5. **Evaluate:** measure performance using suitable metrics and validation techniques.
6. **Analyze:** investigate errors, limitations, computational costs, and trade-offs.
7. **Practice:** complete exercises and apply the concepts to realistic datasets.

Python scripts use `# %%` cell markers to support interactive execution in VS Code.

## 6. 🛠️ Technology Stack

- **Python:** implementation and experimentation
- **NumPy:** numerical computation
- **pandas:** data manipulation and analysis
- **Matplotlib:** visualization
- **scikit-learn:** algorithms, preprocessing, evaluation, and model selection
- **XGBoost and LightGBM:** gradient-boosting experiments

## 7. 💡 Key Principles

- **Generalization:** optimize performance on unseen data, not just the training set.
- **Reliable evaluation:** select appropriate metrics and validation strategies before tuning.
- **Data leakage prevention:** fit preprocessing steps only on training data.
- **Baseline first:** establish a reference model before introducing complexity.
- **Reproducibility:** document experimental settings and control randomness where appropriate.
- **Interpretability:** understand what the model learns and where it can fail.
- **Trade-offs:** consider accuracy, complexity, inference cost, and maintainability.

## 8. 📝 Exercises and Self-Assessment

For each topic, practice answering the following questions:

1. What problem does the algorithm solve?
2. What are its mathematical foundations and assumptions?
3. Which parameters have the greatest influence on its behavior?
4. How can overfitting or underfitting be detected?
5. Which evaluation metrics are appropriate, and why?
6. What are the algorithm's strengths, limitations, and computational costs?
7. When would you choose it over an alternative?
8. How would you explain its behavior and results in a technical interview?
