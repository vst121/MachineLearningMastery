# 🌳 Decision Trees

Decision Trees are supervised learning algorithms that make predictions by learning a sequence of decision rules from data. They can be used for both classification and regression.

Unlike models that rely on a fixed mathematical relationship between input features and the target, Decision Trees recursively divide the feature space into regions based on feature values.

## 🎯 Learning Objectives

By completing this module, you will learn how to:

- Understand the structure of a Decision Tree, including root nodes, internal nodes, branches, and leaf nodes.
- Train Decision Tree classifiers and regressors using scikit-learn.
- Understand how trees select features and split points.
- Evaluate classification and regression performance.
- Control model complexity using tree depth and minimum sample constraints.
- Understand overfitting and the role of pruning.
- Visualize and interpret learned decision rules.
- Tune Decision Tree hyperparameters using cross-validation.

## 📚 Topics

1. **Decision Tree Fundamentals**: Tree structure, decision rules, and predictions.
2. **Classification Trees**: Gini impurity, entropy, and information gain.
3. **Regression Trees**: Variance reduction and mean-based predictions.
4. **Tree Complexity**: Maximum depth and minimum samples per split or leaf.
5. **Overfitting and Pruning**: Controlling complexity and improving generalization.
6. **Feature Importance**: Understanding how features contribute to splits.
7. **Hyperparameter Tuning**: Selecting configurations with cross-validation.
8. **Model Evaluation**: Assessing performance on unseen data.

## 📁 Module Structure

| File                                        | Focus                                             |
| ------------------------------------------- | ------------------------------------------------- |
| `01_decision_tree_classifier.py`            | Train and evaluate a classification tree          |
| `02_gini_entropy_and_splits.py`             | Explore Gini impurity, entropy, and split quality |
| `03_decision_tree_visualization.py`         | Visualize tree structure and decision boundaries  |
| `04_decision_tree_regression.py`            | Train and evaluate a regression tree              |
| `05_tree_complexity_and_overfitting.py`     | Investigate depth, leaf size, and overfitting     |
| `06_pruning_decision_trees.py`              | Explore cost-complexity pruning                   |
| `07_feature_importance.py`                  | Analyze feature importance                        |
| `08_decision_tree_hyperparameter_tuning.py` | Tune tree parameters with cross-validation        |

## 🧰 Tools and Libraries

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn

## 💡 Key Concepts

- Recursive partitioning
- Gini impurity
- Entropy and information gain
- Variance reduction
- Maximum tree depth
- Minimum samples per split and leaf
- Cost-complexity pruning
- Feature importance
- Bias-variance trade-off
- Cross-validation

## 📝 Practical Exercises

1. Train a Decision Tree classifier on the Iris dataset.
2. Compare Gini impurity and entropy as split criteria.
3. Visualize how tree depth changes decision boundaries.
4. Train a regression tree and evaluate its predictions.
5. Compare shallow and deep trees to investigate overfitting.
6. Apply pruning and measure its effect on generalization.
7. Examine feature importance and its limitations.
8. Select hyperparameters using cross-validation and evaluate the final model on a held-out test set.
