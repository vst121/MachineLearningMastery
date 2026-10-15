# K-Nearest Neighbors (KNN)

## 1. 🎯 Overview

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression.

KNN makes predictions by identifying the nearest training examples to a new observation. For classification, it predicts a class based on the labels of its neighbors. For regression, it estimates a continuous value from neighboring target values.

Unlike parametric models, KNN does not learn an explicit set of model coefficients during training. Instead, it retains the training data and uses distance calculations to make predictions.

## 2. 🧠 Core Concepts

- **Nearest Neighbors:** Identifying the most similar observations using a distance metric.
- **Distance Metrics:** Measuring similarity using Euclidean or Manhattan distance.
- **Number of Neighbors (k):** Controlling how many observations contribute to a prediction.
- **Uniform and Distance-Weighted Voting:** Giving neighbors equal influence or weighting closer neighbors more heavily.
- **Feature Scaling:** Ensuring that features with larger numerical ranges do not disproportionately influence distances.
- **Hyperparameter Tuning:** Selecting suitable values for k, voting strategy, and distance metric.
- **Decision Boundaries:** Understanding how KNN separates classes in a feature space.
- **KNN Regression:** Predicting continuous values using neighboring observations.
- **Model Evaluation:** Assessing generalization using appropriate metrics and cross-validation.

## 3. 📐 Mathematical Foundation

### 3.1. Euclidean Distance

The straight-line distance between two observations x and y is:

$$
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

### 3.2. Manhattan Distance

The sum of the absolute differences between two observations is:

$$
d(x,y)=\sum_{i=1}^{n}|x_i-y_i|
$$

### 3.3. Classification by Majority Voting

For a classification problem, KNN predicts the class that occurs most frequently among the k nearest neighbors.

$$
\hat{y}=\text{mode}(y_1,y_2,\ldots,y_k)
$$

### 3.4. Distance-Weighted Voting

Closer neighbors can receive greater influence than more distant neighbors. A common weighting function is:

$$
w_i=\frac{1}{d_i+\epsilon}
$$

where dᵢ is the distance to neighbor i and ε is a small positive value that prevents division by zero.

### 3.5. Feature Standardization

Standardization transforms each feature using its mean and standard deviation:

$$
z=\frac{x-\mu}{\sigma}
$$

This is important because KNN relies on distances between observations.

## 4. 💻 Implementation and Experiments

The following scripts are planned for this module:

| File                              | Focus                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------- |
| `01_knn_classifier.py`            | Basic KNN classification, predictions, class probabilities, and the effect of k |
| `02_knn_hyperparameter_tuning.py` | Hyperparameter selection using grid search and stratified cross-validation      |
| `03_knn_distance_metrics.py`      | Comparing Euclidean and Manhattan distance                                      |
| `04_knn_distance_weighting.py`    | Comparing uniform voting with distance-weighted voting                          |
| `05_knn_feature_scaling.py`       | Investigating the impact of feature scaling on KNN performance                  |
| `06_knn_decision_boundaries.py`   | Visualizing KNN decision boundaries in a two-dimensional feature space          |
| `07_knn_regression.py`            | KNN regression, continuous predictions, and regression metrics                  |

## 5. 🧪 Datasets and Evaluation

The module uses datasets suitable for exploring KNN classification and regression.

- **Iris:** A three-class classification dataset for understanding KNN fundamentals and decision boundaries.
- **Additional datasets where appropriate:** Used to investigate feature scaling, hyperparameter sensitivity, and regression behavior.

Classification experiments use metrics such as accuracy, precision, recall, F1-score, and confusion matrices. Regression experiments use metrics such as MAE, MSE, RMSE, and R².

Stratified cross-validation is used for classification hyperparameter selection where appropriate. The test set is reserved for final evaluation to reduce the risk of biased performance estimates.

## 6. 🔑 Key Takeaways

- KNN is an instance-based, distance-driven learning algorithm.
- The choice of k influences the balance between sensitivity to local patterns and generalization.
- Distance metrics and feature scaling can significantly affect predictions.
- Distance-weighted voting can give closer observations more influence.
- Cross-validation provides a systematic approach to hyperparameter selection.
- KNN supports both classification and regression.
- Prediction can become computationally expensive as the training dataset grows.
