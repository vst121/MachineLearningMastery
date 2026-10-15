# Model Evaluation 📊

Model evaluation is the process of measuring how well a machine learning model performs on unseen data. It helps us compare models, detect overfitting, select appropriate metrics, and estimate how well a trained model may generalize to real-world data.

A model that performs well on training data is not necessarily a good model. Reliable evaluation requires appropriate data splitting, suitable metrics, and careful prevention of data leakage.

## 1. 🎯 Learning Objectives

By completing this module, you will be able to:

- Understand training, validation, and test datasets.
- Distinguish overfitting, underfitting, and generalization.
- Apply holdout validation and cross-validation.
- Interpret confusion matrices and classification metrics.
- Evaluate binary classifiers using ROC-AUC and precision-recall curves.
- Understand regression evaluation metrics.
- Detect data leakage and common evaluation mistakes.
- Compare models using consistent and reproducible experiments.
- Select evaluation metrics based on business objectives and error costs.

## 2. 🧠 Core Concepts

### 2.1 Dataset Splitting

Machine learning datasets are commonly divided into three subsets:

| Dataset        | Purpose                                                  |
| -------------- | -------------------------------------------------------- |
| Training set   | Fits model parameters                                    |
| Validation set | Selects models, hyperparameters, and decision thresholds |
| Test set       | Estimates final performance on unseen data               |

The test set should remain separate from decisions made during model development.

For smaller datasets, cross-validation can help use the available training data more efficiently.

### 2.2 Generalization

Generalization describes how well a model performs on data it has not encountered during training.

- **Underfitting:** The model is too simple or insufficiently trained to capture important patterns.
- **Good generalization:** The model captures useful patterns and performs consistently on unseen data.
- **Overfitting:** The model learns training-specific patterns, including noise, that do not generalize.

### 2.3 Cross-Validation

In k-fold cross-validation, the training data is divided into k folds. Each fold is used once for validation while the remaining folds are used for training.

The final performance estimate is typically calculated by averaging the scores across folds.

Cross-validation helps estimate model performance more reliably than a single split, although the estimate still depends on dataset size, sampling, and the evaluation procedure.

### 2.4 Classification Metrics

Important classification metrics include:

- Accuracy
- Precision
- Recall (sensitivity)
- Specificity
- F1-score
- ROC-AUC
- Average precision
- Log loss

The appropriate metric depends on the problem. Accuracy may be misleading for imbalanced datasets, while precision and recall help describe different types of classification errors.

### 2.5 Regression Metrics

Regression models predict continuous numerical values.

Common evaluation metrics include:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² score

MAE measures average absolute error. MSE penalizes larger errors more heavily. RMSE expresses error in the target variable's original units. R² compares the model against a baseline that predicts the mean of the target values.

### 2.6 Data Leakage

Data leakage occurs when information unavailable at prediction time influences model training or evaluation.

Common examples include:

- Fitting a scaler before splitting the data.
- Using test data to select hyperparameters.
- Including features derived from future information.
- Allowing related samples to appear in both training and test sets when they should be grouped.

Use pipelines and appropriate splitting strategies to reduce these risks.

## 3. 📐 Mathematical Foundations

### Accuracy

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

### Precision

$$
Precision = \frac{TP}{TP + FP}
$$

### Recall

$$
Recall = \frac{TP}{TP + FN}
$$

### F1-score

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

### Mean Absolute Error

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

### Root Mean Squared Error

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

### Coefficient of Determination

$$
R^2 = 1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

Here, y represents an actual target value, ŷ represents a prediction, and n is the number of evaluated samples.

## 4. 💻 Implementation and Experiments

The following scripts are planned for this module:

| File                                    | Focus                                                    |
| --------------------------------------- | -------------------------------------------------------- |
| `01_train_validation_test_split.py`     | Dataset splitting, stratification, and reproducibility   |
| `02_cross_validation.py`                | K-fold and stratified cross-validation                   |
| `03_classification_metrics.py`          | Confusion matrix, precision, recall, F1, and specificity |
| `04_roc_auc_and_pr_curves.py`           | ROC-AUC and precision-recall analysis                    |
| `05_regression_metrics.py`              | MAE, MSE, RMSE, and R²                                   |
| `06_overfitting_and_learning_curves.py` | Training versus validation performance                   |
| `07_data_leakage.py`                    | Leakage examples and pipeline-based prevention           |
| `08_model_comparison.py`                | Consistent evaluation and model comparison               |

These filenames define the proposed learning sequence. Add each script as its implementation is completed.

## 5. 🛠️ Technology Stack

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn

Primary scikit-learn components:

- `train_test_split`
- `cross_val_score`
- `StratifiedKFold`
- `KFold`
- `Pipeline`
- `ConfusionMatrixDisplay`
- `roc_auc_score`
- `precision_recall_curve`
- `mean_absolute_error`
- `mean_squared_error`
- `r2_score`
- `learning_curve`

## 6. 💡 Key Principles

1. Evaluate generalization, not just training performance.
2. Keep the final test set independent from model selection.
3. Choose metrics according to the problem and cost of errors.
4. Use stratification when appropriate for classification.
5. Use group-based or time-aware splitting when the data structure requires it.
6. Fit preprocessing transformations only on the training data.
7. Report cross-validation variability, not only the mean score.
8. Compare models on consistent splits and metrics.
9. Preserve reproducibility through controlled random seeds and documented experiments.

## 7. 📝 Exercises and Self-Assessment

- [ ] Explain the difference between training, validation, and test data.
- [ ] Implement stratified train-test splitting.
- [ ] Compare a single holdout score with cross-validation results.
- [ ] Calculate precision, recall, specificity, and F1-score from a confusion matrix.
- [ ] Explain when ROC-AUC and average precision are useful.
- [ ] Evaluate a regression model using MAE, RMSE, and R².
- [ ] Demonstrate how preprocessing before splitting can cause data leakage.
- [ ] Compare multiple models without repeatedly tuning against the test set.
- [ ] Explain why random splitting is inappropriate for some time-series and grouped datasets.
