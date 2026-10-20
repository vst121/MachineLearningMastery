# NLP Evaluation and Experimental Design

This module explores how to evaluate NLP systems rigorously, establish meaningful baselines, design reliable experiments, analyze model errors, and determine whether observed performance differences are statistically meaningful.

## Learning Objectives

- Select appropriate evaluation metrics for NLP tasks.
- Establish baseline performance before training complex models.
- Understand training, validation, and test splits and prevent data leakage.
- Apply cross-validation and compare models fairly.
- Interpret confusion matrices and investigate classification errors.
- Understand uncertainty, confidence intervals, and statistical significance.
- Design reproducible experiments and report results responsibly.

## Core Concepts

### 1. Evaluation Metrics and Baselines

Accuracy alone can be misleading when classes are imbalanced. Precision, recall, and F1-score provide complementary views of classification performance.

A simple baseline, such as majority-class prediction, helps determine whether a trained model provides meaningful improvement.

### 2. Data Splitting and Cross-Validation

Training data is used to fit model parameters, validation data supports model selection, and test data estimates final generalization performance.

Cross-validation evaluates a model across multiple data splits. Preprocessing, feature selection, and other learned transformations must be fitted only on the training portion of each split to avoid data leakage.

### 3. Error Analysis and Confusion Matrices

Aggregate metrics summarize performance but can hide systematic weaknesses. Examining false positives, false negatives, class-specific errors, and difficult examples helps identify opportunities for improvement.

### 4. Statistical Significance and Model Comparison

A difference in measured scores does not automatically establish that one model is better. Experimental design, variability, sample size, paired comparisons, and uncertainty must be considered before drawing conclusions.

### 5. Reproducibility

Reliable experiments document dataset versions, preprocessing, random seeds, model configurations, evaluation metrics, and computational conditions. Reproducibility makes results easier to verify and compare.

## Mathematical Foundations

### 1. Confusion Matrix

For binary classification, predictions can be categorized as true positives, true negatives, false positives, and false negatives.

|                 | Predicted positive | Predicted negative |
| --------------- | -----------------: | -----------------: |
| Actual positive |                 TP |                 FN |
| Actual negative |                 FP |                 TN |

### 2. Accuracy

Accuracy measures the proportion of correct predictions.

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

Accuracy can be misleading when one class dominates the dataset.

### 3. Precision and Recall

Precision measures how many predicted positive examples are actually positive.

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

Recall measures how many actual positive examples are correctly identified.

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

The choice between emphasizing precision or recall depends on the cost of false positives and false negatives.

### 4. F1-Score

The F1-score is the harmonic mean of precision and recall.

$$
F_1=
2\frac{\text{Precision}\cdot\text{Recall}}
{\text{Precision}+\text{Recall}}
$$

It is useful when both false positives and false negatives matter. For multiclass classification, macro-F1 gives equal weight to each class, while weighted-F1 weights each class by its support.

### 5. Majority-Class Baseline

For a dataset with class proportions represented by the class counts, a majority-class classifier always predicts the most frequent class.

$$
\text{Baseline Accuracy} = \frac{\max_c N_c}{N}
$$

Here, \(N_c\) is the number of examples in class \(c\), and \(N\) is the total number of examples.

A trained model should be compared against a relevant baseline rather than evaluated in isolation.

### 6. Cross-Validation

For \(K\) validation folds, the mean score is:

$$
\bar{s}=\frac{1}{K}\sum_{k=1}^{K}s_k
$$

The sample standard deviation across folds is:

$$
s_{\text{std}}=
\sqrt{
\frac{1}{K-1}
\sum_{k=1}^{K}(s_k-\bar{s})^2
}
$$

The mean summarizes observed performance across folds, while the standard deviation describes variation between fold scores. Fold scores are not necessarily independent, so this standard deviation should not automatically be interpreted as a confidence interval.

### 7. Comparing Two Models

For paired evaluation units, define the difference between the scores of two models:

$$
d_i=s_{A,i}-s_{B,i}
$$

The mean paired difference is:

$$
\bar{d}=\frac{1}{n}\sum_{i=1}^{n}d_i
$$

A positive value favors Model A when higher scores are better. The interpretation depends on the metric, evaluation design, and variability of the measurements.

### 8. Confidence Intervals

A confidence interval communicates uncertainty around an estimated quantity. For independent observations, a conventional interval for a sample mean can be expressed as:

$$
\bar{x}\pm t_{1-\alpha/2,n-1}
\frac{s}{\sqrt{n}}
$$

Here, $$\(\bar{x}\)$$ is the sample mean, \(s\) is the sample standard deviation, \(n\) is the sample size, and the critical value comes from the Student's t-distribution.

The assumptions behind an interval matter. For example, individual examples from a test set, repeated cross-validation folds, and repeated training runs are different experimental units and should not be treated as interchangeable.

### 9. Statistical Significance

A statistical test evaluates a null hypothesis under specified assumptions. A small p-value indicates that the observed result would be relatively unusual under the null hypothesis; it does not measure the probability that a model is superior.

Statistical significance should be considered alongside effect size, uncertainty, practical relevance, and the number of comparisons performed.

## Implementation and Experiments

| File                                                       | Experiment                                                                                                                             |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `01_evaluation_metrics_and_baselines.py`                   | Calculate classification metrics, compare against a majority-class baseline, and understand how class imbalance affects evaluation.    |
| `02_data_splitting_and_cross_validation.py`                | Create train, validation, and test splits; perform cross-validation; and demonstrate how preprocessing leakage can distort evaluation. |
| `03_error_analysis_and_confusion_matrices.py`              | Inspect confusion matrices, class-specific performance, misclassified examples, and recurring error patterns.                          |
| `04_statistical_significance_and_experiment_comparison.py` | Compare models using paired differences, quantify uncertainty, and explore statistical tests and their assumptions.                    |

## Experimental Design Principles

- **Define the evaluation target:** Decide which task, population, and deployment conditions the evaluation should represent.
- **Establish baselines:** Measure simple approaches before introducing more complex models.
- **Prevent leakage:** Keep test information out of training, feature selection, preprocessing, and hyperparameter tuning.
- **Use appropriate splits:** Preserve class proportions when suitable, and use group-based or time-based splits when the data structure requires them.
- **Compare fairly:** Keep datasets, metrics, and evaluation conditions consistent across models.
- **Report uncertainty:** Include variation and uncertainty where the experimental design supports them.
- **Preserve reproducibility:** Record seeds, configurations, data versions, and relevant software versions.

## Expected Outcome

After completing this module, you will be able to evaluate NLP models beyond a single accuracy score, detect common experimental pitfalls, diagnose systematic prediction errors, and compare model performance with appropriate statistical caution.

These skills are essential for making evidence-based model-selection decisions and building reliable, production-oriented AI systems.
