# Machine Learning Foundations 🚀

This level introduces four fundamental areas of Machine Learning: **Regression, Classification, Clustering, and Optimization**.

The goal is to understand how these algorithms work, implement them in Python, evaluate their results, and recognize when to use each one.

## Learning Roadmap

| Topic                           | Learning Focus                                                           | Type                  |
| ------------------------------- | ------------------------------------------------------------------------ | --------------------- |
| Regression                      | Predicting continuous values                                             | Supervised Learning   |
| Classification                  | Predicting categorical labels, decision boundaries, margins, and kernels | Supervised Learning   |
| Clustering                      | Discovering groups in unlabeled data                                     | Unsupervised Learning |
| Optimization and Cost Functions | Understanding model error, Gradient Descent, and parameter optimization  | Core ML Concepts      |

## Repository Structure

```text
01-ML-Foundations/
├── 01-regression/
│   ├── 01_regression.py
│   └── ...
├── 02-classification/
│   ├── 01_svm.py
│   ├── ...
├── 03-clustering/
│   ├── 01_kmeans.py
│   └── ...
├── 04-optimization/
│   ├── 01_cost_function.py
│   └── ...
└── README.md
```

The filenames above describe the intended learning sequence. Keep existing filenames unchanged if they differ in your actual repository.

## Prerequisites

- Python 3.12+
- Basic Python programming
- Basic algebra and familiarity with functions
- Basic knowledge of NumPy, pandas, and data visualization is helpful

## Setup

From the repository root:

```bash
cd 01-ML-Foundations
```

## Run the Demonstrations

You can execute your scripts in two flexible ways depending on your workflow.

### 1. Full Script Execution via Terminal

Run the entire script end-to-end using `uv run`:

```bash
uv run python 01-regression/01_regression.py
uv run python 02-classification/01_svm.py
uv run python 03-clustering/01_kmeans.py
uv run python 04-optimization/01_cost_function.py
```

### 2. Interactive Cell-by-Cell Execution (`# %%`)

Each script is structured with Jupyter-style cell markers (`# %%`). In VS Code, this allows you to run individual sections of the file interactively, inspect variables, adjust hyperparameters, and view Matplotlib plots without rerunning the entire script.

## 1. Regression

Regression predicts a **continuous numerical value** from input features.

Examples include predicting house prices, estimating demand, and forecasting numerical measurements.

### Key Concepts

- Simple and multiple linear regression
- Features, targets, model parameters, and predictions
- Cost functions and model fitting
- Gradient Descent
- Training and test datasets
- Model evaluation

### Linear Regression

A linear regression model with \(n\) features can be expressed as:

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \cdots + \theta_n x_n
$$

Where:

- $$\hat{y}$$ is the predicted value.
- $$x_i $$ represents an input feature.
- $$\theta_0$$ is the intercept.
- $$\theta_i $$ represents a learned coefficient.

### Mean Squared Error

Mean Squared Error (MSE) measures the average squared difference between actual and predicted values:

$$
\text{MSE} = \frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

Here, \(m\) is the number of samples, \(y_i\) is the actual value, and \(\hat{y}\_i\) is the predicted value.

### Evaluation Metrics

- **MAE:** Average absolute prediction error.
- **MSE:** Average squared prediction error.
- **RMSE:** Square root of MSE, expressed in the target's units.
- **\(R^2\):** Measures model performance relative to a baseline that predicts the target mean.

### Exercises

1. Train a simple linear regression model using one feature.
2. Extend the model to multiple features.
3. Compare MAE, MSE, RMSE, and \(R^2\).
4. Investigate how outliers affect regression performance.

## 2. Classification

Classification is a supervised learning task that predicts **categorical labels** from input features. This section introduces Support Vector Machines (SVM) as a foundation for understanding classification algorithms.

### Key Concepts

- Classification versus regression
- Features, labels, and class predictions
- Decision boundaries and separating hyperplanes
- Maximum-margin classification
- Support vectors
- Hard and soft margins
- Linear and nonlinear kernels
- Feature scaling
- Cross-validation
- Hyperparameter tuning
- Model generalization and overfitting

### Support Vector Machines (SVM)

SVMs seek a decision boundary that separates classes while maximizing the margin between them. Support vectors are the training samples that determine the learned boundary.

A linear decision boundary can be represented as:

$$
\mathbf{w}^{T}\mathbf{x}+b=0
$$

Where \(\mathbf{w}\) defines the orientation of the boundary and \(b\) is the bias.

For binary classification, the prediction is determined by the sign of the decision function:

$$
f(\mathbf{x})=\text{sign}(\mathbf{w}^{T}\mathbf{x}+b)
$$

### Maximum Margin

For a linearly separable dataset, a hard-margin SVM seeks to minimize:

$$
\min_{\mathbf{w},b}\frac{1}{2}\|\mathbf{w}\|^2
$$

Subject to:

$$
y_i(\mathbf{w}^{T}\mathbf{x}_i+b)\geq 1
$$

These constraints enforce correct classification and maximize the margin under the assumption that the classes are linearly separable.

### Kernels

Kernels allow SVMs to model nonlinear decision boundaries.

- **Linear:** Suitable for approximately linearly separable data.
- **Polynomial:** Models polynomial relationships.
- **RBF:** Models more flexible nonlinear relationships.

### The `C` and `gamma` Parameters

The `C` parameter controls the trade-off between a wider margin and training classification errors.

- **Smaller `C`:** Allows more training errors in exchange for stronger regularization.
- **Larger `C`:** Penalizes training errors more strongly and may produce a more complex boundary.

For an RBF kernel, `gamma` controls how quickly the influence of individual training samples decreases with distance.

- **Smaller `gamma`:** Produces broader influence and generally smoother decision boundaries.
- **Larger `gamma`:** Produces more localized influence and can create more complex boundaries.

The effects of `C` and `gamma` interact. Cross-validation helps identify combinations that generalize well to unseen data.

### Cross-Validation and Hyperparameter Tuning

- **StratifiedKFold:** Preserves class proportions across cross-validation folds.
- **GridSearchCV:** Evaluates specified hyperparameter combinations.
- **RandomizedSearchCV:** Samples hyperparameter combinations from specified distributions.
- **Learning curves:** Compare training and cross-validation performance at different training-set sizes.
- **Decision-boundary visualization:** Compare linear and nonlinear separation and inspect support vectors.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For imbalanced datasets, precision, recall, and F1-score can be more informative than accuracy alone.

### Exercises

1. Train an SVM classifier on a sample dataset.
2. Compare linear, polynomial, and RBF kernels.
3. Observe how changing `C` affects the decision boundary.
4. Compare performance with and without feature scaling.
5. Use `StratifiedKFold` for cross-validation.
6. Tune hyperparameters with `GridSearchCV`.
7. Compare grid search with `RandomizedSearchCV`.
8. Plot learning curves to investigate underfitting and overfitting.
9. Experiment with different `gamma` values for the RBF kernel.
10. Visualize decision boundaries and support vectors.

## 3. Clustering

K-Means is an unsupervised learning algorithm that partitions data into \(K\) clusters based on distances to cluster centroids.

Unlike regression and classification, K-Means does not require labeled target values.

### Key Concepts

- Unsupervised learning
- Centroids and cluster assignments
- Choosing the number of clusters \(K\)
- Within-cluster sum of squares (WCSS)
- The elbow method
- Silhouette score
- Feature scaling

### Clustering Objective

K-Means aims to minimize the within-cluster sum of squared distances:

$$
J = \sum_{k=1}^{K}\sum_{\mathbf{x}_i\in C_k}
\|\mathbf{x}_i-\boldsymbol{\mu}_k\|^2
$$

Where:

- \(K\) is the number of clusters.
- \(C_k\) is the set of samples assigned to cluster \(k\).
- \(\boldsymbol{\mu}\_k\) is the centroid of cluster \(k\).

### Algorithm

1. Initialize \(K\) centroids.
2. Assign each sample to its nearest centroid.
3. Recalculate each centroid as the mean of its assigned samples.
4. Repeat assignment and update steps until convergence or a stopping condition is reached.

### Choosing \(K\)

The **elbow method** compares WCSS across different values of \(K\). The preferred value may be near the point where additional clusters produce diminishing improvements.

The **silhouette score** measures how well each sample fits its assigned cluster compared with neighboring clusters. Higher scores generally indicate better-separated clusters.

### Exercises

1. Apply K-Means to a dataset without labels.
2. Visualize the resulting clusters and centroids.
3. Compare several values of \(K\) using the elbow method.
4. Evaluate cluster separation using the silhouette score.
5. Observe how feature scaling changes the results.

## 4. Optimization and Cost Functions

Optimization is the process of finding model parameters that minimize or maximize an objective. In Machine Learning, optimization commonly means minimizing a cost function to improve model predictions.

This topic introduces the foundations of how models learn and how their parameters are adjusted to reduce prediction errors.

### Key Concepts

- Cost functions and loss functions
- Mean Squared Error (MSE) and Mean Absolute Error (MAE)
- Model parameters and gradients
- Gradient Descent
- Learning rate
- Convergence and stopping criteria
- Local and global minima

### Cost Function

A cost function measures how well a model's predictions match the actual values across a dataset.

For regression, Mean Squared Error is a common choice:

$$
J(\boldsymbol{\theta}) =
\frac{1}{m}\sum_{i=1}^{m}
\left(y_i-\hat{y}_i\right)^2
$$

Where:

- \(J(\boldsymbol{\theta})\) is the cost associated with the model parameters.
- \(m\) is the number of samples.
- \(y_i\) is the actual value.
- \(\hat{y}\_i\) is the predicted value.

The objective is to find model parameters that minimize the cost.

### Gradient Descent

Gradient Descent iteratively updates model parameters in the direction that reduces the cost.

The update rule is:

$$
\theta_j :=
\theta_j-\alpha\frac{\partial J(\boldsymbol{\theta})}{\partial\theta_j}
$$

Where:

- $$\\theta_j\$$ is a model parameter.
- $$\\alpha\$$ is the learning rate.
- $$\\frac{\partial J}{\partial\theta_j}\$$ is the partial derivative of the cost with respect to that parameter.

The learning rate controls the size of each update. If it is too small, convergence can be slow. If it is too large, optimization may overshoot or fail to converge.

### Learning Rate and Convergence

During optimization, track the cost across iterations to understand whether the model is learning effectively.

- A steadily decreasing cost usually indicates progress.
- A very small learning rate can require many iterations.
- An excessively large learning rate can cause oscillation or divergence.
- A stopping criterion can be based on a maximum number of iterations or sufficiently small changes in cost.

### Exercises

1. Implement Gradient Descent from scratch using Python and NumPy.
2. Minimize a simple mathematical cost function.
3. Apply Gradient Descent to fit a linear regression model.
4. Compare different learning rates and
