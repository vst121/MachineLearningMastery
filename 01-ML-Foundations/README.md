# Machine Learning Foundations 🚀

This level introduces four fundamental areas of Machine Learning: **Regression, Support Vector Machines (SVM), K-Means Clustering, and Optimization**.

The goal is to understand how these algorithms work, implement them in Python, evaluate their results, and recognize when to use each one.

## Learning Roadmap

| Topic                           | Learning Focus                                                          | Type                  |
| ------------------------------- | ----------------------------------------------------------------------- | --------------------- |
| Regression                      | Predicting continuous values                                            | Supervised Learning   |
| Support Vector Machines (SVM)   | Classification, decision boundaries, margins, and kernels               | Supervised Learning   |
| K-Means Clustering              | Discovering groups in unlabeled data                                    | Unsupervised Learning |
| Optimization and Cost Functions | Understanding model error, Gradient Descent, and parameter optimization | Core ML Concepts      |

## Repository Structure

```text
01-ML-Foundations/
├── 01-regression/
│   ├── 01_regression_demo.py
│   └── ...
├── 02-classification/
│   ├── 01_svm_demo.py
│   └── ...
├── 03-clustering/
│   ├── 01_kmeans_demo.py
│   └── ...
├── 04-optimization/
│   ├── 01_cost_function_demo.py
│   └── ...
└── README.md
```

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
uv run python 01-regression/01_regression_demo.py
uv run python 02-classification/01_svm_demo.py
uv run python 03-clustering/01_kmeans_demo.py
uv run python 04-optimization/01_cost_function_demo.py
```

### 2. Interactive Cell-by-Cell Execution (`# %%`)

Each script is structured with Jupyter-style cell markers (`# %%`). In VS Code, this allows you to run individual sections of the file interactively, letting you inspect variables, tweak hyperparameters, and view Matplotlib plots block by block without rerunning the entire script.

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

- \(\hat{y}\) is the predicted value.
- \(x_i\) represents an input feature.
- \(\theta_0\) is the intercept.
- \(\theta_i\) represents a learned coefficient.

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

## 2. Support Vector Machines (SVM)

Support Vector Machines are supervised learning algorithms used primarily for classification. Support Vector Regression (SVR) applies related principles to regression.

### Key Concepts

- Decision boundaries and separating hyperplanes
- Maximum-margin classification
- Support vectors
- Hard and soft margins
- The regularization parameter \(C\)
- Linear and nonlinear kernels
- Feature scaling

### Decision Boundary

A linear decision boundary can be represented as:

$$
\mathbf{w}^{T}\mathbf{x}+b=0
$$

Where \(\mathbf{w}\) defines the orientation of the boundary and \(b\) is its bias.

The classifier predicts a class based on the sign of the decision function:

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

These constraints encourage correct classification while maximizing the margin between the classes.

### Kernels

Kernels allow SVMs to model nonlinear decision boundaries.

Common choices include:

- **Linear:** Suitable for approximately linearly separable data.
- **Polynomial:** Models polynomial relationships.
- **RBF:** Can model more flexible nonlinear boundaries.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For imbalanced datasets, precision, recall, and F1-score can be more informative than accuracy alone.

### Exercises

1. Train an SVM classifier on a sample dataset.
2. Compare linear and RBF kernels.
3. Observe how changing \(C\) affects the decision boundary.
4. Compare performance with and without feature scaling.

## 3. K-Means Clustering

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

- \(\theta_j\) is a model parameter.
- \(\alpha\) is the learning rate.
- \(\frac{\partial J}{\partial\theta_j}\) is the partial derivative of the cost with respect to that parameter.

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
4. Compare different learning rates and plot the cost over iterations.
5. Compare the results with a solution from scikit-learn.

## Common Machine Learning Workflow

Across all four topics, follow a consistent workflow:

1. Understand the problem and dataset.
2. Inspect and prepare the data.
3. Split data into training and test sets when appropriate.
4. Build a baseline model.
5. Train or optimize the model.
6. Evaluate results using suitable metrics.
7. Inspect errors and limitations.
8. Document findings and experiments.

**Avoid data leakage:** Information from the test set must not influence training or preprocessing decisions. Fit preprocessing transformations on the training data and apply them to the test data.

## Technology Stack

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- uv

## Learning Resources

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [Clustering](https://scikit-learn.org/stable/modules/clustering.html)
