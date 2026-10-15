# Dimensionality Reduction

## 1. Overview

Dimensionality reduction reduces the number of features in a dataset while preserving as much relevant information as possible.

High-dimensional datasets can increase computational costs, contain redundant information, make visualization difficult, and contribute to overfitting in some models.

Dimensionality reduction techniques create compact representations of data, making them useful for exploratory analysis, feature engineering, visualization, and machine learning pipelines.

## 2. Learning Objectives

- Understand the purpose and benefits of dimensionality reduction.
- Distinguish feature selection from feature extraction.
- Understand Principal Component Analysis (PCA).
- Interpret explained variance and cumulative explained variance.
- Select an appropriate number of principal components.
- Visualize high-dimensional data in lower-dimensional spaces.
- Evaluate classification performance before and after dimensionality reduction.
- Understand Singular Value Decomposition (SVD) and its applications.
- Compare nonlinear visualization techniques such as t-SNE and UMAP.
- Understand supervised dimensionality reduction with Linear Discriminant Analysis (LDA).
- Avoid data leakage when applying dimensionality reduction in ML pipelines.

## 3. Topics and Practical Exercises

| File                                 | Topic                                      | Key Concepts                                                       |
| ------------------------------------ | ------------------------------------------ | ------------------------------------------------------------------ |
| `01_pca.py`                          | Principal Component Analysis               | Feature scaling, principal components, 2D visualization            |
| `02_pca_explained_variance.py`       | Explained Variance and Component Selection | Explained variance ratio, cumulative variance, component selection |
| `03_pca_classification.py`           | Classification Before and After PCA        | Pipelines, feature reduction, model evaluation                     |
| `04_truncated_svd.py`                | Truncated Singular Value Decomposition     | SVD, sparse data, text feature representations                     |
| `05_tsne_and_umap.py`                | Nonlinear Dimensionality Reduction         | Local neighborhoods, embeddings, visualization                     |
| `06_lda_dimensionality_reduction.py` | Linear Discriminant Analysis               | Supervised dimensionality reduction, class separation              |

## 4. Key Concepts

### 4.1 Feature Selection vs. Feature Extraction

- **Feature selection:** Selects a subset of existing features.
- **Feature extraction:** Transforms existing features into a new representation.

PCA and SVD are feature extraction techniques. They create new components rather than simply selecting original features.

### 4.2 Principal Component Analysis (PCA)

PCA is a linear dimensionality reduction technique that transforms data into orthogonal principal components ordered by the variance they explain.

Important concepts:

- Feature scaling
- Principal components
- Explained variance ratio
- Cumulative explained variance
- Information loss

### 4.3 Truncated Singular Value Decomposition (SVD)

Truncated SVD reduces dimensionality using a limited number of singular components. Unlike conventional PCA preprocessing, it does not require centering the input matrix, making it particularly useful for sparse, high-dimensional data such as text feature matrices.

### 4.4 Nonlinear Dimensionality Reduction

- **t-SNE:** Emphasizes local neighborhood relationships and is primarily used for exploratory visualization.
- **UMAP:** Learns a lower-dimensional representation intended to preserve aspects of the data's structure.

Both methods can help visualize complex datasets and embeddings. Their visualizations should not automatically be interpreted as proof of distinct clusters or reliable global distances.

### 4.5 Linear Discriminant Analysis (LDA)

LDA is a supervised technique that uses class labels to find directions that improve class separation.

Unlike PCA, which maximizes overall variance without using labels, LDA focuses on separating known classes.

## 5. Best Practices

- Scale features before PCA when their scales differ substantially.
- Fit transformations only on training data.
- Use scikit-learn pipelines to prevent data leakage during cross-validation.
- Select components based on explained variance, predictive performance, computational cost, and interpretability.
- Do not assume that retaining more variance guarantees better classification.
- Evaluate reduced feature representations against a baseline using the original features.
- Use t-SNE and UMAP primarily for exploratory visualization unless their use in downstream predictive pipelines is carefully validated.
- Set random states where supported to improve reproducibility.
- Consider whether dimensionality reduction improves the actual objective before introducing additional complexity.

## 6. References

- [Scikit-learn: Decomposing signals in components](https://scikit-learn.org/stable/modules/decomposition.html)
- [Scikit-learn: Manifold learning](https://scikit-learn.org/stable/modules/manifold.html)
- [Scikit-learn: Linear and Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html)
