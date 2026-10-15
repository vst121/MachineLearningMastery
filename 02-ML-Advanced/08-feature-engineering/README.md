# Feature Engineering

## 1. Overview

Feature engineering transforms raw data into meaningful representations that help machine learning models learn patterns more effectively.

This section covers essential preprocessing, feature transformation, categorical encoding, feature construction, feature selection, and evaluation techniques using Python, pandas, and scikit-learn.

## 2. Learning Objectives

- Build reusable preprocessing pipelines.
- Handle missing values in numerical and categorical features.
- Apply numerical transformations and feature scaling.
- Encode categorical variables for machine learning models.
- Create polynomial features and interaction terms.
- Select relevant features and reduce unnecessary complexity.
- Evaluate feature engineering strategies using cross-validation.
- Prevent data leakage throughout the ML workflow.

## 3. Topics and Practical Exercises

| File                                       | Topic                              | Key Concepts                                                                                        |
| ------------------------------------------ | ---------------------------------- | --------------------------------------------------------------------------------------------------- |
| `01_preprocessing_pipeline.py`             | Preprocessing Pipelines            | `Pipeline`, `ColumnTransformer`, `SimpleImputer`, mixed data types                                  |
| `02_feature_transformation_and_scaling.py` | Feature Transformation and Scaling | Log transformation, power transformation, `StandardScaler`, `MinMaxScaler`, `RobustScaler`          |
| `03_categorical_encoding.py`               | Categorical Encoding               | `OneHotEncoder`, `OrdinalEncoder`, unknown categories                                               |
| `04_polynomial_features_and_selection.py`  | Feature Construction and Selection | `PolynomialFeatures`, interaction terms, variance thresholds, statistical and model-based selection |
| `05_feature_engineering_evaluation.py`     | Feature Engineering Evaluation     | Cross-validation, baseline comparison, pipelines, data leakage prevention                           |

## 4. Key Concepts

### Data Preprocessing

- Missing-value imputation
- Numerical and categorical feature handling
- `Pipeline` and `ColumnTransformer`
- Reproducible preprocessing workflows

### Feature Transformation and Scaling

- Log and power transformations
- Standardization
- Min-max scaling
- Robust scaling
- Choosing transformations based on data distributions and model requirements

### Categorical Encoding

- One-hot encoding
- Ordinal encoding
- Handling unseen categories
- Choosing encoding strategies based on feature meaning

### Feature Construction and Selection

- Polynomial features
- Feature interactions
- Variance threshold
- Statistical feature selection
- Model-based feature selection

### Evaluation

- Cross-validation
- Baseline comparisons
- Data leakage prevention
- Generalization to unseen data

## 5. Best Practices

- Split data before fitting preprocessing transformations.
- Fit imputers, scalers, encoders, and feature selectors on training data only.
- Use `Pipeline` and `ColumnTransformer` to keep preprocessing consistent.
- Handle unknown categorical values explicitly.
- Apply scaling when required by the model or algorithm.
- Avoid unnecessary transformations and excessive feature creation.
- Evaluate feature engineering choices using cross-validation.
- Keep the test set independent until final evaluation.
- Compare engineered features against a baseline rather than assuming complexity improves performance.

## 6. References

- [Scikit-learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Scikit-learn: Pipelines and Composite Estimators](https://scikit-learn.org/stable/modules/compose.html)
- [Scikit-learn: Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Scikit-learn: Polynomial and Spline Features](https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
