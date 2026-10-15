# %% 1. Imports
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% 2. Regression: Load and Split the Dataset
housing = fetch_california_housing()

X_reg = pd.DataFrame(housing.data, columns=housing.feature_names)
y_reg = pd.Series(housing.target, name="MedHouseVal")

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42,
)

regression_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("Regression training samples:", len(X_reg_train))
print("Regression test samples:", len(X_reg_test))


# %% 3. Regression: Define Cross-Validation Searches
alphas = np.logspace(-3, 2, 20)

regression_searches = {
    "Ridge": GridSearchCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]),
        param_grid={
            "model__alpha": alphas,
        },
        scoring="neg_root_mean_squared_error",
        cv=regression_cv,
        n_jobs=-1,
    ),
    "Lasso": GridSearchCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=20000)),
        ]),
        param_grid={
            "model__alpha": alphas,
        },
        scoring="neg_root_mean_squared_error",
        cv=regression_cv,
        n_jobs=-1,
    ),
}


# %% 4. Regression: Select and Evaluate Models
regression_results = {}
regression_models = {}

for name, search in regression_searches.items():
    search.fit(X_reg_train, y_reg_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_reg_test)

    regression_models[name] = best_model

    regression_results[name] = {
        "Best Alpha": search.best_params_["model__alpha"],
        "Best CV RMSE": -search.best_score_,
        "Test MAE": mean_absolute_error(y_reg_test, predictions),
        "Test RMSE": np.sqrt(
            mean_squared_error(y_reg_test, predictions)
        ),
        "Test R2": r2_score(y_reg_test, predictions),
    }

regression_results_df = pd.DataFrame(regression_results).T

print("\nRegression Model Selection Results:")
print(
    regression_results_df.to_string(
        float_format=lambda value: f"{value:.4f}"
    )
)


# %% 5. Regression: Inspect Selected Coefficients
coefficient_data = {"Feature": X_reg.columns}

for name, model in regression_models.items():
    coefficient_data[name] = model.named_steps["model"].coef_

regression_coefficients_df = pd.DataFrame(coefficient_data)

print("\nSelected Regression Model Coefficients:")
print(
    regression_coefficients_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

for name, model in regression_models.items():
    coefficients = model.named_steps["model"].coef_

    print(f"\n{name}")
    print(
        "Coefficient L2 norm:",
        round(np.linalg.norm(coefficients), 4),
    )
    print(
        "Nonzero coefficients:",
        np.count_nonzero(coefficients),
    )


# %% 6. Classification: Load and Split the Dataset
cancer = load_breast_cancer()

X_clf = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names,
)

# Positive class: malignant
# Original target: 0 = malignant, 1 = benign
y_clf = pd.Series(
    (cancer.target == 0).astype(int),
    name="malignant",
)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf,
    y_clf,
    test_size=0.2,
    stratify=y_clf,
    random_state=42,
)

classification_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("\nClassification training samples:", len(X_clf_train))
print("Classification test samples:", len(X_clf_test))


# %% 7. Classification: Define Cross-Validation Searches
C_values = np.logspace(-3, 2, 20)

classification_searches = {
    "Logistic L1": GridSearchCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                l1_ratio=1.0,
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            )),
        ]),
        param_grid={
            "model__C": C_values,
        },
        scoring="roc_auc",
        cv=classification_cv,
        n_jobs=-1,
    ),
    "Logistic L2": GridSearchCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                l1_ratio=0.0,
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            )),
        ]),
        param_grid={
            "model__C": C_values,
        },
        scoring="roc_auc",
        cv=classification_cv,
        n_jobs=-1,
    ),
}


# %% 8. Classification: Select and Evaluate Models
classification_results = {}
classification_models = {}

for name, search in classification_searches.items():
    search.fit(X_clf_train, y_clf_train)

    best_model = search.best_estimator_

    predictions = best_model.predict(X_clf_test)
    probabilities = best_model.predict_proba(X_clf_test)[:, 1]

    classification_models[name] = best_model

    classification_results[name] = {
        "Best C": search.best_params_["model__C"],
        "Best CV ROC-AUC": search.best_score_,
        "Test Accuracy": accuracy_score(
            y_clf_test,
            predictions,
        ),
        "Test Precision": precision_score(
            y_clf_test,
            predictions,
            zero_division=0,
        ),
        "Test Recall": recall_score(
            y_clf_test,
            predictions,
            zero_division=0,
        ),
        "Test F1": f1_score(
            y_clf_test,
            predictions,
            zero_division=0,
        ),
        "Test ROC-AUC": roc_auc_score(
            y_clf_test,
            probabilities,
        ),
    }

classification_results_df = pd.DataFrame(
    classification_results
).T

print("\nClassification Model Selection Results:")
print(
    classification_results_df.to_string(
        float_format=lambda value: f"{value:.4f}"
    )
)


# %% 9. Classification: Inspect Selected Coefficients
classification_coefficient_data = {
    "Feature": X_clf.columns,
}

for name, model in classification_models.items():
    classification_coefficient_data[name] = (
        model.named_steps["model"].coef_[0]
    )

classification_coefficients_df = pd.DataFrame(
    classification_coefficient_data
)

print("\nSelected Classification Model Coefficients:")
print(
    classification_coefficients_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

for name, model in classification_models.items():
    coefficients = model.named_steps["model"].coef_[0]

    print(f"\n{name}")
    print(
        "Coefficient L2 norm:",
        round(np.linalg.norm(coefficients), 4),
    )
    print(
        "Nonzero coefficients:",
        np.count_nonzero(coefficients),
    )
    print(
        "Zero coefficients:",
        np.count_nonzero(coefficients == 0),
    )


# %% 10. Summarize the Model Selection Workflow
print("\nWorkflow Summary")
print("1. Split the data into training and test sets.")
print("2. Fit preprocessing and models within cross-validation.")
print("3. Select hyperparameters using training data only.")
print("4. Refit each best estimator on the full training set.")
print("5. Evaluate the selected estimator on the held-out test set.")
print("6. Report test performance without using it to retune the model.")
