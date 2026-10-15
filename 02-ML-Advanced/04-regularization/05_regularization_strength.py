# %% 1. Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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

print("Regression training samples:", len(X_reg_train))
print("Regression test samples:", len(X_reg_test))


# %% 3. Regression: Evaluate Ridge and Lasso Across Alpha Values
alphas = np.logspace(-3, 2, 20)

regression_results = []

for model_name, estimator in [
    ("Ridge", Ridge()),
    ("Lasso", Lasso(max_iter=20000)),
]:
    for alpha in alphas:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator.set_params(alpha=alpha)),
        ])

        pipeline.fit(X_reg_train, y_reg_train)

        predictions = pipeline.predict(X_reg_test)
        coefficients = pipeline.named_steps["model"].coef_

        regression_results.append({
            "Model": model_name,
            "Alpha": alpha,
            "MAE": mean_absolute_error(y_reg_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_reg_test, predictions)),
            "R2": r2_score(y_reg_test, predictions),
            "Coefficient L2 Norm": np.linalg.norm(coefficients),
            "Nonzero Coefficients": np.count_nonzero(coefficients),
        })

regression_df = pd.DataFrame(regression_results)

print("\nRegression Results:")
print(
    regression_df.groupby("Model").head(5).to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 4. Regression: Plot Performance and Complexity
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for model_name in ["Ridge", "Lasso"]:
    subset = regression_df[regression_df["Model"] == model_name]

    axes[0].semilogx(
        subset["Alpha"], subset["RMSE"], marker="o", label=model_name
    )
    axes[1].semilogx(
        subset["Alpha"],
        subset["Coefficient L2 Norm"],
        marker="o",
        label=model_name,
    )
    axes[2].semilogx(
        subset["Alpha"],
        subset["Nonzero Coefficients"],
        marker="o",
        label=model_name,
    )

axes[0].set_title("Regression RMSE vs. Alpha")
axes[0].set_ylabel("RMSE")

axes[1].set_title("Coefficient Magnitude vs. Alpha")
axes[1].set_ylabel("Coefficient L2 Norm")

axes[2].set_title("Nonzero Coefficients vs. Alpha")
axes[2].set_ylabel("Nonzero coefficient count")

for ax in axes:
    ax.set_xlabel("Alpha (regularization strength)")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()


# %% 5. Regression: Select Alpha Using Cross-Validation
regression_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

for model_name, estimator in [
    ("Ridge", Ridge()),
    ("Lasso", Lasso(max_iter=20000)),
]:
    search = GridSearchCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]),
        param_grid={
            "model__alpha": alphas,
        },
        scoring="neg_root_mean_squared_error",
        cv=regression_cv,
        n_jobs=-1,
    )

    search.fit(X_reg_train, y_reg_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_reg_test)

    print(f"\n{model_name} Cross-Validation")
    print("Best alpha:", search.best_params_["model__alpha"])
    print("Best CV RMSE:", -search.best_score_)
    print("Test RMSE:", np.sqrt(mean_squared_error(y_reg_test, predictions)))
    print("Test R2:", r2_score(y_reg_test, predictions))


# %% 6. Classification: Load and Split the Dataset
cancer = load_breast_cancer()

X_clf = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# Positive class: malignant
y_clf = pd.Series((cancer.target == 0).astype(int), name="malignant")

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf,
    y_clf,
    test_size=0.2,
    stratify=y_clf,
    random_state=42,
)

print("\nClassification training samples:", len(X_clf_train))
print("Classification test samples:", len(X_clf_test))


# %% 7. Classification: Evaluate Logistic Regression Across C Values
C_values = np.logspace(-3, 2, 20)

classification_results = []

for penalty in ["l1", "l2"]:
    for C in C_values:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty=penalty,
                solver="liblinear",
                C=C,
                max_iter=5000,
                random_state=42,
            )),
        ])

        pipeline.fit(X_clf_train, y_clf_train)

        predictions = pipeline.predict(X_clf_test)
        probabilities = pipeline.predict_proba(X_clf_test)[:, 1]
        coefficients = pipeline.named_steps["model"].coef_[0]

        classification_results.append({
            "Penalty": penalty.upper(),
            "C": C,
            "F1": f1_score(y_clf_test, predictions),
            "ROC-AUC": roc_auc_score(y_clf_test, probabilities),
            "Coefficient L2 Norm": np.linalg.norm(coefficients),
            "Nonzero Coefficients": np.count_nonzero(coefficients),
        })

classification_df = pd.DataFrame(classification_results)


# %% 8. Classification: Plot Performance and Complexity
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for penalty in ["L1", "L2"]:
    subset = classification_df[classification_df["Penalty"] == penalty]

    axes[0].semilogx(
        subset["C"], subset["ROC-AUC"], marker="o", label=penalty
    )
    axes[1].semilogx(
        subset["C"],
        subset["Coefficient L2 Norm"],
        marker="o",
        label=penalty,
    )
    axes[2].semilogx(
        subset["C"],
        subset["Nonzero Coefficients"],
        marker="o",
        label=penalty,
    )

axes[0].set_title("Classification ROC-AUC vs. C")
axes[0].set_ylabel("ROC-AUC")

axes[1].set_title("Coefficient Magnitude vs. C")
axes[1].set_ylabel("Coefficient L2 Norm")

axes[2].set_title("Nonzero Coefficients vs. C")
axes[2].set_ylabel("Nonzero coefficient count")

for ax in axes:
    ax.set_xlabel("C (inverse regularization strength)")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()


# %% 9. Classification: Select C Using Cross-Validation
classification_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

for penalty in ["l1", "l2"]:
    search = GridSearchCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty=penalty,
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
    )

    search.fit(X_clf_train, y_clf_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_clf_test)
    probabilities = best_model.predict_proba(X_clf_test)[:, 1]

    print(f"\nLogistic Regression {penalty.upper()} Cross-Validation")
    print("Best C:", search.best_params_["model__C"])
    print("Best CV ROC-AUC:", search.best_score_)
    print("Test F1:", f1_score(y_clf_test, predictions))
    print("Test ROC-AUC:", roc_auc_score(y_clf_test, probabilities))
