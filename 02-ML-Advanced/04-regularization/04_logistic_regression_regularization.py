# %% 1. Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% 2. Load the Dataset
dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# Make malignant the positive class:
# Original target: 0 = malignant, 1 = benign
y = pd.Series((dataset.target == 0).astype(int), name="malignant")

print("Dataset shape:", X.shape)
print("\nClass distribution:")
print(y.value_counts().rename(index={0: "Benign", 1: "Malignant"}))


# %% 3. Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))


# %% 4. Define Models
models = {
    "L2 C=0.1": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l2",
            C=0.1,
            max_iter=5000,
            random_state=42,
        )),
    ]),
    "L2 C=1.0": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=5000,
            random_state=42,
        )),
    ]),
    "L1 C=0.1": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.1,
            max_iter=5000,
            random_state=42,
        )),
    ]),
    "L1 C=1.0": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            max_iter=5000,
            random_state=42,
        )),
    ]),
}


# %% 5. Train and Evaluate Models
results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    })

    fitted_models[name] = model

results_df = pd.DataFrame(results).sort_values(
    "ROC-AUC",
    ascending=False,
)

print("\nTest Set Performance:")
print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


# %% 6. Inspect a Model in Detail
selected_model_name = "L2 C=1.0"
selected_model = fitted_models[selected_model_name]

predictions = selected_model.predict(X_test)

print(f"\nDetailed Evaluation: {selected_model_name}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Benign", "Malignant"],
    zero_division=0,
))


# %% 7. Compare Coefficients
coefficient_data = {"Feature": X.columns}

for name, model in fitted_models.items():
    estimator = model.named_steps["model"]
    coefficient_data[name] = estimator.coef_[0]

coefficients_df = pd.DataFrame(coefficient_data)

print("\nFirst 15 Coefficients:")
print(
    coefficients_df.head(15).to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 8. Compare Coefficient Norms and Sparsity
coefficient_summary = []

for name, model in fitted_models.items():
    coefficients = model.named_steps["model"].coef_[0]

    coefficient_summary.append({
        "Model": name,
        "L1 Norm": np.sum(np.abs(coefficients)),
        "L2 Norm": np.linalg.norm(coefficients),
        "Nonzero Coefficients": np.count_nonzero(coefficients),
        "Zero Coefficients": np.count_nonzero(coefficients == 0),
    })

coefficient_summary_df = pd.DataFrame(coefficient_summary)

print("\nCoefficient Summary:")
print(
    coefficient_summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 9. Visualize Coefficients
coefficients_df.set_index("Feature").plot(
    kind="bar",
    figsize=(16, 6),
)

plt.title("Logistic Regression: L1 vs. L2 Regularization")
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(rotation=90)
plt.legend(title="Model")
plt.tight_layout()
plt.show()


# %% 10. Investigate Regularization Strength
C_values = np.logspace(-3, 2, 20)

strength_results = []

for penalty in ["l1", "l2"]:
    for C in C_values:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty=penalty,
                solver="liblinear",
                C=C,
                max_iter=5000,
                random_state=42,
            )),
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        coefficients = model.named_steps["model"].coef_[0]

        strength_results.append({
            "Penalty": penalty.upper(),
            "C": C,
            "Accuracy": accuracy_score(y_test, predictions),
            "F1": f1_score(y_test, predictions),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "Coefficient L2 Norm": np.linalg.norm(coefficients),
            "Nonzero Coefficients": np.count_nonzero(coefficients),
        })

strength_df = pd.DataFrame(strength_results)

print("\nSelected Regularization Strength Results:")
print(
    strength_df.groupby("Penalty").head(5).to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 11. Plot Performance and Coefficient Behavior
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for penalty in ["L1", "L2"]:
    subset = strength_df[strength_df["Penalty"] == penalty]

    axes[0].semilogx(
        subset["C"],
        subset["ROC-AUC"],
        marker="o",
        label=penalty,
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

axes[0].set_title("ROC-AUC vs. C")
axes[0].set_xlabel("C (inverse regularization strength)")
axes[0].set_ylabel("ROC-AUC")

axes[1].set_title("Coefficient Magnitude vs. C")
axes[1].set_xlabel("C")
axes[1].set_ylabel("Coefficient L2 Norm")

axes[2].set_title("Nonzero Coefficients vs. C")
axes[2].set_xlabel("C")
axes[2].set_ylabel("Nonzero coefficient count")

for ax in axes:
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
