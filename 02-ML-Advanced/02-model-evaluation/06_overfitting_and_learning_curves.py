# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# %% 2. Load the dataset
data = load_breast_cancer()

X = data.data

# Original labels: 0 = malignant, 1 = benign.
# Redefine the positive class: 1 = malignant, 0 = benign.
y = (data.target == 0).astype(int)

print("Dataset shape:", X.shape)
print("Features:", X.shape[1])
print("Positive class: Malignant")
print("Negative class: Benign")
print("Class distribution:", np.bincount(y))


# %% 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))


# %% 4. Compare regularization strengths
# Smaller C means stronger regularization.
# Larger C means weaker regularization.
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

regularization_results = []

print("\nRegularization experiment")
print("-" * 82)
print(
    f"{'C':>10}"
    f"{'Train Accuracy':>18}"
    f"{'CV Accuracy':>16}"
    f"{'CV Std':>12}"
    f"{'Coefficient Norm':>20}"
)

for C in C_values:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            max_iter=5000,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )

    # Evaluate on validation folds drawn only from the training set.
    fold_train_scores = []
    fold_validation_scores = []

    for train_indices, validation_indices in cv.split(
        X_train,
        y_train,
    ):
        X_fold_train = X_train[train_indices]
        y_fold_train = y_train[train_indices]

        X_fold_validation = X_train[validation_indices]
        y_fold_validation = y_train[validation_indices]

        fold_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=5000,
                random_state=42,
            ),
        )

        fold_model.fit(
            X_fold_train,
            y_fold_train,
        )

        fold_train_predictions = fold_model.predict(
            X_fold_train,
        )

        fold_validation_predictions = fold_model.predict(
            X_fold_validation,
        )

        fold_train_scores.append(
            accuracy_score(
                y_fold_train,
                fold_train_predictions,
            )
        )

        fold_validation_scores.append(
            accuracy_score(
                y_fold_validation,
                fold_validation_predictions,
            )
        )

    cv_train_accuracy = np.mean(fold_train_scores)
    cv_accuracy = np.mean(fold_validation_scores)
    cv_std = np.std(fold_validation_scores)

    # Coefficient norm provides a rough indication of coefficient magnitude.
    coefficients = model[-1].coef_
    coefficient_norm = np.linalg.norm(coefficients)

    regularization_results.append(
        {
            "C": C,
            "train_accuracy": train_accuracy,
            "cv_train_accuracy": cv_train_accuracy,
            "cv_accuracy": cv_accuracy,
            "cv_std": cv_std,
            "coefficient_norm": coefficient_norm,
        }
    )

    print(
        f"{C:>10.3g}"
        f"{train_accuracy:>18.4f}"
        f"{cv_accuracy:>16.4f}"
        f"{cv_std:>12.4f}"
        f"{coefficient_norm:>20.4f}"
    )


# %% 5. Plot regularization strength and generalization
train_scores = [
    result["cv_train_accuracy"]
    for result in regularization_results
]

validation_scores = [
    result["cv_accuracy"]
    for result in regularization_results
]

validation_stds = [
    result["cv_std"]
    for result in regularization_results
]

fig, ax = plt.subplots(figsize=(9, 6))

ax.semilogx(
    C_values,
    train_scores,
    marker="o",
    label="Cross-validation fold training accuracy",
)

ax.errorbar(
    C_values,
    validation_scores,
    yerr=validation_stds,
    marker="s",
    capsize=4,
    label="Cross-validation validation accuracy",
)

ax.set_xlabel("Regularization Parameter C (log scale)")
ax.set_ylabel("Accuracy")
ax.set_title("Regularization and Generalization")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %% 6. Plot coefficient magnitude
coefficient_norms = [
    result["coefficient_norm"]
    for result in regularization_results
]

fig, ax = plt.subplots(figsize=(9, 6))

ax.semilogx(
    C_values,
    coefficient_norms,
    marker="o",
)

ax.set_xlabel("Regularization Parameter C (log scale)")
ax.set_ylabel("L2 Coefficient Norm")
ax.set_title("Regularization Strength and Coefficient Magnitude")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %% 7. Calculate learning curves
# Use a representative regularization strength.
# The goal here is to study data quantity, not tune C.
selected_C = 1.0

learning_curve_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=selected_C,
        max_iter=5000,
        random_state=42,
    ),
)

train_sizes = np.linspace(0.1, 1.0, 8)

sizes, train_scores, validation_scores = learning_curve(
    estimator=learning_curve_model,
    X=X_train,
    y=y_train,
    train_sizes=train_sizes,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    shuffle=True,
    random_state=42,
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)

validation_mean = validation_scores.mean(axis=1)
validation_std = validation_scores.std(axis=1)

print("\nLearning curve results")
print("-" * 68)
print(
    f"{'Samples':>10}"
    f"{'Train Accuracy':>18}"
    f"{'CV Accuracy':>16}"
    f"{'CV Std':>12}"
)

for i, size in enumerate(sizes):
    print(
        f"{size:>10}"
        f"{train_mean[i]:>18.4f}"
        f"{validation_mean[i]:>16.4f}"
        f"{validation_std[i]:>12.4f}"
    )


# %% 8. Plot the learning curve
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    sizes,
    train_mean,
    marker="o",
    label="Training accuracy",
)

ax.plot(
    sizes,
    validation_mean,
    marker="s",
    label="Cross-validation accuracy",
)

ax.fill_between(
    sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15,
)

ax.fill_between(
    sizes,
    validation_mean - validation_std,
    validation_mean + validation_std,
    alpha=0.15,
)

ax.set_xlabel("Number of Training Samples")
ax.set_ylabel("Accuracy")
ax.set_title(
    f"Learning Curve: Logistic Regression (C={selected_C})"
)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %% 9. Evaluate the selected configuration on the test set
# This is a demonstration of final evaluation.
# The test set has not been used to choose C.
final_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=selected_C,
        max_iter=5000,
        random_state=42,
    ),
)

final_model.fit(X_train, y_train)

test_predictions = final_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)

print("\nFinal test evaluation")
print("-" * 30)
print(f"Selected C: {selected_C}")
print(f"Test accuracy: {test_accuracy:.4f}")
