# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% 2. Load the dataset
data = load_breast_cancer()

X = data.data
y = data.target

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Classes: {data.target_names}")

# %% 3. Create training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

# %% 4. Define regularization strengths
C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

results = []

# %% 5. Train and evaluate models
for C in C_values:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    train_probabilities = model.predict_proba(X_train)
    val_probabilities = model.predict_proba(X_val)

    train_accuracy = accuracy_score(y_train, train_predictions)
    val_accuracy = accuracy_score(y_val, val_predictions)

    train_loss = log_loss(y_train, train_probabilities)
    val_loss = log_loss(y_val, val_probabilities)

    classifier = model.named_steps["logisticregression"]
    coefficient_norm = np.linalg.norm(classifier.coef_)

    results.append(
        {
            "C": C,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "coefficient_norm": coefficient_norm,
            "model": model,
        }
    )

# %% 6. Display the results
print(
    f"{'C':>8} {'Train Acc':>11} {'Val Acc':>9} "
    f"{'Train Loss':>11} {'Val Loss':>9} {'Coef Norm':>10}"
)
print("-" * 65)

for result in results:
    print(
        f"{result['C']:8.3f} "
        f"{result['train_accuracy']:11.3f} "
        f"{result['val_accuracy']:9.3f} "
        f"{result['train_loss']:11.3f} "
        f"{result['val_loss']:9.3f} "
        f"{result['coefficient_norm']:10.3f}"
    )

# %% 7. Plot accuracy against regularization strength
C_values_array = np.array(C_values)

train_accuracies = [r["train_accuracy"] for r in results]
val_accuracies = [r["val_accuracy"] for r in results]

plt.figure(figsize=(9, 5))
plt.semilogx(
    C_values_array,
    train_accuracies,
    marker="o",
    label="Training accuracy",
)
plt.semilogx(
    C_values_array,
    val_accuracies,
    marker="s",
    label="Validation accuracy",
)

plt.xlabel("C (inverse regularization strength)")
plt.ylabel("Accuracy")
plt.title("Regularization Strength vs. Accuracy")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% 8. Plot validation log loss
val_losses = [r["val_loss"] for r in results]

plt.figure(figsize=(9, 5))
plt.semilogx(
    C_values_array,
    val_losses,
    marker="o",
)

plt.xlabel("C (inverse regularization strength)")
plt.ylabel("Validation log loss")
plt.title("Regularization Strength vs. Validation Log Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% 9. Plot coefficient magnitude
coefficient_norms = [r["coefficient_norm"] for r in results]

plt.figure(figsize=(9, 5))
plt.semilogx(
    C_values_array,
    coefficient_norms,
    marker="o",
)

plt.xlabel("C (inverse regularization strength)")
plt.ylabel("L2 norm of coefficients")
plt.title("Regularization Strength vs. Coefficient Magnitude")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% 10. Select a model using validation log loss
best_result = min(results, key=lambda r: r["val_loss"])
best_C = best_result["C"]

print(f"\nSelected C: {best_C}")
print(f"Best validation log loss: {best_result['val_loss']:.3f}")

# %% 11. Evaluate the selected model on the test set
best_model = best_result["model"]

test_predictions = best_model.predict(X_test)
test_probabilities = best_model.predict_proba(X_test)

print(f"Test accuracy: {accuracy_score(y_test, test_predictions):.3f}")
print(f"Test log loss: {log_loss(y_test, test_probabilities):.3f}")
