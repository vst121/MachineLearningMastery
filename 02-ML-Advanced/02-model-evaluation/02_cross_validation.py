# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% 2. Load the dataset
data = load_breast_cancer()

X = data.data
y = data.target

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Class counts: {np.bincount(y)}")

# %% 3. Create a model pipeline
def create_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            random_state=42,
        ),
    )


# %% 4. Evaluate using K-Fold cross-validation
# Each fold is used once for validation.
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

kfold_scores = cross_val_score(
    create_model(),
    X,
    y,
    cv=kfold,
    scoring="accuracy",
    n_jobs=-1,
)

print("\nK-Fold cross-validation")
for fold_index, score in enumerate(kfold_scores, start=1):
    print(f"Fold {fold_index}: {score:.3f}")

print(f"Mean accuracy: {kfold_scores.mean():.3f}")
print(f"Standard deviation: {kfold_scores.std():.3f}")

# %% 5. Evaluate using Stratified K-Fold
# Stratification attempts to preserve class proportions in each fold.
stratified_kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

stratified_scores = cross_val_score(
    create_model(),
    X,
    y,
    cv=stratified_kfold,
    scoring="accuracy",
    n_jobs=-1,
)

print("\nStratified K-Fold cross-validation")
for fold_index, score in enumerate(stratified_scores, start=1):
    print(f"Fold {fold_index}: {score:.3f}")

print(f"Mean accuracy: {stratified_scores.mean():.3f}")
print(f"Standard deviation: {stratified_scores.std():.3f}")

# %% 6. Compare fold scores
fold_numbers = np.arange(1, len(kfold_scores) + 1)

plt.figure(figsize=(9, 5))

plt.plot(
    fold_numbers,
    kfold_scores,
    marker="o",
    label="K-Fold",
)

plt.plot(
    fold_numbers,
    stratified_scores,
    marker="s",
    label="Stratified K-Fold",
)

plt.axhline(
    kfold_scores.mean(),
    linestyle="--",
    label=f"K-Fold mean = {kfold_scores.mean():.3f}",
)

plt.axhline(
    stratified_scores.mean(),
    linestyle=":",
    label=f"Stratified mean = {stratified_scores.mean():.3f}",
)

plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("K-Fold vs. Stratified K-Fold")
plt.xticks(fold_numbers)
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% 7. Compare variability
print("\nCross-validation summary")
print(
    f"K-Fold:          mean={kfold_scores.mean():.3f}, "
    f"std={kfold_scores.std():.3f}"
)
print(
    f"Stratified K-Fold: mean={stratified_scores.mean():.3f}, "
    f"std={stratified_scores.std():.3f}"
)

# %% 8. Keep a final test set independent
# Cross-validation is performed only on the development data.
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

development_scores = cross_val_score(
    create_model(),
    X_dev,
    y_dev,
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    ),
    scoring="accuracy",
    n_jobs=-1,
)

print("\nDevelopment-set cross-validation")
print(f"Mean accuracy: {development_scores.mean():.3f}")
print(f"Standard deviation: {development_scores.std():.3f}")

# The test set remains untouched until the final evaluation.
final_model = create_model()
final_model.fit(X_dev, y_dev)

test_accuracy = final_model.score(X_test, y_test)

print(f"\nFinal test accuracy: {test_accuracy:.3f}")
