# %% [markdown]
# # Stacking
#
# Stacking combines multiple base models using a meta-model.
#
# We will:
# 1. Train multiple base classifiers.
# 2. Combine them using StackingClassifier.
# 3. Compare stacking with individual models and voting.
# 4. Evaluate the models using accuracy and F1-score.
# 5. Visualize the model comparison.
#
# Dataset: Breast Cancer Wisconsin Classification

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# %% [markdown]
# ## 1. Load and Split the Dataset

# %%
data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")


# %% [markdown]
# ## 2. Define the Base Models
#
# Scaling is important for Logistic Regression, SVM, and KNN.
# Random Forest does not require feature scaling.
#
# CalibratedClassifierCV provides probability estimates for SVC
# without using the deprecated SVC(probability=True) parameter.

# %%
base_models = [
    (
        "logistic_regression",
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ),
    (
        "svm",
        make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                SVC(random_state=42),
                ensemble=False,
            ),
        ),
    ),
    (
        "knn",
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
    ),
    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    ),
]


# %% [markdown]
# ## 3. Train and Evaluate Individual Models

# %%
results = {}

for name, model in base_models:
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print(f"\n{name}")
    print(f"Accuracy: {results[name]['accuracy']:.4f}")
    print(f"F1-score: {results[name]['f1_score']:.4f}")


# %% [markdown]
# ## 4. Build the Stacking Classifier
#
# The base models generate predictions.
# Logistic Regression learns how to combine those predictions.
#
# StackingClassifier uses cross-validation to generate out-of-fold
# predictions for training the final estimator.
#
# This helps prevent the meta-model from learning from in-sample
# predictions that are overly optimistic.

# %%
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(
        max_iter=2000,
    ),
    stack_method="auto",
    cv=5,
    n_jobs=-1,
)

stacking_model.fit(X_train, y_train)

stacking_pred = stacking_model.predict(X_test)

results["stacking"] = {
    "accuracy": accuracy_score(y_test, stacking_pred),
    "f1_score": f1_score(y_test, stacking_pred),
}

print("\nStacking Classifier")
print(f"Accuracy: {results['stacking']['accuracy']:.4f}")
print(f"F1-score: {results['stacking']['f1_score']:.4f}")


# %% [markdown]
# ## 5. Build a Voting Classifier
#
# Soft voting averages the predicted class probabilities.
# Unlike stacking, it does not train a meta-model to learn the
# combination of base-model predictions.

# %%
voting_model = VotingClassifier(
    estimators=base_models,
    voting="soft",
    n_jobs=-1,
)

voting_model.fit(X_train, y_train)

voting_pred = voting_model.predict(X_test)

results["soft_voting"] = {
    "accuracy": accuracy_score(y_test, voting_pred),
    "f1_score": f1_score(y_test, voting_pred),
}

print("\nSoft Voting Classifier")
print(f"Accuracy: {results['soft_voting']['accuracy']:.4f}")
print(f"F1-score: {results['soft_voting']['f1_score']:.4f}")


# %% [markdown]
# ## 6. Compare Model Performance

# %%
model_names = list(results.keys())

accuracies = [
    results[name]["accuracy"]
    for name in model_names
]

f1_scores = [
    results[name]["f1_score"]
    for name in model_names
]

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(
    x - width / 2,
    accuracies,
    width,
    label="Accuracy",
)

plt.bar(
    x + width / 2,
    f1_scores,
    width,
    label="F1-score",
)

plt.xticks(
    x,
    model_names,
    rotation=25,
    ha="right",
)

plt.ylabel("Score")
plt.title("Individual Models vs Stacking vs Soft Voting")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Inspect the Stacking Architecture
#
# The fitted StackingClassifier contains the trained base estimators
# and the final estimator.

# %%
print("Number of fitted base estimators:")
print(len(stacking_model.estimators_))

print("\nFinal estimator:")
print(stacking_model.final_estimator_)

print("\nStacking method:")
print(stacking_model.stack_method_)


# %% [markdown]
# ## 8. Key Takeaways
#
# - Stacking combines different models using a learned meta-model.
# - Base models should ideally make complementary errors.
# - Cross-validation generates out-of-fold predictions for training
#   the meta-model.
# - The final estimator learns from base-model predictions.
# - Soft voting averages predicted class probabilities.
# - CalibratedClassifierCV provides probability estimates for SVC.
# - Stacking is more flexible than voting but can be more expensive.
# - Stacking does not guarantee better generalization.
# - Compare models on unseen data and use cross-validation for
#   more reliable performance estimates.
