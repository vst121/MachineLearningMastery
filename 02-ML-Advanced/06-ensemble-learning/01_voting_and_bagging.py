# %% [markdown]
# # Voting and Bagging
#
# This script explores two ensemble learning strategies:
# 1. Voting: combine predictions from different model types.
# 2. Bagging: train copies of a model on different bootstrap samples.
#
# Dataset: Iris classification.
# Random state: 42 for reproducibility.

# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# %% [markdown]
# ## 1. Load and split the dataset

# %%
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}")

# %% [markdown]
# ## 2. Establish individual-model baselines
#
# Different algorithms may make different errors. An ensemble can benefit
# when its component models are reasonably accurate and their errors differ.

# %%
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
}

baseline_results = {}

for name, model in base_models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    baseline_results[name] = accuracy_score(y_test, predictions)

print("\nIndividual model accuracy:")
for name, accuracy in baseline_results.items():
    print(f"{name:22s}: {accuracy:.3f}")

# %% [markdown]
# ## 3. Hard voting
#
# Hard voting selects the class that receives the most votes.
# Each model contributes its predicted class label.

# %%
hard_voting = VotingClassifier(
    estimators=[
        ("logistic_regression", LogisticRegression(max_iter=1000)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        (
            "decision_tree",
            DecisionTreeClassifier(
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
    ],
    voting="hard",
)

hard_voting.fit(X_train, y_train)
hard_predictions = hard_voting.predict(X_test)
hard_accuracy = accuracy_score(y_test, hard_predictions)

print(f"\nHard voting accuracy: {hard_accuracy:.3f}")

# %% [markdown]
# ## 4. Soft voting
#
# Soft voting averages predicted class probabilities and selects the
# class with the highest average probability.
#
# Every component model must support predict_proba.

# %%
soft_voting = VotingClassifier(
    estimators=[
        ("logistic_regression", LogisticRegression(max_iter=1000)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        (
            "decision_tree",
            DecisionTreeClassifier(
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
    ],
    voting="soft",
)

soft_voting.fit(X_train, y_train)
soft_predictions = soft_voting.predict(X_test)
soft_accuracy = accuracy_score(y_test, soft_predictions)

print(f"Soft voting accuracy: {soft_accuracy:.3f}")

# %% [markdown]
# ## 5. Bagging with Decision Trees
#
# Bagging means Bootstrap Aggregating.
#
# Bootstrap sampling draws training rows with replacement. Some rows may
# appear multiple times, while others may not appear in a particular sample.
# Each tree learns from its own sample, and their predictions are aggregated.

# %%
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
    n_estimators=100,
    max_samples=1.0,
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

bagging.fit(X_train, y_train)
bagging_predictions = bagging.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_predictions)

print(f"Bagging accuracy: {bagging_accuracy:.3f}")
print(f"Number of estimators: {len(bagging.estimators_)}")

# %% [markdown]
# ## 6. Compare results
#
# These test results are for final evaluation, not hyperparameter tuning.

# %%
results = {
    **baseline_results,
    "Hard Voting": hard_accuracy,
    "Soft Voting": soft_accuracy,
    "Bagging (Trees)": bagging_accuracy,
}

print("\nModel comparison:")
for name, accuracy in sorted(
    results.items(),
    key=lambda item: item[1],
    reverse=True,
):
    print(f"{name:22s}: {accuracy:.3f}")

plt.figure(figsize=(10, 5))
names = list(results.keys())
scores = list(results.values())

bars = plt.bar(names, scores)
plt.ylabel("Test accuracy")
plt.ylim(0, 1.05)
plt.title("Individual Models vs. Voting and Bagging")
plt.xticks(rotation=25, ha="right")
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Inspect the Bagging model
#
# Compare one Decision Tree with the Bagging ensemble.

# %%
single_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
single_tree.fit(X_train, y_train)
single_tree_predictions = single_tree.predict(X_test)

print("\nSingle Decision Tree:")
print(f"Accuracy: {accuracy_score(y_test, single_tree_predictions):.3f}")

print("\nBagging classification report:")
print(
    classification_report(
        y_test,
        bagging_predictions,
        target_names=iris.target_names,
        zero_division=0,
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    bagging_predictions,
    display_labels=iris.target_names,
    cmap="Blues",
)
plt.title("Bagging Classifier: Confusion Matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Inspect diversity among the bagged trees
#
# This measures how often each tree disagrees with the ensemble's final
# prediction. It is a simple diagnostic, not a complete diversity measure.

# %%
individual_tree_predictions = np.array(
    [tree.predict(X_test) for tree in bagging.estimators_]
)

disagreement_rates = np.mean(
    individual_tree_predictions != bagging_predictions[np.newaxis, :],
    axis=1,
)

print("\nIndividual tree disagreement with ensemble prediction:")
print(f"Mean disagreement: {disagreement_rates.mean():.3f}")
print(f"Minimum disagreement: {disagreement_rates.min():.3f}")
print(f"Maximum disagreement: {disagreement_rates.max():.3f}")

# %% [markdown]
# ## Key Takeaways
#
# - Hard voting combines predicted labels.
# - Soft voting combines predicted probabilities.
# - Bagging trains multiple estimators on bootstrap samples.
# - Bagging often reduces the variance of unstable models such as Decision Trees.
# - Ensembles do not always outperform individual models.
# - Keep the test set out of model selection and hyperparameter tuning.
