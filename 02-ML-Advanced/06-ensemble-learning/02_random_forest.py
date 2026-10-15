
# %% [markdown]
# # Random Forest
#
# Topics:
# 1. Random Forest classification
# 2. Bootstrap sampling and random feature selection
# 3. Out-of-bag (OOB) evaluation
# 4. Impurity-based and permutation feature importance
# 5. Random Forest regression
# 6. A brief comparison with Extra Trees
#
# Plots are displayed interactively and are not saved to files.

# %%
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# %% [markdown]
# ## 1. Random Forest classification
#
# A Random Forest trains many Decision Trees.
#
# Each tree is trained using a bootstrap sample of the training data.
# At each split, the tree considers a random subset of features.
# Classification predictions are aggregated by voting.

# %%
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y,
)

single_tree = DecisionTreeClassifier(
    random_state=RANDOM_STATE,
)

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    min_samples_leaf=1,
    bootstrap=True,
    oob_score=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

single_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

tree_predictions = single_tree.predict(X_test)
forest_predictions = random_forest.predict(X_test)

print("Classification results")
print(
    f"Single Decision Tree accuracy: "
    f"{accuracy_score(y_test, tree_predictions):.3f}"
)
print(
    f"Random Forest accuracy:        "
    f"{accuracy_score(y_test, forest_predictions):.3f}"
)
print(
    f"Random Forest OOB score:        "
    f"{random_forest.oob_score_:.3f}"
)
print(f"Number of trees:                {len(random_forest.estimators_)}")

# %% [markdown]
# ## 2. Understand the main hyperparameters
#
# - n_estimators: number of trees.
# - max_features: features considered at each split.
# - max_depth: maximum tree depth.
# - min_samples_leaf: minimum samples required in a leaf.
# - bootstrap: whether each tree uses bootstrap sampling.
# - oob_score: whether out-of-bag samples are used for evaluation.
#
# Increasing the number of trees often stabilizes predictions, but costs
# more computation. More trees do not automatically eliminate overfitting.

# %%
print("\nRandom Forest configuration")
print(f"n_estimators: {random_forest.n_estimators}")
print(f"max_features: {random_forest.max_features}")
print(f"max_depth: {random_forest.max_depth}")
print(f"min_samples_leaf: {random_forest.min_samples_leaf}")
print(f"bootstrap: {random_forest.bootstrap}")

# %% [markdown]
# ## 3. Out-of-bag evaluation
#
# Bootstrap sampling draws training rows with replacement. Some rows may
# appear multiple times, while others are omitted from a given tree's sample.
#
# Those omitted rows are called out-of-bag (OOB) samples. Predictions from
# trees that did not train on a particular row provide an internal estimate
# of generalization performance.
#
# OOB evaluation complements, but does not replace, a held-out test set.

# %%
print(f"\nOOB classification accuracy: {random_forest.oob_score_:.3f}")
print(
    "OOB predictions available: "
    f"{hasattr(random_forest, 'oob_decision_function_')}"
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    forest_predictions,
    display_labels=cancer.target_names,
    cmap="Blues",
)
plt.title("Random Forest: Test Confusion Matrix")
plt.tight_layout()
plt.show()
plt.close("all")

# %% [markdown]
# ## 4. Impurity-based feature importance
#
# Impurity-based importance measures how much a feature contributes to
# impurity reduction across the trees.
#
# Limitation: it can favor high-cardinality features and does not establish
# that a feature causes the target outcome.

# %%
impurity_importance = random_forest.feature_importances_
top_indices = np.argsort(impurity_importance)[-10:][::-1]

print("\nTop 10 features by impurity-based importance:")
for index in top_indices:
    print(
        f"{cancer.feature_names[index]:30s}: "
        f"{impurity_importance[index]:.4f}"
    )

plt.figure(figsize=(10, 6))
plt.barh(
    cancer.feature_names[top_indices][::-1],
    impurity_importance[top_indices][::-1],
)
plt.xlabel("Impurity-based importance")
plt.title("Random Forest: Top 10 Features")
plt.tight_layout()
plt.show()
plt.close("all")

# %% [markdown]
# ## 5. Permutation importance
#
# Permutation importance shuffles one feature at a time and measures how
# much model performance decreases.
#
# Here it is calculated on the test set for illustration only. Do not use
# these test-set results to select or tune the model.
#
# A feature can have near-zero permutation importance if it contributes
# little additional predictive information or if correlated features
# provide similar information.

# %%
permutation = permutation_importance(
    random_forest,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

permutation_indices = np.argsort(
    permutation.importances_mean
)[-10:][::-1]

print("\nTop 10 features by permutation importance:")
for index in permutation_indices:
    mean = permutation.importances_mean[index]
    std = permutation.importances_std[index]

    print(
        f"{cancer.feature_names[index]:30s}: "
        f"{mean:.4f} +/- {std:.4f}"
    )

plt.figure(figsize=(10, 6))
plt.barh(
    cancer.feature_names[permutation_indices][::-1],
    permutation.importances_mean[permutation_indices][::-1],
    xerr=permutation.importances_std[permutation_indices][::-1],
)
plt.xlabel("Mean decrease in accuracy")
plt.title("Permutation Importance on Test Data")
plt.tight_layout()
plt.show()
plt.close("all")

# %% [markdown]
# ## 6. Random Forest regression
#
# For regression, a forest typically averages the numerical predictions
# from its trees.
#
# The Diabetes dataset is included with scikit-learn, so no download is
# required.

# %%
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.25,
    random_state=RANDOM_STATE,
)

forest_regressor = RandomForestRegressor(
    n_estimators=200,
    max_features=1.0,
    min_samples_leaf=2,
    bootstrap=True,
    oob_score=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

forest_regressor.fit(X_train_reg, y_train_reg)
regression_predictions = forest_regressor.predict(X_test_reg)

mae = mean_absolute_error(
    y_test_reg,
    regression_predictions,
)
rmse = np.sqrt(
    mean_squared_error(y_test_reg, regression_predictions)
)
r2 = r2_score(
    y_test_reg,
    regression_predictions,
)

print("\nRandom Forest regression results")
print(f"MAE:    {mae:.2f}")
print(f"RMSE:   {rmse:.2f}")
print(f"R²:     {r2:.3f}")
print(f"OOB R²: {forest_regressor.oob_score_:.3f}")

# %%
plt.figure(figsize=(7, 6))
plt.scatter(
    y_test_reg,
    regression_predictions,
    alpha=0.7,
)

lower = min(
    y_test_reg.min(),
    regression_predictions.min(),
)
upper = max(
    y_test_reg.max(),
    regression_predictions.max(),
)

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
)
plt.xlabel("Actual target")
plt.ylabel("Predicted target")
plt.title("Random Forest Regression: Actual vs Predicted")
plt.tight_layout()
plt.show()
plt.close("all")

# %% [markdown]
# ## 7. Extra Trees comparison
#
# Extra Trees (Extremely Randomized Trees) introduce additional randomness
# in split selection, including randomized split thresholds.
#
# Random Forests generally search for a good split among randomly selected
# features. Both methods aggregate many trees, and neither is guaranteed
# to outperform the other on every dataset.

# %%
extra_trees = ExtraTreesClassifier(
    n_estimators=200,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

extra_trees.fit(X_train, y_train)
extra_predictions = extra_trees.predict(X_test)

comparison_names = [
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
]
comparison_scores = [
    accuracy_score(y_test, tree_predictions),
    accuracy_score(y_test, forest_predictions),
    accuracy_score(y_test, extra_predictions),
]

print("\nClassifier comparison")
for name, score in zip(comparison_names, comparison_scores):
    print(f"{name:22s}: {score:.3f}")

# %%
plt.figure(figsize=(8, 5))
bars = plt.bar(comparison_names, comparison_scores)
plt.ylabel("Test accuracy")
plt.ylim(0, 1.05)
plt.title("Decision Tree vs Random Forest vs Extra Trees")
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()
plt.close("all")

# %% [markdown]
# ## Key Takeaways
#
# - Random Forest combines bootstrap sampling and random feature selection.
# - Aggregating trees often reduces variance compared with a single tree.
# - OOB evaluation uses training rows omitted from individual bootstrap samples.
# - Impurity and permutation importance measure different aspects of feature relevance.
# - Random Forest supports classification and regression.
# - Extra Trees introduce additional randomness in split selection.
# - Use cross-validation for model selection and keep the test set for final evaluation.

# %%
print("\nRandom Forest script completed.")
