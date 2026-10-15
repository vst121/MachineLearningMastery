# %% [markdown]
# # Boosting: AdaBoost and Gradient Boosting
#
# Topics:
# 1. Understand sequential ensemble learning.
# 2. Train AdaBoost classifiers.
# 3. Train Gradient Boosting classifiers.
# 4. Compare Bagging and Boosting conceptually.
# 5. Train Gradient Boosting regressors.
# 6. Explore learning rate and the number of boosting stages.
#
# Datasets: Breast Cancer and Diabetes from scikit-learn.

# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RANDOM_STATE = 42

# %% [markdown]
# ## 1. Load the classification dataset
#
# We will compare a single Decision Tree with two boosting algorithms.
# The test set is held out for final evaluation.

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

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {list(cancer.target_names)}")

# %% [markdown]
# ## 2. Establish a Decision Tree baseline
#
# A single tree learns its own decision rules. Boosting combines multiple
# weak or relatively simple learners into a stronger ensemble.

# %%
baseline_tree = DecisionTreeClassifier(
    max_depth=2,
    random_state=RANDOM_STATE,
)

baseline_tree.fit(X_train, y_train)
baseline_predictions = baseline_tree.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print("\nDecision Tree baseline")
print(f"Test accuracy: {baseline_accuracy:.3f}")

# %% [markdown]
# ## 3. AdaBoost classification
#
# AdaBoost means Adaptive Boosting.
#
# Conceptually:
# 1. Train a weak learner on the training data.
# 2. Give more influence to examples that were misclassified.
# 3. Train another learner using the updated example weights.
# 4. Combine learners using weighted voting.
#
# Modern scikit-learn versions use estimator= to specify the base learner.

# %%
adaboost_classifier = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=RANDOM_STATE,
    ),
    n_estimators=100,
    learning_rate=0.5,
    random_state=RANDOM_STATE,
)

adaboost_classifier.fit(X_train, y_train)
ada_predictions = adaboost_classifier.predict(X_test)

ada_accuracy = accuracy_score(y_test, ada_predictions)

print("\nAdaBoost classification")
print(f"Test accuracy: {ada_accuracy:.3f}")
print(f"Number of boosting stages: {len(adaboost_classifier.estimators_)}")

# %% [markdown]
# ## 4. Gradient Boosting classification
#
# Gradient Boosting builds an additive model in stages.
#
# Each new tree helps reduce the current ensemble's loss. For common
# regression losses, this can be understood through residual-like signals.
# For classification, the trees instead follow gradients of the chosen loss.
#
# learning_rate controls how strongly each new tree contributes.
# n_estimators controls the maximum number of boosting stages.

# %%
gradient_classifier = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=2,
    random_state=RANDOM_STATE,
)

gradient_classifier.fit(X_train, y_train)
gradient_predictions = gradient_classifier.predict(X_test)

gradient_accuracy = accuracy_score(y_test, gradient_predictions)

print("\nGradient Boosting classification")
print(f"Test accuracy: {gradient_accuracy:.3f}")
print(f"Number of boosting stages: {len(gradient_classifier.estimators_)}")

# %% [markdown]
# ## 5. Compare classification models
#
# All models use the same train-test split.
# Test results are for evaluation, not for tuning the models.

# %%
classification_results = {
    "Decision Tree": baseline_accuracy,
    "AdaBoost": ada_accuracy,
    "Gradient Boosting": gradient_accuracy,
}

print("\nClassification comparison")
for name, score in sorted(
    classification_results.items(),
    key=lambda item: item[1],
    reverse=True,
):
    print(f"{name:22s}: {score:.3f}")

# %%
plt.figure(figsize=(9, 5))
bars = plt.bar(
    list(classification_results.keys()),
    list(classification_results.values()),
)
plt.ylabel("Test accuracy")
plt.ylim(0, 1.05)
plt.title("Decision Tree vs. Boosting Classifiers")
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.tight_layout()
plt.show()
plt.close("all")

# %% [markdown]
# ## 6. Inspect Gradient Boosting classification errors
#
# The built-in Breast Cancer target uses:
# 0 = malignant
# 1 = benign
#
# Accuracy alone may hide important class-specific errors. In medical
# applications, evaluate the consequences of false negatives and false
# positives, and choose metrics that match the intended use.

# %%
print("\nGradient Boosting classification report")
print(
    classification_report(
        y_test,
        gradient_predictions,
        target_names=cancer.target_names,
        zero_division=0,
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    gradient_predictions,
    display_labels=cancer.target_names,
    cmap="Blues",
)
plt.title("Gradient Boosting: Confusion Matrix")
plt.tight_layout()
plt.show()
plt.close("all")

# %% [markdown]
# ## 7. Gradient Boosting regression
#
# For regression, Gradient Boosting successively improves a numerical
# prediction by adding trees that reduce the selected loss.
#
# We use the built-in Diabetes dataset and compare against a single
# Decision Tree regressor.

# %%
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.25,
    random_state=RANDOM_STATE,
)

tree_regressor = DecisionTreeRegressor(
    max_depth=3,
    random_state=RANDOM_STATE,
)

gradient_regressor = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    loss="squared_error",
    random_state=RANDOM_STATE,
)

ada_regressor = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
    n_estimators=100,
    learning_rate=0.05,
    loss="linear",
    random_state=RANDOM_STATE,
)

tree_regressor.fit(X_train_reg, y_train_reg)
gradient_regressor.fit(X_train_reg, y_train_reg)
ada_regressor.fit(X_train_reg, y_train_reg)

tree_reg_predictions = tree_regressor.predict(X_test_reg)
gradient_reg_predictions = gradient_regressor.predict(X_test_reg)
ada_reg_predictions = ada_regressor.predict(X_test_reg)

# %% [markdown]
# ## 8. Evaluate regression models
#
# MAE: average absolute prediction error.
# RMSE: penalizes larger errors more heavily.
# R²: compares the model with a baseline that predicts the test-target mean.
#
# Higher R² is generally better. Lower MAE and RMSE are generally better.

# %%
regression_predictions = {
    "Decision Tree": tree_reg_predictions,
    "Gradient Boosting": gradient_reg_predictions,
    "AdaBoost": ada_reg_predictions,
}

regression_results = {}

print("\nRegression comparison")

for name, predictions in regression_predictions.items():
    mae = mean_absolute_error(y_test_reg, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_reg, predictions))
    r2 = r2_score(y_test_reg, predictions)

    regression_results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")

# %% [markdown]
# ## 9. Visualize regression predictions
#
# Points closer to the diagonal line indicate more accurate predictions.
# This plot helps reveal systematic errors and unusually large deviations.

# %%
plt.figure(figsize=(8, 6))

for name, predictions in regression_predictions.items():
    plt.scatter(
        y_test_reg,
        predictions,
        alpha=0.55,
        label=name,
    )

lower = min(
    y_test_reg.min(),
    min(predictions.min() for predictions in regression_predictions.values()),
)
upper = max(
    y_test_reg.max(),
    max(predictions.max() for predictions in regression_predictions.values()),
)

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
    color="black",
    label="Perfect prediction",
)

plt.xlabel("Actual target")
plt.ylabel("Predicted target")
plt.title("Regression Model Predictions")
plt.legend()
plt.tight_layout()
plt.show()
plt.close("all")

# %% [markdown]
# ## 10. Explore learning rate and number of stages
#
# A smaller learning rate reduces each tree's contribution. It often
# requires more boosting stages to achieve comparable performance.
#
# We compare several configurations using a validation split taken only
# from the training data. The test set remains untouched for final evaluation.

# %%
X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(
    X_train_reg,
    y_train_reg,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

learning_rates = [0.01, 0.05, 0.1]
n_estimators_values = [50, 100, 200]

validation_results = []

for learning_rate in learning_rates:
    for n_estimators in n_estimators_values:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=2,
            random_state=RANDOM_STATE,
        )

        model.fit(X_subtrain, y_subtrain)
        validation_predictions = model.predict(X_validation)

        validation_rmse = np.sqrt(
            mean_squared_error(
                y_validation,
                validation_predictions,
            )
        )

        validation_results.append(
            (
                learning_rate,
                n_estimators,
                validation_rmse,
            )
        )

print("\nValidation results")
print("Learning rate | Estimators | Validation RMSE")

for learning_rate, n_estimators, rmse in validation_results:
    print(f"{learning_rate:13.2f} | {n_estimators:10d} | {rmse:16.2f}")

best_configuration = min(
    validation_results,
    key=lambda result: result[2],
)

print("\nBest validation configuration")
print(f"Learning rate: {best_configuration[0]}")
print(f"Estimators:    {best_configuration[1]}")
print(f"Validation RMSE: {best_configuration[2]:.2f}")

# %% [markdown]
# ## 11. Bagging vs. Boosting
#
# Bagging:
# - Trains models independently, often on bootstrap samples.
# - Aggregates their predictions.
# - Often reduces variance.
#
# Boosting:
# - Adds learners sequentially.
# - Each stage improves the current ensemble according to its algorithm.
# - Can reduce bias, but may overfit if configured poorly.
#
# Random Forest is a Bagging-based ensemble of randomized trees.
# AdaBoost and Gradient Boosting are boosting algorithms.

# %%
print("\nConceptual comparison")
print("Bagging: independent learners, commonly bootstrap sampling.")
print("Boosting: sequential learners that improve the current ensemble.")
print("Random Forest: Bagging plus random feature selection.")
print("AdaBoost: adapts example weights after weak-learner errors.")
print("Gradient Boosting: adds learners guided by loss gradients.")

# %% [markdown]
# ## Key Takeaways
#
# - AdaBoost and Gradient Boosting build ensembles sequentially.
# - AdaBoost adjusts the influence of training examples.
# - Gradient Boosting adds trees that reduce a loss function.
# - learning_rate and n_estimators should be considered together.
# - Boosting can perform very well on tabular data, but it is not guaranteed
#   to outperform every baseline.
# - Use validation data or cross-validation for model selection.
# - Preserve the test set for final evaluation.
