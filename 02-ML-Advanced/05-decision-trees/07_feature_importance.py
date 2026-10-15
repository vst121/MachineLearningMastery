# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# %% 2. Load the Dataset
cancer = load_breast_cancer()

X = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names,
)

y = pd.Series(
    cancer.target,
    name="target",
)

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])

print("\nTarget classes:")
print(dict(enumerate(cancer.target_names)))


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


# %% 4. Train a Decision Tree
model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=5,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nTest Accuracy:", round(
    accuracy_score(y_test, y_pred),
    4,
))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=cancer.target_names,
        zero_division=0,
    )
)


# %% 5. Calculate Built-in Feature Importance
impurity_importance = pd.Series(
    model.feature_importances_,
    index=X.columns,
    name="Impurity Importance",
).sort_values(ascending=False)

print("\nTop 15 Features by Impurity Importance:")
print(
    impurity_importance.head(15).to_string(
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nSum of Importances:", round(
    impurity_importance.sum(),
    4,
))


# %% 6. Plot Built-in Feature Importance
top_impurity = impurity_importance.head(15).sort_values()

top_impurity.plot(
    kind="barh",
    figsize=(10, 7),
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features: Impurity-Based Importance")
plt.tight_layout()
plt.show()


# %% 7. Calculate Permutation Importance
# The model remains fixed. Each feature is shuffled on the test set,
# and the decrease in accuracy is measured.
permutation_result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=20,
    random_state=42,
    n_jobs=-1,
)

permutation_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean Importance": permutation_result.importances_mean,
    "Std Importance": permutation_result.importances_std,
})

permutation_importance_df = permutation_importance_df.sort_values(
    "Mean Importance",
    ascending=False,
)

print("\nTop 15 Features by Permutation Importance:")
print(
    permutation_importance_df.head(15).to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 8. Plot Permutation Importance
top_permutation = permutation_importance_df.head(15).sort_values(
    "Mean Importance",
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_permutation["Feature"],
    top_permutation["Mean Importance"],
    xerr=top_permutation["Std Importance"],
    capsize=3,
)

plt.axvline(0, linestyle="--")

plt.xlabel("Mean Decrease in Test Accuracy")
plt.ylabel("Feature")
plt.title("Top 15 Features: Permutation Importance")
plt.tight_layout()
plt.show()


# %% 9. Compare Both Importance Methods
comparison_df = pd.DataFrame({
    "Impurity Importance": impurity_importance,
    "Permutation Importance": permutation_importance_df.set_index(
        "Feature"
    )["Mean Importance"],
})

comparison_df["Impurity Rank"] = comparison_df[
    "Impurity Importance"
].rank(ascending=False, method="min")

comparison_df["Permutation Rank"] = comparison_df[
    "Permutation Importance"
].rank(ascending=False, method="min")

comparison_df["Rank Difference"] = (
    comparison_df["Impurity Rank"]
    - comparison_df["Permutation Rank"]
).abs()

comparison_df = comparison_df.sort_values(
    "Permutation Importance",
    ascending=False,
)

print("\nImportance Method Comparison:")
print(
    comparison_df.head(20).to_string(
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nSpearman Rank Correlation:")
print(
    round(
        comparison_df["Impurity Rank"].corr(
            comparison_df["Permutation Rank"],
            method="spearman",
        ),
        4,
    )
)


# %% 10. Inspect Predictions for a Few Samples
sample_results = X_test.iloc[:5].copy()

sample_results["Actual Class"] = [
    cancer.target_names[index]
    for index in y_test.iloc[:5]
]

sample_results["Predicted Class"] = [
    cancer.target_names[index]
    for index in y_pred[:5]
]

print("\nSample Predictions:")
print(sample_results[
    ["Actual Class", "Predicted Class"]
].to_string())
