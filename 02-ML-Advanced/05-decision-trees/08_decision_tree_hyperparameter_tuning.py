# %% 1. Imports
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
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

print("\nClass distribution:")
print(y.value_counts().rename(index={
    0: "malignant",
    1: "benign",
}))


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


# %% 4. Configure Cross-Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# %% 5. Define the Hyperparameter Grid
# Keep the grid reasonably small so the experiment runs efficiently.
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "ccp_alpha": [0.0, 0.001, 0.01],
}

total_configurations = 1

for values in param_grid.values():
    total_configurations *= len(values)

print("\nHyperparameter configurations:", total_configurations)
print("Cross-validation folds:", cv.get_n_splits())


# %% 6. Run Grid Search
base_model = DecisionTreeClassifier(
    random_state=42,
)

search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest Parameters:")
print(search.best_params_)

print("\nBest Mean CV F1:", round(search.best_score_, 4))


# %% 7. Inspect the Top Configurations
cv_results = pd.DataFrame(search.cv_results_)

top_results = cv_results[
    [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "param_criterion",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "param_ccp_alpha",
    ]
].sort_values("rank_test_score").head(15)

print("\nTop 15 Configurations:")
print(
    top_results.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# %% 8. Compare Training and Cross-Validation Scores
print("\nSelected Model Complexity:")
print("Tree depth:", best_model.get_depth())
print("Number of leaves:", best_model.get_n_leaves())
print("Number of nodes:", best_model.tree_.node_count)

print("\nSelected Model Scores:")
print("Mean training F1:", round(
    search.cv_results_["mean_train_score"][search.best_index_],
    4,
))
print("Mean validation F1:", round(search.best_score_, 4))


# %% 9. Evaluate on the Held-Out Test Set
test_predictions = best_model.predict(X_test)
test_probabilities = best_model.predict_proba(X_test)[:, 1]

print("\nFinal Test Results")
print("Accuracy:", round(
    accuracy_score(y_test, test_predictions),
    4,
))
print("F1:", round(
    f1_score(y_test, test_predictions),
    4,
))
print("ROC-AUC:", round(
    roc_auc_score(y_test, test_probabilities),
    4,
))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=cancer.target_names,
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print(
    pd.DataFrame(
        confusion_matrix(y_test, test_predictions),
        index=[
            "Actual malignant",
            "Actual benign",
        ],
        columns=[
            "Predicted malignant",
            "Predicted benign",
        ],
    )
)


# %% 10. Visualize the Selected Tree
from sklearn.tree import plot_tree

plt.figure(figsize=(22, 10))

plot_tree(
    best_model,
    feature_names=X.columns,
    class_names=cancer.target_names,
    filled=True,
    rounded=True,
    fontsize=7,
)

plt.title("Best Decision Tree Selected by Cross-Validation")
plt.tight_layout()
plt.show()


# %% 11. Visualize the Most Important Features
feature_importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

top_features = feature_importances.head(15).sort_values()

top_features.plot(
    kind="barh",
    figsize=(10, 7),
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Features in the Selected Decision Tree")
plt.tight_layout()
plt.show()


# %% 12. Summarize the Workflow
print("\nModel Selection Workflow")
print("1. Split the dataset into training and test sets.")
print("2. Define candidate hyperparameter combinations.")
print("3. Evaluate candidates using stratified cross-validation.")
print("4. Select the configuration with the highest mean CV F1.")
print("5. Refit the selected model on the full training set.")
print("6. Evaluate the final model on the held-out test set.")
