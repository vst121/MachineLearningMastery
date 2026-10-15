# %% [markdown]
# # Pipeline Hyperparameter Tuning
#
# Tune preprocessing and model hyperparameters together.
#
# Workflow:
# 1. Create a mixed-type classification dataset.
# 2. Split the data into training and test sets.
# 3. Build numerical and categorical preprocessing pipelines.
# 4. Combine preprocessing and classification in one pipeline.
# 5. Tune preprocessing and classifier parameters with GridSearchCV.
# 6. Inspect the best configuration and evaluate on the test set.
#
# All learned preprocessing is fitted inside each CV training fold.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# %% [markdown]
# ## 1. Create a mixed-type dataset
#
# This synthetic dataset contains numerical and categorical features,
# including missing values. The target represents subscription
# renewal.
#
# The example is intended to demonstrate the workflow, not to
# represent a real business prediction problem.

# %%
data = pd.DataFrame(
    {
        "age": [
            24, 35, 42, 51, 29, 46, 33, 58, 39, 27,
            44, 31, 62, 37, 49, 26, 54, 41, 32, 47,
            28, 56, 36, 43, 30, 52, 40, 34, 60, 45,
            25, 48, 38, 55, 31, 42, 50, 29, 57, 35,
        ],
        "income": [
            32000, 55000, 61000, 78000, 41000, 72000, 48000, 90000,
            59000, 36000, 68000, 45000, 95000, 53000, 81000, 34000,
            85000, 63000, 47000, 76000, 39000, 88000, 52000, 66000,
            43000, 82000, 60000, 49000, 92000, 71000, 33000, 79000,
            56000, 87000, 46000, 64000, 80000, 40000, 91000, 54000,
        ],
        "city": [
            "Berlin", "Mannheim", "Berlin", "Hamburg", "Cologne",
            "Mannheim", "Cologne", "Berlin", "Hamburg", "Cologne",
            "Berlin", "Mannheim", "Hamburg", "Cologne", "Berlin",
            "Mannheim", "Hamburg", "Berlin", "Cologne", "Mannheim",
            "Berlin", "Hamburg", "Cologne", "Mannheim", "Berlin",
            "Hamburg", "Cologne", "Mannheim", "Berlin", "Hamburg",
            "Cologne", "Berlin", "Mannheim", "Hamburg", "Cologne",
            "Berlin", "Mannheim", "Hamburg", "Cologne", "Berlin",
        ],
        "membership": [
            "Basic", "Premium", "Standard", "Premium", "Basic",
            "Premium", "Standard", "Premium", "Standard", "Basic",
            "Premium", "Basic", "Premium", "Standard", "Premium",
            "Basic", "Premium", "Standard", "Basic", "Premium",
            "Basic", "Premium", "Standard", "Premium", "Basic",
            "Premium", "Standard", "Basic", "Premium", "Standard",
            "Basic", "Premium", "Standard", "Premium", "Basic",
            "Standard", "Premium", "Standard", "Premium", "Basic",
        ],
        "renewed": [
            0, 1, 1, 1, 0, 1, 0, 1, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 1, 0, 1, 1,
            0, 1, 1, 1, 0, 1, 1, 0, 1, 0,
        ],
    }
)

# Introduce missing values for preprocessing demonstrations.
data.loc[[2, 9, 18], "age"] = float("nan")
data.loc[[5, 14, 27], "income"] = float("nan")
data.loc[[3, 11, 24], "city"] = None
data.loc[[7, 16, 30], "membership"] = None

X = data.drop(columns="renewed")
y = data["renewed"]

numerical_features = ["age", "income"]
categorical_features = ["city", "membership"]

print("Dataset shape:", data.shape)
print("\nMissing values:")
print(X.isna().sum())
print("\nTarget distribution:")
print(y.value_counts().sort_index())


# %% [markdown]
# ## 2. Split the dataset
#
# Split before fitting any learned preprocessing or model-selection
# components. Stratification preserves the target proportions.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42,
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))


# %% [markdown]
# ## 3. Build preprocessing pipelines
#
# Numerical features:
# - Median imputation
# - Standardization
#
# Categorical features:
# - Most-frequent imputation
# - One-hot encoding
#
# handle_unknown="ignore" allows the pipeline to handle categories
# appearing at prediction time that were not observed during fitting.

# %%
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
)


# %% [markdown]
# ## 4. Combine preprocessing and classification
#
# GridSearchCV can tune parameters belonging to any named pipeline
# step. Use double underscores to address nested parameters.

# %%
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)

param_grid = {
    "preprocessor__num__imputer__strategy": [
        "mean",
        "median",
    ],
    "preprocessor__cat__encoder__min_frequency": [
        None,
        2,
    ],
    "classifier__C": [
        0.1,
        1.0,
        10.0,
    ],
    "classifier__class_weight": [
        None,
        "balanced",
    ],
}

print("Number of candidate configurations:", 2 * 2 * 3 * 2)


# %% [markdown]
# ## 5. Tune the complete pipeline
#
# The pipeline is cloned and fitted independently within each
# cross-validation training fold. Imputation statistics, scaling
# parameters, and encoding categories are learned only from that fold.

# %%
cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:")
for parameter, value in grid_search.best_params_.items():
    print(f"  {parameter}: {value}")

print(
    "\nBest mean cross-validation ROC AUC:",
    f"{grid_search.best_score_:.4f}",
)


# %% [markdown]
# ## 6. Inspect cross-validation results
#
# Compare the leading configurations and examine the gap between
# training and validation scores.

# %%
results = pd.DataFrame(grid_search.cv_results_)

results = results[
    [
        "param_preprocessor__num__imputer__strategy",
        "param_preprocessor__cat__encoder__min_frequency",
        "param_classifier__C",
        "param_classifier__class_weight",
        "mean_train_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
].sort_values("rank_test_score")

results = results.rename(
    columns={
        "param_preprocessor__num__imputer__strategy": "imputation",
        "param_preprocessor__cat__encoder__min_frequency": "min_frequency",
        "param_classifier__C": "C",
        "param_classifier__class_weight": "class_weight",
        "mean_train_score": "mean_train_roc_auc",
        "mean_test_score": "mean_cv_roc_auc",
        "std_test_score": "std_cv_roc_auc",
        "rank_test_score": "rank",
    }
)

print("\nTop configurations:")
print(results.head(10).to_string(index=False))


# %% [markdown]
# ## 7. Evaluate on the held-out test set
#
# refit=True means GridSearchCV has already refitted the best
# pipeline on the complete training set.
#
# The test set was not used for preprocessing or model selection.

# %%
best_model = grid_search.best_estimator_

test_predictions = best_model.predict(X_test)
test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, test_predictions)
test_roc_auc = roc_auc_score(y_test, test_probabilities)

print("\nFinal test results")
print("Test accuracy:", f"{test_accuracy:.4f}")
print("Test ROC AUC:", f"{test_roc_auc:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    )
)

ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    y_test,
)

plt.title("Pipeline Tuning: Test-Set Confusion Matrix")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 8. Inspect transformed feature names
#
# The fitted ColumnTransformer exposes the names of the generated
# features, including the one-hot encoded categorical features.

# %%
fitted_preprocessor = best_model.named_steps["preprocessor"]

transformed_feature_names = fitted_preprocessor.get_feature_names_out()

print("\nNumber of transformed features:", len(transformed_feature_names))
print("\nTransformed feature names:")

for name in transformed_feature_names:
    print(" -", name)


# %% [markdown]
# ## 9. Predict a new observation
#
# Pass a DataFrame with the same input columns used during training.
# Missing values and previously unseen categories are handled by
# the fitted preprocessing pipeline.

# %%
new_customer = pd.DataFrame(
    [
        {
            "age": 38,
            "income": 62000,
            "city": "Frankfurt",
            "membership": "Premium",
        }
    ]
)

prediction = best_model.predict(new_customer)[0]
probability = best_model.predict_proba(new_customer)[0, 1]

print("\nNew customer prediction:")
print("Predicted renewal:", int(prediction))
print("Estimated renewal probability:", f"{probability:.4f}")
