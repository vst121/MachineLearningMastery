# %% [markdown]
# # Preprocessing Pipeline
#
# A preprocessing pipeline combines data preparation and model training
# into one reproducible workflow.
#
# This example demonstrates:
# - Handling missing numerical and categorical values
# - Scaling numerical features
# - Encoding categorical features
# - Combining transformations with ColumnTransformer
# - Preventing data leakage with Pipeline

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# %% [markdown]
# ## 1. Create a sample dataset
#
# Predict whether a customer subscribes to a service.
# The dataset contains numerical and categorical features, with deliberately
# introduced missing values.
#
# This small synthetic dataset is intended for learning, not business conclusions.

# %%
data = pd.DataFrame(
    {
        "age": [
            22, 25, 47, 52, 46, 56, 23, 34, 41, 29,
            38, 49, 31, 44, 27, 53, 36, 24, 45, 33,
            58, 28, 39, 51, 26, 43, 35, 48, 30, 42,
        ],
        "income": [
            28000, 32000, 75000, 82000, 68000, 91000, 29000,
            51000, 62000, 39000, 55000, 78000, 42000, 71000,
            35000, 87000, 58000, 31000, 69000, 47000, 95000,
            36000, 60000, 84000, 33000, 73000, 49000, 80000,
            40000, 66000,
        ],
        "city": [
            "Mannheim", "Heidelberg", "Stuttgart", "Mannheim",
            "Heidelberg", "Stuttgart", "Mannheim", "Heidelberg",
            "Stuttgart", "Mannheim", "Heidelberg", "Stuttgart",
            "Mannheim", "Stuttgart", "Heidelberg", "Stuttgart",
            "Mannheim", "Heidelberg", "Stuttgart", "Mannheim",
            "Stuttgart", "Heidelberg", "Mannheim", "Stuttgart",
            "Heidelberg", "Mannheim", "Heidelberg", "Stuttgart",
            "Mannheim", "Heidelberg",
        ],
        "membership": [
            "Basic", "Basic", "Premium", "Premium", "Premium",
            "Premium", "Basic", "Standard", "Standard", "Basic",
            "Standard", "Premium", "Basic", "Premium", "Basic",
            "Premium", "Standard", "Basic", "Premium", "Standard",
            "Premium", "Basic", "Standard", "Premium", "Basic",
            "Premium", "Standard", "Premium", "Basic", "Standard",
        ],
        "subscribed": [
            0, 0, 1, 1, 1, 1, 0, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
            1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
        ],
    }
)

# Introduce missing values.
data.loc[[2, 8, 15], "age"] = np.nan
data.loc[[4, 11, 19], "income"] = np.nan
data.loc[[3, 12], "city"] = np.nan
data.loc[[7, 20], "membership"] = np.nan

X = data.drop(columns="subscribed")
y = data["subscribed"]

print("Dataset preview:")
print(data.head())

print("\nMissing values per feature:")
print(X.isna().sum())

print(f"\nDataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")

# %% [markdown]
# ## 2. Split the data
#
# Split before fitting any preprocessing transformation.
# This ensures that imputation values, scaling parameters, and category
# encodings are learned from the training data only.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# %% [markdown]
# ## 3. Define numerical and categorical preprocessing
#
# Numerical features:
# - Impute missing values using the training median.
# - Standardize the features.
#
# Categorical features:
# - Impute missing values using the most frequent category.
# - Apply one-hot encoding.
#
# handle_unknown="ignore" allows the pipeline to handle categories
# not encountered during fitting.

# %%
numeric_features = ["age", "income"]
categorical_features = ["city", "membership"]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

# %% [markdown]
# ## 4. Build the complete machine learning pipeline
#
# The complete pipeline first preprocesses the input and then trains
# Logistic Regression.
#
# Calling fit() on the full pipeline fits every preprocessing step using
# the training data before fitting the classifier.

# %%
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ]
)

model.fit(X_train, y_train)

print("Pipeline fitted successfully.")

# %% [markdown]
# ## 5. Inspect the transformed features
#
# One-hot encoding expands categorical columns into binary columns.
# The resulting matrix can therefore contain more columns than the
# original dataset.

# %%
fitted_preprocessor = model.named_steps["preprocessor"]

X_train_transformed = fitted_preprocessor.transform(X_train)
feature_names = fitted_preprocessor.get_feature_names_out()

print(f"Original feature count: {X_train.shape[1]}")
print(f"Transformed feature count: {len(feature_names)}")
print(f"Transformed matrix shape: {X_train_transformed.shape}")

print("\nTransformed feature names:")
for feature_name in feature_names:
    print(f"  {feature_name}")

# %% [markdown]
# ## 6. Evaluate the model
#
# The test set is passed through the transformations already fitted on
# the training data. It is not used to learn preprocessing parameters.

# %%
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Not subscribed", "Subscribed"],
)
plt.title("Customer Subscription: Confusion Matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Predict for a new customer
#
# Supply raw input features. The fitted pipeline automatically applies
# the same imputation, scaling, and encoding transformations.

# %%
new_customer = pd.DataFrame(
    {
        "age": [37],
        "income": [62000],
        "city": ["Mannheim"],
        "membership": ["Premium"],
    }
)

prediction = model.predict(new_customer)[0]
probability = model.predict_proba(new_customer)[0, 1]

print(f"Predicted class: {prediction}")
print(f"Subscription probability: {probability:.2%}")

# %% [markdown]
# ## Key takeaways
#
# - SimpleImputer handles missing numerical and categorical values.
# - StandardScaler standardizes numerical features.
# - OneHotEncoder converts categorical features into numerical columns.
# - ColumnTransformer applies different transformations to different columns.
# - Pipeline combines preprocessing and model training.
# - Split data before fitting preprocessing to prevent data leakage.
# - Use held-out data to evaluate generalization.
