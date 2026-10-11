"""
Machine Learning Foundations
Topic: Feature Scaling

Learning objectives:
- Understand why feature scales matter.
- Standardize features using StandardScaler.
- Compare SVM performance before and after scaling.
- Avoid data leakage by fitting the scaler on training data only.
- Combine preprocessing and model training with Pipeline.
"""

# %%
# 1. Imports

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# %%
# 2. Prepare the dataset

dataset = load_iris(as_frame=True)
df = dataset.frame

# Keep Setosa and Versicolor for binary classification.
df = df[df["target"].isin([0, 1])].copy()

feature_names = ["sepal length (cm)", "petal length (cm)"]

X = df[feature_names]
y = df["target"]

target_names = dataset.target_names[:2]

print("Dataset preview:")
print(df[feature_names + ["target"]].head())

print("\nFeature ranges:")
print(X.agg(["min", "max"]).T)


# %%
# 3. Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training observations: {len(X_train)}")
print(f"Test observations: {len(X_test)}")


# %%
# 4. Train an SVM without feature scaling

model_without_scaling = SVC(kernel="linear")

model_without_scaling.fit(X_train, y_train)

predictions_without_scaling = model_without_scaling.predict(X_test)

accuracy_without_scaling = accuracy_score(
    y_test,
    predictions_without_scaling,
)

print(f"Accuracy without scaling: {accuracy_without_scaling:.2%}")


# %%
# 5. Standardize the features

# StandardScaler transforms each feature approximately to:
# mean = 0
# standard deviation = 1
#
# Important: fit the scaler on training data only.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Means after scaling (training data):")
print(X_train_scaled.mean(axis=0))

print("\nStandard deviations after scaling (training data):")
print(X_train_scaled.std(axis=0))


# %%
# 6. Train an SVM with scaled features

model_with_scaling = SVC(kernel="linear")

model_with_scaling.fit(X_train_scaled, y_train)

predictions_with_scaling = model_with_scaling.predict(X_test_scaled)

accuracy_with_scaling = accuracy_score(
    y_test,
    predictions_with_scaling,
)

print(f"Accuracy with scaling: {accuracy_with_scaling:.2%}")


# %%
# 7. Compare the results

comparison = pd.DataFrame({
    "Model": [
        "Linear SVM without scaling",
        "Linear SVM with scaling",
    ],
    "Test accuracy": [
        accuracy_without_scaling,
        accuracy_with_scaling,
    ],
})

print(comparison)


# %%
# 8. Use a Pipeline

# Pipeline applies StandardScaler first and then trains the SVM.
# During fit(), the scaler learns its statistics from training data.
# During predict(), the same transformation is applied automatically.

pipeline_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear")),
])

pipeline_model.fit(X_train, y_train)

pipeline_predictions = pipeline_model.predict(X_test)

pipeline_accuracy = accuracy_score(y_test, pipeline_predictions)

print(f"Pipeline test accuracy: {pipeline_accuracy:.2%}")


# %%
# 9. Visualize the feature distributions before and after scaling

X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=feature_names,
    index=X_train.index,
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

X_train.boxplot(ax=axes[0])
axes[0].set_title("Original Features")
axes[0].set_ylabel("Feature value")
axes[0].tick_params(axis="x", rotation=20)

X_train_scaled_df.boxplot(ax=axes[1])
axes[1].set_title("Standardized Features")
axes[1].set_ylabel("Standardized value")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.show()

# %%
