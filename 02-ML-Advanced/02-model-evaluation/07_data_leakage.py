# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# %% 2. Load the dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Features:", X.shape[1])
print("Classes:", data.target_names)


# %% 3. Create a training and test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))


# %% 4. Correct approach: fit preprocessing on training data only
correct_scaler = StandardScaler()

X_train_scaled = correct_scaler.fit_transform(X_train)
X_test_scaled = correct_scaler.transform(X_test)

correct_model = LogisticRegression(
    max_iter=5000,
    random_state=42,
)

correct_model.fit(X_train_scaled, y_train)

correct_train_predictions = correct_model.predict(X_train_scaled)
correct_test_predictions = correct_model.predict(X_test_scaled)

correct_train_accuracy = accuracy_score(
    y_train,
    correct_train_predictions,
)

correct_test_accuracy = accuracy_score(
    y_test,
    correct_test_predictions,
)

print("\nCorrect preprocessing")
print("-" * 35)
print(f"Train accuracy: {correct_train_accuracy:.4f}")
print(f"Test accuracy:  {correct_test_accuracy:.4f}")


# %% 5. Incorrect approach: scale before splitting
# This intentionally demonstrates data leakage.
# The scaler sees the feature distribution of the future test set.
leaky_scaler = StandardScaler()

X_scaled_before_split = leaky_scaler.fit_transform(X)

X_train_leaky, X_test_leaky, y_train_leaky, y_test_leaky = (
    train_test_split(
        X_scaled_before_split,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
)

leaky_model = LogisticRegression(
    max_iter=5000,
    random_state=42,
)

leaky_model.fit(X_train_leaky, y_train_leaky)

leaky_train_predictions = leaky_model.predict(X_train_leaky)
leaky_test_predictions = leaky_model.predict(X_test_leaky)

leaky_train_accuracy = accuracy_score(
    y_train_leaky,
    leaky_train_predictions,
)

leaky_test_accuracy = accuracy_score(
    y_test_leaky,
    leaky_test_predictions,
)

print("\nIncorrect preprocessing: leakage")
print("-" * 42)
print(f"Train accuracy: {leaky_train_accuracy:.4f}")
print(f"Test accuracy:  {leaky_test_accuracy:.4f}")


# %% 6. Compare the two approaches
print("\nComparison")
print("-" * 55)
print(
    f"{'Approach':<32}"
    f"{'Train Accuracy':>16}"
    f"{'Test Accuracy':>16}"
)

print(
    f"{'Correct preprocessing':<32}"
    f"{correct_train_accuracy:>16.4f}"
    f"{correct_test_accuracy:>16.4f}"
)

print(
    f"{'Scaling before split':<32}"
    f"{leaky_train_accuracy:>16.4f}"
    f"{leaky_test_accuracy:>16.4f}"
)


# %% 7. Compare the feature means used by the scalers
# The correctly fitted scaler sees only training samples.
# The leaky scaler sees the complete dataset.
correct_means = correct_scaler.mean_
leaky_means = leaky_scaler.mean_

mean_differences = np.abs(correct_means - leaky_means)

print("\nFeature mean differences")
print("-" * 35)
print("Largest mean difference:", mean_differences.max())
print("Average mean difference:", mean_differences.mean())


# %% 8. Visualize the difference in fitted feature means
feature_indices = np.arange(len(data.feature_names))

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    feature_indices,
    correct_means,
    marker="o",
    label="Training-only means",
)

ax.plot(
    feature_indices,
    leaky_means,
    marker="x",
    label="Full-dataset means",
)

ax.set_xlabel("Feature Index")
ax.set_ylabel("Feature Mean (original units)")
ax.set_title("Preprocessing Statistics: Training Data vs Full Dataset")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %% 9. Recommended approach: use a Pipeline
# Pipeline ensures the scaler is fitted only on the data
# passed to model.fit().
pipeline_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=5000,
        random_state=42,
    ),
)

pipeline_model.fit(X_train, y_train)

pipeline_train_accuracy = accuracy_score(
    y_train,
    pipeline_model.predict(X_train),
)

pipeline_test_accuracy = accuracy_score(
    y_test,
    pipeline_model.predict(X_test),
)

print("\nPipeline approach")
print("-" * 30)
print(f"Train accuracy: {pipeline_train_accuracy:.4f}")
print(f"Test accuracy:  {pipeline_test_accuracy:.4f}")


# %% 10. Verify the pipeline's preprocessing statistics
pipeline_scaler = pipeline_model.named_steps["standardscaler"]

print("\nPipeline verification")
print("-" * 35)
print(
    "Scaler fitted on training samples:",
    np.allclose(pipeline_scaler.mean_, X_train.mean(axis=0)),
)

print(
    "Scaler matches training-only scaler:",
    np.allclose(pipeline_scaler.mean_, correct_scaler.mean_),
)
