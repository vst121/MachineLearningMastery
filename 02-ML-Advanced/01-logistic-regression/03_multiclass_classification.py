
# %% 1. Imports
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% 2. Load the dataset
iris = load_iris()

X = iris.data
y = iris.target

class_names = iris.target_names

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Classes: {class_names}")
print(f"Class counts: {np.bincount(y)}")

# %% 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# %% 4. Train One-vs-Rest Logistic Regression
ovr_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    ),
)

ovr_model.fit(X_train, y_train)

# %% 5. Train multinomial Logistic Regression
multinomial_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    ),
)

multinomial_model.fit(X_train, y_train)

# %% 6. Compare model performance
models = {
    "One-vs-Rest": ovr_model,
    "Multinomial": multinomial_model,
}

for model_name, model in models.items():
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{model_name}")
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=class_names,
            zero_division=0,
        )
    )

# %% 7. Inspect predicted probabilities
sample_count = 10

probabilities = multinomial_model.predict_proba(X_test[:sample_count])
predictions = multinomial_model.predict(X_test[:sample_count])

print("\nPredicted probabilities for the first 10 test samples:")
print("Columns:", class_names)
print(np.round(probabilities, 3))

print("\nActual classes:")
print(class_names[y_test[:sample_count]])

print("\nPredicted classes:")
print(class_names[predictions])

# %% 8. Visualize predicted probabilities
sample_indices = np.arange(sample_count)
bar_width = 0.25

plt.figure(figsize=(11, 6))

for class_index, class_name in enumerate(class_names):
    plt.bar(
        sample_indices + class_index * bar_width,
        probabilities[:, class_index],
        width=bar_width,
        label=class_name,
    )

plt.xlabel("Test sample index")
plt.ylabel("Predicted probability")
plt.title("Multinomial Logistic Regression: Class Probabilities")
plt.xticks(
    sample_indices + bar_width,
    [str(i) for i in sample_indices],
)
plt.ylim(0, 1.05)
plt.legend(title="Predicted class")
plt.tight_layout()
plt.show()

# %% 9. Inspect model coefficients
# The final pipeline step is the Logistic Regression estimator.
classifier = multinomial_model.named_steps["logisticregression"]

print("\nCoefficient matrix shape:", classifier.coef_.shape)
print("Intercept shape:", classifier.intercept_.shape)

for class_index, class_name in enumerate(class_names):
    print(f"\nClass: {class_name}")
    for feature_name, coefficient in zip(
        iris.feature_names,
        classifier.coef_[class_index],
    ):
        print(f"  {feature_name}: {coefficient:.3f}")
