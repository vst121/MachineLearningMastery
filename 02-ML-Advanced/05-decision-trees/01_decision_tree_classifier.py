# %% 1. Imports
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# %% 2. Load the Dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names,
)

y = pd.Series(
    iris.target,
    name="species",
)

print("Dataset shape:", X.shape)
print("\nClass distribution:")
print(y.value_counts().sort_index().rename(index=dict(enumerate(iris.target_names))))


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


# %% 4. Train the Decision Tree Classifier
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)


# %% 5. Make Predictions
y_pred = model.predict(X_test)

print("\nFirst 10 predictions:")
print(y_pred[:10])

print("\nFirst 10 actual labels:")
print(y_test.iloc[:10].to_numpy())


# %% 6. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        zero_division=0,
    )
)

confusion = confusion_matrix(y_test, y_pred)

confusion_df = pd.DataFrame(
    confusion,
    index=iris.target_names,
    columns=iris.target_names,
)

print("\nConfusion Matrix:")
print(confusion_df)


# %% 7. Inspect the Learned Tree
print("\nTree Depth:", model.get_depth())
print("Number of Leaves:", model.get_n_leaves())
print("Number of Nodes:", model.tree_.node_count)

print("\nFeature Importances:")
feature_importances = pd.Series(
    model.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print(feature_importances)


# %% 8. Visualize the Decision Tree
plt.figure(figsize=(18, 9))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    proportion=False,
    impurity=True,
    fontsize=9,
)

plt.title("Decision Tree Classifier for Iris")
plt.tight_layout()
plt.show()


# %% 9. Visualize Feature Importances
feature_importances.sort_values().plot(
    kind="barh",
    figsize=(9, 5),
)

plt.title("Decision Tree Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# %% 10. Predict a New Sample
new_sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns,
)

predicted_class = model.predict(new_sample)[0]
predicted_probabilities = model.predict_proba(new_sample)[0]

print("\nNew Sample Prediction:")
print("Predicted species:", iris.target_names[predicted_class])

print("\nClass Probabilities:")
for class_name, probability in zip(
    iris.target_names,
    predicted_probabilities,
):
    print(f"{class_name}: {probability:.4f}")
