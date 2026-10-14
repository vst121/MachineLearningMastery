# %% [markdown]
# # SVM Learning Curves
#
# Learning curves show how model performance changes as the
# training dataset grows.
#
# They help diagnose:
# - Underfitting: training and validation scores are both low.
# - Overfitting: training score is high, validation score is lower.
# - Data limitations: validation performance may improve with
#   additional training samples.
#
# A Pipeline ensures scaling is fitted independently in each fold.

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# %% [markdown]
# ## 1. Load the dataset

# %%
iris = load_iris()
X = iris.data
y = iris.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")


# %% [markdown]
# ## 2. Define the model pipeline
#
# These hyperparameters are fixed for this experiment.
# In a separate workflow, they could be selected using
# GridSearchCV on training data only.

# %%
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ]
)


# %% [markdown]
# ## 3. Configure stratified cross-validation

# %%
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# %% [markdown]
# ## 4. Calculate learning curves
#
# Each training size is evaluated across the CV folds.
# The scores measure classification accuracy.

# %%
train_sizes, train_scores, validation_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    shuffle=True,
    random_state=42,
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)

validation_mean = validation_scores.mean(axis=1)
validation_std = validation_scores.std(axis=1)

results = zip(
    train_sizes,
    train_mean,
    validation_mean,
    strict=True,
)

print(f"{'Samples':>8} {'Train accuracy':>16} {'CV accuracy':>14}")
print("-" * 42)

for size, train_score, cv_score in results:
    print(f"{size:>8} {train_score:>16.4f} {cv_score:>14.4f}")


# %% [markdown]
# ## 5. Visualize the learning curves

# %%
plt.figure(figsize=(9, 6))

plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training accuracy",
)
plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15,
)

plt.plot(
    train_sizes,
    validation_mean,
    marker="o",
    label="Cross-validation accuracy",
)
plt.fill_between(
    train_sizes,
    validation_mean - validation_std,
    validation_mean + validation_std,
    alpha=0.15,
)

plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")
plt.title("SVM Learning Curves")
plt.ylim(0.0, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 6. Interpret the results
#
# - Both scores low and close together: possible underfitting.
# - Training score high, validation score much lower:
#   possible overfitting.
# - Scores converge at a high level: good generalization.
# - Validation score still improving at the largest training
#   size: more data may help.
#
# These are diagnostic patterns, not absolute rules. Interpret
# them alongside dataset size, score variability, and the task.

# %%
score_gap = train_mean[-1] - validation_mean[-1]

print(f"Final training accuracy: {train_mean[-1]:.4f}")
print(f"Final CV accuracy:       {validation_mean[-1]:.4f}")
print(f"Final score gap:         {score_gap:.4f}")
