# %% [markdown]
# # Feature Transformation and Scaling
#
# Feature transformations change the distribution or representation of
# numerical features. Feature scaling adjusts their numerical ranges.
#
# This example covers:
# - Log transformation
# - Power transformation
# - StandardScaler
# - MinMaxScaler
# - RobustScaler
#
# Important: transformations and scaling are not interchangeable.
# A transformation can change the shape of a distribution, while scaling
# primarily changes its location and/or range.

# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

# %% [markdown]
# ## 1. Load and inspect the Iris dataset
#
# The Iris dataset contains four numerical features measured in centimeters.
# We use petal length as an example for comparing distributions.

# %%
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

feature_index = feature_names.index("petal length (cm)")
petal_length = X[:, feature_index].reshape(-1, 1)

print(f"Dataset shape: {X.shape}")
print(f"Feature: {feature_names[feature_index]}")
print(f"Mean: {petal_length.mean():.3f}")
print(f"Standard deviation: {petal_length.std():.3f}")
print(f"Minimum: {petal_length.min():.3f}")
print(f"Maximum: {petal_length.max():.3f}")

# %% [markdown]
# ## 2. Understand the original distribution
#
# Histograms help identify skewness, spread, and unusual observations.
# The Iris petal-length feature is not strongly right-skewed, so it is
# useful for demonstrating techniques but not ideal for illustrating
# every possible benefit of a log transformation.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(petal_length.ravel(), bins=15, edgecolor="black")
axes[0].set_title("Original Petal Length")
axes[0].set_xlabel("Petal length (cm)")
axes[0].set_ylabel("Frequency")

axes[1].boxplot(petal_length.ravel(), vert=True)
axes[1].set_title("Original Petal Length: Box Plot")
axes[1].set_ylabel("Petal length (cm)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Apply a log transformation
#
# Log transformations can reduce right skew in positive-valued features.
# The natural logarithm is defined only for positive values.
#
# For data containing zeros or negative values, choose a suitable
# transformation or shift the data only when that shift is meaningful.

# %%
if np.all(petal_length > 0):
    log_transformed = np.log1p(petal_length)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(petal_length.ravel(), bins=15, edgecolor="black")
    axes[0].set_title("Original Distribution")
    axes[0].set_xlabel("Petal length (cm)")

    axes[1].hist(log_transformed.ravel(), bins=15, edgecolor="black")
    axes[1].set_title("After log1p Transformation")
    axes[1].set_xlabel("log(1 + petal length)")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Apply a power transformation
#
# Yeo-Johnson supports both positive and negative values.
# Unlike a manually specified logarithm, PowerTransformer estimates
# a transformation parameter from the data.
#
# For predictive workflows, fit the transformer on training data only.

# %%
power_transformer = PowerTransformer(method="yeo-johnson")
petal_power = power_transformer.fit_transform(petal_length)

print("Power transformation:")
print(f"Original mean: {petal_length.mean():.3f}")
print(f"Transformed mean: {petal_power.mean():.3f}")
print(f"Original standard deviation: {petal_length.std():.3f}")
print(f"Transformed standard deviation: {petal_power.std():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(petal_length.ravel(), bins=15, edgecolor="black")
axes[0].set_title("Original Distribution")

axes[1].hist(petal_power.ravel(), bins=15, edgecolor="black")
axes[1].set_title("After Yeo-Johnson Transformation")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Compare feature scaling techniques
#
# We apply three scalers to the complete Iris feature matrix:
#
# - StandardScaler: centers features around zero and scales by standard deviation.
# - MinMaxScaler: maps features to a specified range, usually [0, 1].
# - RobustScaler: uses the median and interquartile range, reducing sensitivity
#   to extreme values.
#
# The scalers do not generally make distributions normally distributed.

# %%
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
}

scaled_data = {}

for name, scaler in scalers.items():
    scaled_data[name] = scaler.fit_transform(X)

print("Original feature ranges:")
for name, values in zip(feature_names, X.T):
    print(
        f"{name:<25} "
        f"min={values.min():.3f}, max={values.max():.3f}"
    )

for scaler_name, values in scaled_data.items():
    print(f"\n{scaler_name}:")
    for feature_name, column in zip(feature_names, values.T):
        print(
            f"{feature_name:<25} "
            f"min={column.min():.3f}, max={column.max():.3f}, "
            f"mean={column.mean():.3f}"
        )

# %% [markdown]
# ## 6. Visualize the effect of scaling
#
# Box plots make it easier to compare the distributions of features
# after scaling. Each feature is plotted separately so differences in
# the original measurement units do not dominate the visualization.

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

datasets = [
    ("Original", X),
    ("StandardScaler", scaled_data["StandardScaler"]),
    ("MinMaxScaler", scaled_data["MinMaxScaler"]),
    ("RobustScaler", scaled_data["RobustScaler"]),
]

for ax, (dataset_name, values) in zip(axes, datasets):
    ax.boxplot(values, tick_labels=["SL", "SW", "PL", "PW"])
    ax.set_title(dataset_name)
    ax.set_xlabel("Feature")
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Feature value")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Compare scaling with and without an extreme value
#
# RobustScaler is less affected by extreme observations because it uses
# the median and interquartile range.
#
# This synthetic example makes the difference easier to observe.

# %%
sample = np.array(
    [10, 11, 12, 12, 13, 14, 15, 16, 17, 100],
    dtype=float,
).reshape(-1, 1)

standard_sample = StandardScaler().fit_transform(sample)
minmax_sample = MinMaxScaler().fit_transform(sample)
robust_sample = RobustScaler().fit_transform(sample)

print("Original sample:")
print(sample.ravel())

print("\nScaled values:")
print(f"StandardScaler: {standard_sample.ravel().round(2)}")
print(f"MinMaxScaler:   {minmax_sample.ravel().round(2)}")
print(f"RobustScaler:   {robust_sample.ravel().round(2)}")

# %% [markdown]
# ## Key takeaways
#
# - Log and power transformations can change feature distributions.
# - Log transformations require suitable input values.
# - Yeo-Johnson supports positive and negative values.
# - StandardScaler is useful for many distance-based and gradient-based models.
# - MinMaxScaler maps features into a chosen range.
# - RobustScaler is less sensitive to extreme values.
# - Scaling is not always necessary for tree-based models.
# - Fit data-dependent transformations on training data only.
# - Choose transformations based on the data and model, then validate their impact.
