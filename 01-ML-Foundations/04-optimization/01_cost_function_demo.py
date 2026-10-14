# %% [markdown]
# # Cost Functions
#
# A cost function measures how well a model's predictions match
# the actual target values.
#
# This example implements Mean Squared Error (MSE) and
# Mean Absolute Error (MAE) from scratch.

# %%
import numpy as np
import matplotlib.pyplot as plt


# %% [markdown]
# ## 1. Define the cost functions

# %%
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Squared Error."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Absolute Error."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    return float(np.mean(np.abs(y_true - y_pred)))


# %% [markdown]
# ## 2. Compare predictions

# %%
y_true = np.array([3.0, 5.0, 7.0, 9.0])

predictions = {
    "Good predictions": np.array([3.2, 4.8, 7.1, 8.9]),
    "Moderate predictions": np.array([2.0, 6.0, 8.0, 7.0]),
    "Poor predictions": np.array([0.0, 10.0, 3.0, 15.0]),
}

print("Cost Function Comparison")
print("-" * 58)
print(f"{'Model':<24}{'MSE':>16}{'MAE':>16}")
print("-" * 58)

for name, y_pred in predictions.items():
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name:<24}{mse:>16.3f}{mae:>16.3f}")


# %% [markdown]
# ## 3. Visualize the predictions

# %%
x = np.arange(len(y_true))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(x, y_true, "ko-", linewidth=2, label="Actual")
for name, y_pred in predictions.items():
    axes[0].plot(x, y_pred, "o--", label=name)

axes[0].set_title("Actual vs. Predicted Values")
axes[0].set_xlabel("Sample")
axes[0].set_ylabel("Target")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

mse_values = [
    mean_squared_error(y_true, y_pred)
    for y_pred in predictions.values()
]
mae_values = [
    mean_absolute_error(y_true, y_pred)
    for y_pred in predictions.values()
]
names = list(predictions.keys())
positions = np.arange(len(names))
width = 0.35

axes[1].bar(positions - width / 2, mse_values, width, label="MSE")
axes[1].bar(positions + width / 2, mae_values, width, label="MAE")
axes[1].set_title("Cost Function Comparison")
axes[1].set_xticks(positions)
axes[1].set_xticklabels(names, rotation=15, ha="right")
axes[1].set_ylabel("Error")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4. Key Takeaways
#
# - MSE squares errors, so large errors have a greater influence.
# - MAE measures the average absolute error and is less sensitive
#   to outliers than MSE.
# - Lower cost generally indicates better predictions for the
#   chosen cost function and dataset.
# - Gradient descent will use a cost function to guide parameter
#   updates in the next example.
