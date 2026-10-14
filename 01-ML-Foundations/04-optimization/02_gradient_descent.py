
# %% [markdown]
# # Gradient Descent
#
# Gradient Descent is an optimization algorithm that iteratively
# updates model parameters to minimize a cost function.
#
# In this example, we train a simple linear regression model:
#
#     y_pred = w * x + b
#
# We implement the algorithm from scratch using NumPy.

# %%
import numpy as np
import matplotlib.pyplot as plt


# %% [markdown]
# ## 1. Prepare the dataset
#
# The data follows an approximately linear relationship.
# The model must learn the slope (w) and intercept (b).

# %%
np.random.seed(42)

X = np.linspace(0, 10, 100)
y = 3.5 * X + 5 + np.random.normal(0, 2, size=X.shape)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.7, label="Training data")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Training Dataset")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# %% [markdown]
# ## 2. Define the model and cost function
#
# The model predicts:
#
#     y_pred = w * X + b
#
# We minimize Mean Squared Error (MSE).

# %%
def predict(X: np.ndarray, w: float, b: float) -> np.ndarray:
    """Predict target values using a linear model."""
    return w * X + b


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


# %% [markdown]
# ## 3. Implement Gradient Descent
#
# The gradients of MSE with respect to w and b are:
#
#     dJ/dw = (-2/m) * sum(X * (y - y_pred))
#     dJ/db = (-2/m) * sum(y - y_pred)
#
# Each update moves the parameters in the opposite direction
# of their gradients.

# %%
def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 1000,
) -> tuple[float, float, list[float]]:
    """Train a linear regression model using batch gradient descent."""
    if X.ndim != 1 or y.ndim != 1 or X.shape != y.shape:
        raise ValueError("X and y must be 1D arrays with matching shapes.")
    if X.size == 0:
        raise ValueError("X and y must not be empty.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    w = 0.0
    b = 0.0
    m = X.size
    cost_history = []

    for _ in range(epochs):
        y_pred = predict(X, w, b)
        errors = y_pred - y

        dw = (2 / m) * np.sum(X * errors)
        db = (2 / m) * np.sum(errors)

        w -= learning_rate * dw
        b -= learning_rate * db

        cost = mean_squared_error(y, predict(X, w, b))
        cost_history.append(cost)

    return w, b, cost_history


# %% [markdown]
# ## 4. Train the model

# %%
learning_rate = 0.01
epochs = 1000

w, b, cost_history = gradient_descent(
    X,
    y,
    learning_rate=learning_rate,
    epochs=epochs,
)

print(f"Learned slope (w):     {w:.4f}")
print(f"Learned intercept (b): {b:.4f}")
print(f"Final MSE:             {cost_history[-1]:.4f}")
print(f"Initial MSE:           {mean_squared_error(y, np.zeros_like(y)):.4f}")


# %% [markdown]
# ## 5. Visualize the fitted model

# %%
y_pred = predict(X, w, b)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.7, label="Training data")
plt.plot(X, y_pred, color="red", linewidth=2, label="Learned model")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Trained with Gradient Descent")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# %% [markdown]
# ## 6. Visualize convergence
#
# A decreasing cost indicates that optimization is progressing.
# The learning rate determines the size of each parameter update.

# %%
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), cost_history)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent Convergence")
plt.grid(True, alpha=0.3)
plt.show()


# %% [markdown]
# ## 7. Key Takeaways
#
# - The gradient indicates the direction of steepest increase.
# - Gradient Descent updates parameters in the opposite direction.
# - The learning rate controls the size of each update.
# - Too large a learning rate can cause divergence.
# - Too small a learning rate can make convergence slow.
# - Feature scaling can improve optimization when features have
#   very different scales.
