
# %% [markdown]
# # Multilayer Perceptron (MLP)
#
# An MLP is a feedforward neural network with one or more hidden layers.
#
# This experiment compares:
# 1. A linear model.
# 2. A network with multiple linear layers but no nonlinear activations.
# 3. An MLP with ReLU activations.
#
# All three models are trained on the same nonlinear regression problem.

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# %% [markdown]
# ## 1. Reproducibility
#
# Set random seeds to make the experiment more reproducible.

# %%
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# %% [markdown]
# ## 2. Generate a nonlinear dataset
#
# The target function is:
#   y = sin(x) + noise
#
# A linear model cannot represent this function accurately across
# the full input range.

# %%
X_all = np.linspace(-2 * np.pi, 2 * np.pi, 500, dtype=np.float32)
noise = np.random.normal(0.0, 0.1, size=X_all.shape).astype(np.float32)

y_all = np.sin(X_all) + noise

X = X_all.reshape(-1, 1)
y = y_all.reshape(-1, 1)

# Shuffle and split the dataset.
indices = np.random.permutation(len(X))
split_index = int(0.8 * len(X))

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = torch.tensor(X[train_indices], device=device)
y_train = torch.tensor(y[train_indices], device=device)
X_test = torch.tensor(X[test_indices], device=device)
y_test = torch.tensor(y[test_indices], device=device)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


# %% [markdown]
# ## 3. Define three network architectures
#
# Model A: one linear layer.
# Model B: multiple linear layers without nonlinear activations.
# Model C: hidden layers with ReLU activations.
#
# A stack of linear transformations is still equivalent to a single
# linear transformation. Nonlinear activations change that limitation.

# %%
linear_model = nn.Sequential(
    nn.Linear(1, 1)
).to(device)

stacked_linear_model = nn.Sequential(
    nn.Linear(1, 16),
    nn.Linear(16, 16),
    nn.Linear(16, 1),
).to(device)

mlp_model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
).to(device)

models = {
    "Linear": linear_model,
    "Stacked linear": stacked_linear_model,
    "MLP with ReLU": mlp_model,
}


# %% [markdown]
# ## 4. Inspect model parameters
#
# Each nn.Linear layer has a weight matrix and a bias vector.
# These are the learnable parameters of the network.

# %%
for name, model in models.items():
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    print(f"{name}: {parameter_count} parameters")


# %% [markdown]
# ## 5. Train each model
#
# We use mean squared error and Adam for this experiment.
# These are introduced here as tools; their underlying principles
# are covered in the loss-function and optimization modules.

# %%
def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 1500,
    learning_rate: float = 0.01,
) -> list[float]:
    """Train a regression model and return its epoch losses."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = []

    for epoch in range(epochs):
        model.train()

        predictions = model(X_train)
        loss = criterion(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.item())

    return history


histories = {}

for name, model in models.items():
    histories[name] = train_model(model, X_train, y_train)
    print(f"{name}: final training loss = {histories[name][-1]:.4f}")


# %% [markdown]
# ## 6. Evaluate on the held-out test set
#
# The test set was not used for parameter updates.

# %%
criterion = nn.MSELoss()
test_losses = {}

for name, model in models.items():
    model.eval()

    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test).item()

    test_losses[name] = test_loss
    print(f"{name}: test MSE = {test_loss:.4f}")


# %% [markdown]
# ## 7. Visualize the learned functions
#
# Sort the input values before plotting so the predicted curves
# are drawn from left to right.

# %%
plot_indices = np.argsort(X_test.cpu().numpy().ravel())
X_test_sorted = X_test[plot_indices].cpu().numpy()
y_test_sorted = y_test[plot_indices].cpu().numpy()

plt.figure(figsize=(11, 6))
plt.scatter(
    X_test_sorted,
    y_test_sorted,
    s=18,
    alpha=0.5,
    label="Test observations",
)

X_grid = torch.linspace(
    -2 * np.pi,
    2 * np.pi,
    500,
    device=device,
).reshape(-1, 1)

for name, model in models.items():
    model.eval()

    with torch.no_grad():
        predictions = model(X_grid).cpu().numpy()

    plt.plot(
        X_grid.cpu().numpy(),
        predictions,
        label=name,
    )

plt.title("Linear Models vs. Multilayer Perceptron")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 8. Compare the test results

# %%
print("\nTest MSE comparison")
print("-" * 35)

for name, loss in sorted(
    test_losses.items(),
    key=lambda item: item[1],
):
    print(f"{name:<20} {loss:.4f}")


# %% [markdown]
# ## 9. Inspect the MLP architecture
#
# Print the layers and parameter shapes to connect the model's
# architecture with its learned parameters.

# %%
print(mlp_model)

print("\nParameter shapes:")
for name, parameter in mlp_model.named_parameters():
    print(f"{name:<20} {tuple(parameter.shape)}")
