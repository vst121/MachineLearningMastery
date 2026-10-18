# %% [markdown]
# # NumPy vs. PyTorch
#
# Implement the same multilayer perceptron (MLP) using NumPy and PyTorch.
#
# Architecture:
#   Input:        3 features
#   Hidden layer: 4 neurons + ReLU
#   Output layer: 2 neurons
#
# We manually assign identical weights and biases to both implementations
# so that their intermediate and final outputs can be compared directly.

# %%
import numpy as np
import torch
import torch.nn as nn


# %% [markdown]
# ## 1. Configure reproducibility and input data
#
# NumPy uses weights shaped as (input_features, output_features)
# in the matrix multiplication X @ W.
#
# PyTorch nn.Linear stores weights as (output_features, input_features).

# %%
np.random.seed(42)
torch.manual_seed(42)

X = np.array(
    [
        [1.0, 2.0, 3.0],
        [0.5, -1.0, 2.0],
        [-2.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

print("Input matrix:")
print(X)
print("Input shape:", X.shape)


# %% [markdown]
# ## 2. Define shared parameters
#
# We use NumPy's X @ W convention for the manual implementation.
# The same weights are transposed when copied into nn.Linear.

# %%
W1 = np.array(
    [
        [0.2, -0.3, 0.5, 0.1],
        [0.4, 0.2, -0.1, 0.3],
        [-0.5, 0.6, 0.2, -0.4],
    ],
    dtype=np.float32,
)

b1 = np.array(
    [0.1, -0.2, 0.05, 0.3],
    dtype=np.float32,
)

W2 = np.array(
    [
        [0.3, -0.4],
        [0.2, 0.5],
        [-0.6, 0.1],
        [0.4, -0.2],
    ],
    dtype=np.float32,
)

b2 = np.array([0.1, -0.1], dtype=np.float32)

print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)


# %% [markdown]
# ## 3. Implement forward propagation with NumPy
#
# First layer:
#   Z1 = X @ W1 + b1
#   A1 = ReLU(Z1)
#
# Output layer:
#   Z2 = A1 @ W2 + b2
#
# We return intermediate values to make each computation inspectable.

# %%
def relu(x: np.ndarray) -> np.ndarray:
    """Apply ReLU element by element."""
    return np.maximum(0.0, x)


def numpy_forward(
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a two-layer MLP forward pass using NumPy."""
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2

    return Z1, A1, Z2, Z2


numpy_Z1, numpy_A1, numpy_Z2, numpy_output = numpy_forward(
    X, W1, b1, W2, b2
)

print("NumPy hidden pre-activations:")
print(numpy_Z1)

print("\nNumPy hidden activations:")
print(numpy_A1)

print("\nNumPy output:")
print(numpy_output)


# %% [markdown]
# ## 4. Implement the same network with PyTorch
#
# nn.Linear performs the affine transformation:
#   output = input @ weight.T + bias
#
# ReLU is included explicitly between the two linear layers.

# %%
torch_model = nn.Sequential(
    nn.Linear(in_features=3, out_features=4),
    nn.ReLU(),
    nn.Linear(in_features=4, out_features=2),
)

# Copy the same parameters into PyTorch.
with torch.no_grad():
    torch_model[0].weight.copy_(torch.tensor(W1.T))
    torch_model[0].bias.copy_(torch.tensor(b1))

    torch_model[2].weight.copy_(torch.tensor(W2.T))
    torch_model[2].bias.copy_(torch.tensor(b2))

torch_X = torch.tensor(X)

# Inspect the hidden layer separately.
with torch.no_grad():
    torch_Z1 = torch_model[0](torch_X)
    torch_A1 = torch_model[1](torch_Z1)
    torch_output = torch_model[2](torch_A1)

print("PyTorch hidden pre-activations:")
print(torch_Z1)

print("\nPyTorch hidden activations:")
print(torch_A1)

print("\nPyTorch output:")
print(torch_output)


# %% [markdown]
# ## 5. Compare intermediate and final outputs
#
# Comparing intermediate values helps identify errors in layer
# dimensions, parameter orientation, bias addition, or activation logic.

# %%
np.testing.assert_allclose(
    numpy_Z1,
    torch_Z1.detach().numpy(),
    rtol=1e-6,
    atol=1e-6,
)

np.testing.assert_allclose(
    numpy_A1,
    torch_A1.detach().numpy(),
    rtol=1e-6,
    atol=1e-6,
)

np.testing.assert_allclose(
    numpy_output,
    torch_output.detach().numpy(),
    rtol=1e-6,
    atol=1e-6,
)

print("Success: all NumPy and PyTorch outputs match.")


# %% [markdown]
# ## 6. Inspect PyTorch parameters
#
# PyTorch registers layer weights and biases as parameters.
# They can later receive gradients during training.

# %%
for name, parameter in torch_model.named_parameters():
    print(
        f"{name:<20} "
        f"shape={tuple(parameter.shape)}, "
        f"trainable={parameter.requires_grad}"
    )


# %% [markdown]
# ## 7. Count the network parameters
#
# First layer:
#   3 * 4 weights + 4 biases = 16 parameters
#
# Second layer:
#   4 * 2 weights + 2 biases = 10 parameters
#
# Total = 26 trainable parameters.

# %%
parameter_count = sum(
    parameter.numel()
    for parameter in torch_model.parameters()
)

expected_parameter_count = (3 * 4 + 4) + (4 * 2 + 2)

assert parameter_count == expected_parameter_count

print("Actual parameter count:  ", parameter_count)
print("Expected parameter count:", expected_parameter_count)


# %% [markdown]
# ## 8. Experiment with batch sizes
#
# The same network can process one sample or a batch of samples.
# The last dimension of the output remains the number of output neurons.

# %%
with torch.no_grad():
    single_output = torch_model(torch_X[:1])
    batch_output = torch_model(torch_X)

print("Single-sample output shape:", tuple(single_output.shape))
print("Batch output shape:        ", tuple(batch_output.shape))
