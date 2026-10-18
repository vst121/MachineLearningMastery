# %% [markdown]
# # Forward Propagation
#
# Forward propagation passes input data through neural network layers
# to produce predictions.
#
# For a layer:
#   Z = X @ W.T + b
#   A = f(Z)
#
# Convention:
#   X: (batch_size, input_features)
#   W: (neurons, input_features)
#   b: (neurons,)
#   Z, A: (batch_size, neurons)

# %%
import numpy as np
import torch
import torch.nn as nn


# %% [markdown]
# ## 1. Define a batch of input samples
#
# Each row is one sample; each column is one feature.

# %%
X = np.array(
    [
        [2.0, 3.0, 1.0],
        [1.0, 1.0, 2.0],
        [0.0, 2.0, 1.0],
    ],
    dtype=np.float32,
)

print("Input matrix X:")
print(X)
print("Shape:", X.shape)


# %% [markdown]
# ## 2. Forward propagation through one layer
#
# This layer has three input features and two neurons.
# Each neuron has its own weights and bias.

# %%
W = np.array(
    [
        [0.5, -0.2, 0.8],
        [-0.3, 0.4, 0.1],
    ],
    dtype=np.float32,
)

b = np.array([0.1, -0.2], dtype=np.float32)

Z = X @ W.T + b

print("Pre-activation values Z:")
print(Z)
print("Shape:", Z.shape)


# %% [markdown]
# ## 3. Apply ReLU element by element
#
# ReLU(z) = max(0, z)

# %%
A = np.maximum(0.0, Z)

print("Activations A:")
print(A)
print("Shape:", A.shape)


# %% [markdown]
# ## 4. Build the same layer with PyTorch
#
# nn.Linear stores its weight matrix in the shape:
# (out_features, in_features).
#
# Its output is X @ W.T + b.

# %%
layer = nn.Linear(in_features=3, out_features=2)

with torch.no_grad():
    layer.weight.copy_(torch.tensor(W))
    layer.bias.copy_(torch.tensor(b))

torch_X = torch.tensor(X)
torch_Z = layer(torch_X)
torch_A = torch.relu(torch_Z)

print("PyTorch pre-activations:")
print(torch_Z)

print("PyTorch activations:")
print(torch_A)


# %% [markdown]
# ## 5. Verify the implementations

# %%
np.testing.assert_allclose(
    Z,
    torch_Z.detach().numpy(),
    rtol=1e-6,
    atol=1e-6,
)

np.testing.assert_allclose(
    A,
    torch_A.detach().numpy(),
    rtol=1e-6,
    atol=1e-6,
)

print("Success: NumPy and PyTorch outputs match.")


# %% [markdown]
# ## 6. Forward propagation through two layers
#
# The first layer maps 3 input features to 4 hidden neurons.
# The second layer maps 4 hidden activations to 2 outputs.
#
# ReLU is applied after the hidden layer.
# The output layer is linear; the appropriate output activation
# depends on the task and will be studied later.

# %%
W1 = np.array(
    [
        [0.2, 0.4, -0.5],
        [-0.3, 0.1, 0.2],
        [0.7, -0.2, 0.1],
        [0.1, 0.3, 0.6],
    ],
    dtype=np.float32,
)

b1 = np.array([0.1, 0.0, -0.1, 0.2], dtype=np.float32)

W2 = np.array(
    [
        [0.3, -0.4, 0.2, 0.5],
        [-0.2, 0.6, 0.1, -0.3],
    ],
    dtype=np.float32,
)

b2 = np.array([0.05, -0.05], dtype=np.float32)

Z1 = X @ W1.T + b1
A1 = np.maximum(0.0, Z1)

Z2 = A1 @ W2.T + b2

print("Hidden layer shape:", A1.shape)
print("Output layer shape:", Z2.shape)
print("Network outputs:")
print(Z2)


# %% [markdown]
# ## 7. Verify the two-layer network with PyTorch

# %%
network = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
)

with torch.no_grad():
    network[0].weight.copy_(torch.tensor(W1))
    network[0].bias.copy_(torch.tensor(b1))
    network[2].weight.copy_(torch.tensor(W2))
    network[2].bias.copy_(torch.tensor(b2))

torch_Z2 = network(torch_X)

np.testing.assert_allclose(
    Z2,
    torch_Z2.detach().numpy(),
    rtol=1e-6,
    atol=1e-6,
)

print("Success: two-layer NumPy and PyTorch outputs match.")


# %% [markdown]
# ## 8. Inspect the dimensions
#
# Understanding tensor shapes is essential when building neural networks.

# %%
print("Input X:             ", X.shape)
print("First-layer weights: ", W1.shape)
print("First-layer bias:    ", b1.shape)
print("Hidden activations:  ", A1.shape)
print("Second-layer weights:", W2.shape)
print("Second-layer bias:   ", b2.shape)
print("Final output:        ", Z2.shape)
