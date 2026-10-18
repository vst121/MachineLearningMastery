
# %% [markdown]
# # Artificial Neuron
#
# An artificial neuron computes a weighted sum of its inputs, adds a bias,
# and applies an activation function.
#
# z = sum(w_i * x_i) + b
# a = f(z)
#
# This script implements a single neuron using NumPy and verifies the
# same computation using PyTorch.

# %%
import numpy as np
import torch
import torch.nn as nn


# %% [markdown]
# ## 1. Define the inputs and parameters
#
# We use three input features, three weights, and one bias.
# These parameters are chosen manually for demonstration; they are not learned.

# %%
inputs = np.array([2.0, 3.0, 1.0], dtype=np.float32)
weights = np.array([0.5, -0.2, 0.8], dtype=np.float32)
bias = 0.1

print("Inputs: ", inputs)
print("Weights:", weights)
print("Bias:   ", bias)


# %% [markdown]
# ## 2. Calculate the weighted sum
#
# The dot product multiplies corresponding inputs and weights and sums
# the results. The bias is added afterward.

# %%
weighted_sum = np.dot(inputs, weights) + bias

print(f"Weighted sum (z): {weighted_sum:.4f}")

# Expected:
# (2.0 * 0.5) + (3.0 * -0.2) + (1.0 * 0.8) + 0.1 = 1.3


# %% [markdown]
# ## 3. Apply an activation function
#
# ReLU returns the input when it is positive and zero otherwise.
#
# ReLU(z) = max(0, z)

# %%
def relu(value: float) -> float:
    """Apply the ReLU activation function."""
    return max(0.0, value)


activation_output = relu(float(weighted_sum))

print(f"ReLU output: {activation_output:.4f}")


# %% [markdown]
# ## 4. Compare with a PyTorch implementation
#
# nn.Linear performs the weighted sum and bias addition.
# We copy our manually selected parameters into the layer.
#
# PyTorch Linear layers store weights with shape (out_features, in_features).

# %%
linear_layer = nn.Linear(in_features=3, out_features=1)

with torch.no_grad():
    linear_layer.weight.copy_(torch.tensor([weights]))
    linear_layer.bias.copy_(torch.tensor([bias]))

torch_inputs = torch.tensor(inputs)
torch_weighted_sum = linear_layer(torch_inputs)
torch_output = torch.relu(torch_weighted_sum)

print(f"PyTorch weighted sum: {torch_weighted_sum.item():.4f}")
print(f"PyTorch ReLU output:  {torch_output.item():.4f}")


# %% [markdown]
# ## 5. Verify that both implementations agree
#
# Floating-point calculations can introduce tiny numerical differences,
# so we use a tolerance rather than exact equality.

# %%
np.testing.assert_allclose(
    weighted_sum,
    torch_weighted_sum.item(),
    rtol=1e-6,
    atol=1e-6,
)

np.testing.assert_allclose(
    activation_output,
    torch_output.item(),
    rtol=1e-6,
    atol=1e-6,
)

print("Success: NumPy and PyTorch computations match.")


# %% [markdown]
# ## 6. Experiment with different inputs
#
# Change the input values and observe how the neuron's output changes.
# The weights and bias remain fixed.

# %%
experiments = np.array(
    [
        [2.0, 3.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, 2.0, -1.0],
    ],
    dtype=np.float32,
)

for sample in experiments:
    z = np.dot(sample, weights) + bias
    output = relu(float(z))

    print(
        f"Input={sample}, "
        f"z={z:.4f}, "
        f"ReLU output={output:.4f}"
    )
