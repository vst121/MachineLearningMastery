"""
PyTorch nn.Module and Model Composition
=======================================

Experiments:
1. Build a neural network by subclassing nn.Module.
2. Compose layers using nn.Sequential.
3. Inspect model architecture and trainable parameters.
4. Understand weights, biases, and parameter shapes.
5. Compare trainable and non-trainable parameters.
6. Apply different weight initialization strategies.
7. Inspect model state using state_dict().
8. Compare model outputs and parameter counts.
"""

import torch
import torch.nn as nn


def section(title):
    print(f"\n{'=' * 65}")
    print(title)
    print("=" * 65)


# --------------------------------------------------
# 1. Build a model using nn.Module
# --------------------------------------------------

class BinaryClassifier(nn.Module):
    def __init__(self, input_features=10, hidden_features=32):
        super().__init__()

        self.hidden_layer = nn.Linear(
            input_features,
            hidden_features,
        )
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(
            hidden_features,
            1,
        )

    def forward(self, inputs):
        hidden = self.hidden_layer(inputs)
        hidden = self.activation(hidden)
        logits = self.output_layer(hidden)

        return logits


def inspect_custom_model():
    section("1. Model Using nn.Module")

    model = BinaryClassifier()

    print(model)

    inputs = torch.randn(8, 10)
    outputs = model(inputs)

    print("Input shape:", inputs.shape)
    print("Output shape:", outputs.shape)

    # A single output logit per sample.
    print("Output logits:\n", outputs[:3])


# --------------------------------------------------
# 2. Compose layers using nn.Sequential
# --------------------------------------------------

def inspect_sequential_model():
    section("2. Model Using nn.Sequential")

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    print(model)

    inputs = torch.randn(8, 10)
    outputs = model(inputs)

    print("Input shape:", inputs.shape)
    print("Output shape:", outputs.shape)


# --------------------------------------------------
# 3. Inspect trainable parameters
# --------------------------------------------------

def inspect_parameters():
    section("3. Trainable Parameters")

    model = BinaryClassifier(
        input_features=10,
        hidden_features=32,
    )

    total_parameters = 0
    trainable_parameters = 0

    for name, parameter in model.named_parameters():
        print(
            f"{name:<25}",
            f"shape={str(tuple(parameter.shape)):<12}",
            f"requires_grad={parameter.requires_grad}",
            f"count={parameter.numel()}",
        )

        total_parameters += parameter.numel()

        if parameter.requires_grad:
            trainable_parameters += parameter.numel()

    print("\nTotal parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)

    # Linear layer parameter counts:
    # First layer: 10 * 32 weights + 32 biases = 352
    # Output layer: 32 * 1 weights + 1 bias = 33
    # Total: 385 parameters.


# --------------------------------------------------
# 4. Understand weights and biases
# --------------------------------------------------

def inspect_weights_and_biases():
    section("4. Weights and Biases")

    layer = nn.Linear(
        in_features=4,
        out_features=3,
    )

    print("Weight shape:", layer.weight.shape)
    print("Bias shape:", layer.bias.shape)

    print("\nWeights:\n", layer.weight)
    print("\nBiases:\n", layer.bias)

    inputs = torch.randn(2, 4)
    outputs = layer(inputs)

    print("\nInput shape:", inputs.shape)
    print("Output shape:", outputs.shape)

    # Linear transformation:
    # output = input @ weight.T + bias
    manual_outputs = inputs @ layer.weight.T + layer.bias

    print(
        "Manual computation matches layer:",
        torch.allclose(outputs, manual_outputs),
    )


# --------------------------------------------------
# 5. Trainable and non-trainable parameters
# --------------------------------------------------

class ModelWithFrozenLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Linear(10, 16)
        self.classifier = nn.Linear(16, 1)

        # Freeze the feature extractor's parameters.
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        features = self.feature_extractor(inputs)
        features = torch.relu(features)

        return self.classifier(features)


def inspect_frozen_parameters():
    section("5. Trainable and Non-Trainable Parameters")

    model = ModelWithFrozenLayer()

    for name, parameter in model.named_parameters():
        print(
            f"{name:<35}",
            f"requires_grad={parameter.requires_grad}",
        )

    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )

    print("\nTrainable parameters:", trainable_count)
    print("Frozen parameters:", frozen_count)

    # Only trainable parameters should be passed to the optimizer
    # when the intention is to optimize only unfrozen weights.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
    )

    print("Optimizer:", type(optimizer).__name__)


# --------------------------------------------------
# 6. Weight initialization
# --------------------------------------------------

def compare_initialization():
    section("6. Weight Initialization")

    default_layer = nn.Linear(100, 50)
    xavier_layer = nn.Linear(100, 50)
    kaiming_layer = nn.Linear(100, 50)

    nn.init.xavier_uniform_(xavier_layer.weight)
    nn.init.zeros_(xavier_layer.bias)

    nn.init.kaiming_uniform_(
        kaiming_layer.weight,
        nonlinearity="relu",
    )
    nn.init.zeros_(kaiming_layer.bias)

    layers = {
        "Default": default_layer,
        "Xavier": xavier_layer,
        "Kaiming": kaiming_layer,
    }

    for name, layer in layers.items():
        print(f"\n{name} initialization")
        print("Weight mean:", layer.weight.mean().item())
        print("Weight standard deviation:", layer.weight.std().item())
        print("Bias mean:", layer.bias.mean().item())

    # Xavier initialization is often used with tanh-like activations.
    # Kaiming initialization is designed for ReLU-like activations.


# --------------------------------------------------
# 7. Inspect model state
# --------------------------------------------------

def inspect_model_state():
    section("7. Model State Dictionary")

    model = BinaryClassifier()

    state = model.state_dict()

    print("State dictionary keys:")

    for name, tensor in state.items():
        print(f"{name:<25} shape={tuple(tensor.shape)}")

    # Create a second model with the same architecture.
    restored_model = BinaryClassifier()
    restored_model.load_state_dict(state)

    inputs = torch.randn(4, 10)

    model.eval()
    restored_model.eval()

    with torch.no_grad():
        original_output = model(inputs)
        restored_output = restored_model(inputs)

    print(
        "\nOutputs match after loading state:",
        torch.allclose(original_output, restored_output),
    )


# --------------------------------------------------
# 8. Compare model composition
# --------------------------------------------------

def compare_model_composition():
    section("8. Comparing Model Composition")

    custom_model = BinaryClassifier(
        input_features=10,
        hidden_features=32,
    )

    sequential_model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    custom_count = sum(
        parameter.numel()
        for parameter in custom_model.parameters()
    )

    sequential_count = sum(
        parameter.numel()
        for parameter in sequential_model.parameters()
    )

    print("Custom model parameters:", custom_count)
    print("Sequential model parameters:", sequential_count)

    # Both architectures have the same layer dimensions.
    # Their parameter values are initialized independently.
    print("\nCustom model:\n", custom_model)
    print("\nSequential model:\n", sequential_model)


# --------------------------------------------------
# 9. Main
# --------------------------------------------------

def main():
    torch.manual_seed(42)

    inspect_custom_model()
    inspect_sequential_model()
    inspect_parameters()
    inspect_weights_and_biases()
    inspect_frozen_parameters()
    compare_initialization()
    inspect_model_state()
    compare_model_composition()


if __name__ == "__main__":
    main()
