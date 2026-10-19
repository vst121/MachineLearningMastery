"""
Tensor Shapes and Broadcasting
==============================

Experiments:
1. Inspect tensor shapes, dimensions, and sizes.
2. Reshape, flatten, squeeze, and unsqueeze tensors.
3. Explore broadcasting rules with compatible and incompatible shapes.
4. Perform matrix multiplication and verify output dimensions.
5. Work with batches of inputs and batched matrix operations.
6. Understand dimensions used in neural network layers.
7. Diagnose common tensor shape mismatch errors.
"""

import torch


def section(title):
    print(f"\n{'=' * 65}")
    print(title)
    print("=" * 65)


def show_shape(name, tensor):
    print(f"{name}: shape={tuple(tensor.shape)}, ndim={tensor.ndim}")


# --------------------------------------------------
# 1. Inspecting tensor shapes
# --------------------------------------------------

def inspect_shapes():
    section("1. Inspecting Tensor Shapes")

    tensors = {
        "Scalar": torch.tensor(5.0),
        "Vector": torch.tensor([1.0, 2.0, 3.0]),
        "Matrix": torch.randn(2, 3),
        "3D tensor": torch.randn(4, 2, 3),
        "4D tensor": torch.randn(8, 3, 32, 32),
    }

    for name, tensor in tensors.items():
        show_shape(name, tensor)
        print("  Number of elements:", tensor.numel())


# --------------------------------------------------
# 2. Reshaping and dimension manipulation
# --------------------------------------------------

def reshape_tensors():
    section("2. Reshaping and Dimension Manipulation")

    tensor = torch.arange(1, 13)

    matrix = tensor.reshape(3, 4)
    flattened = matrix.flatten()
    row_vector = tensor.unsqueeze(0)
    column_vector = tensor.unsqueeze(1)

    show_shape("Original vector", tensor)
    show_shape("Reshaped matrix", matrix)
    show_shape("Flattened matrix", flattened)
    show_shape("Row vector", row_vector)
    show_shape("Column vector", column_vector)

    print("\nMatrix:\n", matrix)
    print("\nRow vector:\n", row_vector)
    print("\nColumn vector:\n", column_vector)

    # Remove a dimension of size 1.
    restored_vector = row_vector.squeeze(0)
    show_shape("After squeeze(0)", restored_vector)

    # Transpose swaps the two dimensions of a matrix.
    transposed = matrix.T
    show_shape("Transposed matrix", transposed)
    print("\nTransposed matrix:\n", transposed)


# --------------------------------------------------
# 3. Broadcasting rules
# --------------------------------------------------

def broadcasting_examples():
    section("3. Broadcasting Rules")

    matrix = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    row = torch.tensor([[10.0, 20.0, 30.0]])
    column = torch.tensor([[100.0], [200.0]])
    scalar = torch.tensor(5.0)

    show_shape("Matrix", matrix)
    show_shape("Row", row)
    show_shape("Column", column)
    show_shape("Scalar", scalar)

    print("\nMatrix + row:\n", matrix + row)
    print("\nMatrix + column:\n", matrix + column)
    print("\nMatrix + scalar:\n", matrix + scalar)

    # Both singleton dimensions can expand to match (2, 3).
    row_column = row.T
    show_shape("Transposed row", row_column)

    print("\nRow + column:\n", row + column)

    # Broadcasting compares dimensions from right to left.
    # Each pair must be equal, or at least one dimension must be 1.


# --------------------------------------------------
# 4. Compatible and incompatible shapes
# --------------------------------------------------

def test_shape_compatibility():
    section("4. Compatible and Incompatible Shapes")

    examples = [
        (
            "Compatible: (2, 3) + (1, 3)",
            torch.randn(2, 3),
            torch.randn(1, 3),
        ),
        (
            "Compatible: (2, 3) + (2, 1)",
            torch.randn(2, 3),
            torch.randn(2, 1),
        ),
        (
            "Incompatible: (2, 3) + (4, 2)",
            torch.randn(2, 3),
            torch.randn(4, 2),
        ),
    ]

    for description, left, right in examples:
        print(f"\n{description}")
        show_shape("Left", left)
        show_shape("Right", right)

        try:
            result = left + right
            show_shape("Result", result)
        except RuntimeError as error:
            print("Operation failed:", error)


# --------------------------------------------------
# 5. Matrix multiplication
# --------------------------------------------------

def matrix_multiplication():
    section("5. Matrix Multiplication")

    A = torch.randn(2, 3)
    B = torch.randn(3, 4)

    result = A @ B

    show_shape("A", A)
    show_shape("B", B)
    show_shape("A @ B", result)

    print("\nDimension rule:")
    print("(2, 3) @ (3, 4) -> (2, 4)")

    # The inner dimensions must match.
    incompatible_B = torch.randn(2, 4)

    try:
        invalid_result = A @ incompatible_B
        print(invalid_result)
    except RuntimeError as error:
        print("\nIncompatible matrix multiplication:")
        print(error)


# --------------------------------------------------
# 6. Batch operations
# --------------------------------------------------

def batch_operations():
    section("6. Batch Operations")

    # Eight samples, each containing five features.
    inputs = torch.randn(8, 5)

    # A linear layer maps five input features to three outputs.
    weights = torch.randn(5, 3)
    bias = torch.randn(3)

    outputs = inputs @ weights + bias

    show_shape("Input batch", inputs)
    show_shape("Weights", weights)
    show_shape("Bias", bias)
    show_shape("Output batch", outputs)

    print("\nExpected transformation:")
    print("(8, 5) @ (5, 3) + (3,) -> (8, 3)")

    # Each row represents one sample.
    # Each output row contains three transformed features.


# --------------------------------------------------
# 7. Batched matrix multiplication
# --------------------------------------------------

def batched_matrix_multiplication():
    section("7. Batched Matrix Multiplication")

    # Three separate batches of matrices.
    A = torch.randn(3, 2, 4)
    B = torch.randn(3, 4, 5)

    result = torch.bmm(A, B)

    show_shape("A", A)
    show_shape("B", B)
    show_shape("Result", result)

    print("\nExpected transformation:")
    print("(3, 2, 4) @ (3, 4, 5) -> (3, 2, 5)")

    # torch.bmm requires 3D tensors with matching batch sizes.
    # torch.matmul supports broader broadcasting of batch dimensions.


# --------------------------------------------------
# 8. Common neural network shape errors
# --------------------------------------------------

def neural_network_shape_errors():
    section("8. Neural Network Shape Errors")

    inputs = torch.randn(16, 10)

    # This layer expects 10 input features.
    layer = torch.nn.Linear(in_features=10, out_features=4)

    outputs = layer(inputs)

    show_shape("Inputs", inputs)
    show_shape("Layer weights", layer.weight)
    show_shape("Layer bias", layer.bias)
    show_shape("Outputs", outputs)

    print("\nExpected transformation:")
    print("(16, 10) -> (16, 4)")

    # Deliberately provide the wrong number of input features.
    incorrect_inputs = torch.randn(16, 8)

    try:
        incorrect_outputs = layer(incorrect_inputs)
        print(incorrect_outputs)
    except RuntimeError as error:
        print("\nInput feature mismatch:")
        print(error)

    print("\nFix: ensure the last input dimension matches in_features.")


# --------------------------------------------------
# 9. Main
# --------------------------------------------------

def main():
    inspect_shapes()
    reshape_tensors()
    broadcasting_examples()
    test_shape_compatibility()
    matrix_multiplication()
    batch_operations()
    batched_matrix_multiplication()
    neural_network_shape_errors()


if __name__ == "__main__":
    main()
