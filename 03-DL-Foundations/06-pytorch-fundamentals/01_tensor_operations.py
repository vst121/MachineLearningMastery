"""
PyTorch Tensor Operations
=========================

Experiments:
1. Create tensors from lists, NumPy arrays, and built-in functions.
2. Inspect tensor shapes, dimensions, data types, and device placement.
3. Perform indexing, slicing, and element selection.
4. Reshape, flatten, transpose, and combine tensors.
5. Apply element-wise arithmetic and matrix multiplication.
6. Explore broadcasting rules and tensor compatibility.
7. Understand tensor copying, views, and shared memory.
8. Convert tensors between PyTorch and NumPy.
"""

import numpy as np
import torch


def section(title):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def main():
    # --------------------------------------------------
    # 1. Tensor creation
    # --------------------------------------------------

    section("1. Tensor Creation")

    tensor_from_list = torch.tensor([1, 2, 3, 4])
    tensor_zeros = torch.zeros(2, 3)
    tensor_ones = torch.ones(2, 3)
    tensor_random = torch.rand(2, 3)
    tensor_range = torch.arange(0, 10, 2)

    print("From list:\n", tensor_from_list)
    print("Zeros:\n", tensor_zeros)
    print("Ones:\n", tensor_ones)
    print("Random values:\n", tensor_random)
    print("Range:", tensor_range)

    # --------------------------------------------------
    # 2. Tensor properties
    # --------------------------------------------------

    section("2. Tensor Properties")

    tensor = torch.randn(3, 4)

    print("Tensor:\n", tensor)
    print("Shape:", tensor.shape)
    print("Dimensions:", tensor.ndim)
    print("Number of elements:", tensor.numel())
    print("Data type:", tensor.dtype)
    print("Device:", tensor.device)
    print("Requires gradients:", tensor.requires_grad)

    # --------------------------------------------------
    # 3. Indexing and slicing
    # --------------------------------------------------

    section("3. Indexing and Slicing")

    tensor = torch.arange(1, 13).reshape(3, 4)

    print("Original tensor:\n", tensor)
    print("First row:", tensor[0])
    print("Second column:", tensor[:, 1])
    print("Element at row 1, column 2:", tensor[1, 2])
    print("First two rows and three columns:\n", tensor[:2, :3])

    # --------------------------------------------------
    # 4. Reshaping and tensor transformations
    # --------------------------------------------------

    section("4. Reshaping and Transformations")

    tensor = torch.arange(1, 13)

    matrix = tensor.reshape(3, 4)
    flattened = matrix.flatten()
    transposed = matrix.T
    unsqueezed = tensor.unsqueeze(0)
    squeezed = unsqueezed.squeeze(0)

    print("Original:", tensor)
    print("Reshaped to 3x4:\n", matrix)
    print("Flattened:", flattened)
    print("Transposed:\n", transposed)
    print("Unsqueezed shape:", unsqueezed.shape)
    print("Squeezed shape:", squeezed.shape)

    # reshape() can return a view or a copy depending on
    # the tensor's memory layout.
    print("Original shape:", tensor.shape)
    print("Matrix shape:", matrix.shape)

    # --------------------------------------------------
    # 5. Element-wise operations
    # --------------------------------------------------

    section("5. Element-Wise Operations")

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print("a + b:", a + b)
    print("a - b:", a - b)
    print("a * b:", a * b)
    print("a / b:", a / b)
    print("a squared:", a**2)
    print("Exponential:", torch.exp(a))
    print("Mean:", torch.mean(a))
    print("Sum:", torch.sum(a))

    # --------------------------------------------------
    # 6. Matrix multiplication
    # --------------------------------------------------

    section("6. Matrix Multiplication")

    A = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    B = torch.tensor([
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ])

    elementwise_result = A * A
    matrix_result = A @ B

    print("A shape:", A.shape)
    print("B shape:", B.shape)
    print("Element-wise multiplication:\n", elementwise_result)
    print("Matrix multiplication:\n", matrix_result)
    print("Result shape:", matrix_result.shape)

    # --------------------------------------------------
    # 7. Broadcasting
    # --------------------------------------------------

    section("7. Broadcasting")

    matrix = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    row = torch.tensor([10.0, 20.0, 30.0])
    column = torch.tensor([[100.0], [200.0]])

    print("Matrix:\n", matrix)
    print("Row:\n", row)
    print("Matrix + row:\n", matrix + row)
    print("Column:\n", column)
    print("Matrix + column:\n", matrix + column)

    # Broadcasting allows compatible dimensions to expand
    # without explicitly copying the smaller tensor.
    print("Matrix shape:", matrix.shape)
    print("Row shape:", row.shape)
    print("Column shape:", column.shape)

    # --------------------------------------------------
    # 8. Views, copies, and memory sharing
    # --------------------------------------------------

    section("8. Views and Copies")

    original = torch.tensor([1.0, 2.0, 3.0, 4.0])

    view = original.view(2, 2)
    independent_copy = original.clone()

    view[0, 0] = 99.0

    print("Original after modifying view:", original)
    print("View:\n", view)
    print("Independent copy:", independent_copy)

    # A view shares the original tensor's storage when possible.
    # clone() creates independent tensor storage.
    independent_copy[0] = -1.0

    print("Original after modifying copy:", original)
    print("Modified copy:", independent_copy)

    # --------------------------------------------------
    # 9. PyTorch and NumPy interoperability
    # --------------------------------------------------

    section("9. PyTorch and NumPy")

    numpy_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    tensor_from_numpy = torch.from_numpy(numpy_array)
    tensor_to_numpy = tensor_from_numpy.numpy()

    print("NumPy array:", numpy_array)
    print("PyTorch tensor:", tensor_from_numpy)
    print("Converted NumPy array:", tensor_to_numpy)

    # from_numpy() shares memory with the NumPy array.
    tensor_from_numpy[0] = 100.0

    print("NumPy array after tensor modification:", numpy_array)

    # Use clone() to create independent tensor storage.
    independent_tensor = torch.from_numpy(numpy_array).clone()
    independent_tensor[0] = -100.0

    print("Original NumPy array:", numpy_array)
    print("Independent tensor:", independent_tensor)

    # --------------------------------------------------
    # 10. Data types and explicit conversion
    # --------------------------------------------------

    section("10. Data Types")

    integer_tensor = torch.tensor([1, 2, 3])
    floating_tensor = integer_tensor.to(torch.float32)
    double_tensor = floating_tensor.to(torch.float64)

    print("Integer tensor:", integer_tensor.dtype)
    print("Float tensor:", floating_tensor.dtype)
    print("Double tensor:", double_tensor.dtype)

    # Neural network inputs generally use floating-point types.
    model_input = torch.tensor(
        [[1, 2], [3, 4]],
        dtype=torch.float32,
    )

    print("Model input:\n", model_input)
    print("Model input dtype:", model_input.dtype)

    # --------------------------------------------------
    # 11. Common shape mismatch
    # --------------------------------------------------

    section("11. Understanding Shape Mismatches")

    left = torch.randn(2, 3)
    right = torch.randn(4, 2)

    print("Left shape:", left.shape)
    print("Right shape:", right.shape)

    try:
        result = left + right
        print(result)
    except RuntimeError as error:
        print("Addition failed because the shapes are incompatible.")
        print("Error:", error)

    # Matrix multiplication is valid when the inner dimensions match.
    compatible_right = torch.randn(3, 4)
    result = left @ compatible_right

    print("Compatible matrix multiplication shape:", result.shape)


if __name__ == "__main__":
    main()
