"""Explore the fundamentals of 2D convolution operations.

This experiment covers:
1. Manual 2D cross-correlation using NumPy.
2. Kernel movement with stride and padding.
3. Output dimension calculations.
4. Numerical verification against PyTorch.
"""

import numpy as np
import torch
import torch.nn.functional as F


def calculate_output_dimension(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> int:
    """Calculate one spatial output dimension for a 2D convolution."""
    if input_size <= 0:
        raise ValueError("Input size must be positive.")
    if kernel_size <= 0:
        raise ValueError("Kernel size must be positive.")
    if stride <= 0:
        raise ValueError("Stride must be positive.")
    if padding < 0:
        raise ValueError("Padding cannot be negative.")

    return (input_size + 2 * padding - kernel_size) // stride + 1


def conv2d_numpy(
    input_matrix: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Apply single-channel 2D cross-correlation using NumPy."""
    if input_matrix.ndim != 2 or kernel.ndim != 2:
        raise ValueError("Input and kernel must be 2D arrays.")

    if stride <= 0 or padding < 0:
        raise ValueError("Stride must be positive and padding non-negative.")

    if not np.issubdtype(input_matrix.dtype, np.number):
        raise TypeError("Input must contain numeric values.")
    if not np.issubdtype(kernel.dtype, np.number):
        raise TypeError("Kernel must contain numeric values.")

    padded_input = np.pad(
        input_matrix,
        pad_width=padding,
        mode="constant",
    )

    input_height, input_width = padded_input.shape
    kernel_height, kernel_width = kernel.shape

    output_height = calculate_output_dimension(
        input_height, kernel_height, stride
    )
    output_width = calculate_output_dimension(
        input_width, kernel_width, stride
    )

    if output_height <= 0 or output_width <= 0:
        raise ValueError("Kernel is larger than the padded input.")

    output = np.zeros(
        (output_height, output_width),
        dtype=np.result_type(input_matrix, kernel, np.float64),
    )

    for i in range(output_height):
        for j in range(output_width):
            start_i = i * stride
            start_j = j * stride

            patch = padded_input[
                start_i : start_i + kernel_height,
                start_j : start_j + kernel_width,
            ]

            output[i, j] = np.sum(patch * kernel)

    return output


def demonstrate_manual_convolution() -> tuple[np.ndarray, np.ndarray]:
    """Demonstrate a small convolution with a hand-checkable result."""
    input_matrix = np.array(
        [
            [1, 2, 3, 0],
            [4, 5, 6, 1],
            [7, 8, 9, 2],
            [1, 2, 3, 4],
        ],
        dtype=np.float64,
    )

    kernel = np.array(
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ],
        dtype=np.float64,
    )

    output = conv2d_numpy(input_matrix, kernel)

    print("=== Manual 2D Cross-Correlation ===")
    print(f"Input shape:  {input_matrix.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: {output.shape}")
    print("Output:")
    print(output)

    return input_matrix, kernel


def demonstrate_output_dimensions() -> None:
    """Compare output dimensions for different configurations."""
    print("\n=== Output Dimension Experiments ===")

    input_size = 32
    configurations = [
        {"kernel": 3, "stride": 1, "padding": 0},
        {"kernel": 3, "stride": 1, "padding": 1},
        {"kernel": 3, "stride": 2, "padding": 0},
        {"kernel": 5, "stride": 1, "padding": 0},
    ]

    for config in configurations:
        output_size = calculate_output_dimension(
            input_size=input_size,
            kernel_size=config["kernel"],
            stride=config["stride"],
            padding=config["padding"],
        )

        print(
            f"Input={input_size}, "
            f"Kernel={config['kernel']}, "
            f"Stride={config['stride']}, "
            f"Padding={config['padding']} "
            f"-> Output={output_size}"
        )


def demonstrate_stride_and_padding() -> None:
    """Show how stride and padding affect the feature map."""
    input_matrix = np.arange(1, 17, dtype=np.float64).reshape(4, 4)

    kernel = np.ones((3, 3), dtype=np.float64)

    print("\n=== Stride and Padding ===")
    print("Input:")
    print(input_matrix)

    for stride, padding in [(1, 0), (2, 0), (1, 1)]:
        output = conv2d_numpy(
            input_matrix,
            kernel,
            stride=stride,
            padding=padding,
        )

        print(f"\nStride={stride}, Padding={padding}")
        print(f"Output shape: {output.shape}")
        print(output)


def verify_against_pytorch(
    input_matrix: np.ndarray,
    kernel: np.ndarray,
) -> None:
    """Verify the NumPy implementation against PyTorch."""
    print("\n=== PyTorch Verification ===")

    input_tensor = torch.as_tensor(
        input_matrix, dtype=torch.float64
    )[None, None, :, :]

    kernel_tensor = torch.as_tensor(
        kernel, dtype=torch.float64
    )[None, None, :, :]

    configurations = [
        {"stride": 1, "padding": 0},
        {"stride": 1, "padding": 1},
        {"stride": 2, "padding": 0},
    ]

    for config in configurations:
        stride = config["stride"]
        padding = config["padding"]

        numpy_output = conv2d_numpy(
            input_matrix,
            kernel,
            stride=stride,
            padding=padding,
        )

        torch_output = F.conv2d(
            input_tensor,
            kernel_tensor,
            stride=stride,
            padding=padding,
        )

        torch_result = torch_output[0, 0].numpy()

        np.testing.assert_allclose(
            numpy_output,
            torch_result,
            rtol=1e-10,
            atol=1e-10,
        )

        print(
            f"Passed: stride={stride}, padding={padding}, "
            f"output_shape={numpy_output.shape}"
        )


def main() -> None:
    input_matrix, kernel = demonstrate_manual_convolution()
    demonstrate_output_dimensions()
    demonstrate_stride_and_padding()
    verify_against_pytorch(input_matrix, kernel)

    print("\nAll convolution experiments completed successfully.")


if __name__ == "__main__":
    main()
