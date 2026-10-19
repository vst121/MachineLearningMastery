"""Explore and validate convolution output dimensions.

Experiments:
1. Calculate output dimensions for 1D and 2D convolutions.
2. Compare square and rectangular inputs and kernels.
3. Explore stride, padding, and dilation.
4. Validate shape calculations against PyTorch.
5. Test edge cases and invalid configurations.
"""

from math import floor

import torch
import torch.nn as nn


def calculate_output_dimension(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Calculate one spatial dimension of a convolution output."""
    if input_size <= 0:
        raise ValueError("Input size must be positive.")
    if kernel_size <= 0:
        raise ValueError("Kernel size must be positive.")
    if stride <= 0:
        raise ValueError("Stride must be positive.")
    if padding < 0:
        raise ValueError("Padding cannot be negative.")
    if dilation <= 0:
        raise ValueError("Dilation must be positive.")

    effective_kernel_size = dilation * (kernel_size - 1) + 1

    return floor(
        (input_size + 2 * padding - effective_kernel_size) / stride
    ) + 1


def calculate_output_shape(
    input_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> tuple[int, int]:
    """Calculate the spatial output shape of a 2D convolution."""
    input_height, input_width = input_shape
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride
    padding_height, padding_width = padding
    dilation_height, dilation_width = dilation

    output_height = calculate_output_dimension(
        input_size=input_height,
        kernel_size=kernel_height,
        stride=stride_height,
        padding=padding_height,
        dilation=dilation_height,
    )

    output_width = calculate_output_dimension(
        input_size=input_width,
        kernel_size=kernel_width,
        stride=stride_width,
        padding=padding_width,
        dilation=dilation_width,
    )

    if output_height <= 0 or output_width <= 0:
        raise ValueError("Configuration produces an invalid output shape.")

    return output_height, output_width


def experiment_basic_dimensions() -> None:
    """Compare common convolution configurations."""
    print("=== Basic Dimension Experiments ===")

    input_size = 32

    configurations = [
        (3, 1, 0, 1),
        (3, 1, 1, 1),
        (3, 2, 0, 1),
        (5, 1, 0, 1),
        (3, 1, 0, 2),
    ]

    print(
        f"{'Input':>6} {'Kernel':>7} {'Stride':>7} "
        f"{'Padding':>8} {'Dilation':>9} {'Output':>7}"
    )

    for kernel, stride, padding, dilation in configurations:
        output = calculate_output_dimension(
            input_size=input_size,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        print(
            f"{input_size:>6} {kernel:>7} {stride:>7} "
            f"{padding:>8} {dilation:>9} {output:>7}"
        )


def experiment_rectangular_shapes() -> None:
    """Explore different height and width configurations."""
    print("\n=== Rectangular Tensor Dimensions ===")

    configurations = [
        {
            "input": (32, 32),
            "kernel": (3, 3),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
        },
        {
            "input": (32, 48),
            "kernel": (3, 5),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
        },
        {
            "input": (28, 40),
            "kernel": (3, 3),
            "stride": (2, 2),
            "padding": (1, 1),
            "dilation": (1, 1),
        },
        {
            "input": (28, 40),
            "kernel": (3, 5),
            "stride": (1, 2),
            "padding": (2, 1),
            "dilation": (2, 1),
        },
    ]

    for config in configurations:
        output_shape = calculate_output_shape(
            input_shape=config["input"],
            kernel_size=config["kernel"],
            stride=config["stride"],
            padding=config["padding"],
            dilation=config["dilation"],
        )

        print(
            f"Input={config['input']}, "
            f"Kernel={config['kernel']}, "
            f"Stride={config['stride']}, "
            f"Padding={config['padding']}, "
            f"Dilation={config['dilation']} "
            f"-> Output={output_shape}"
        )


def verify_against_pytorch() -> None:
    """Compare predicted dimensions with actual PyTorch output shapes."""
    print("\n=== PyTorch Shape Verification ===")

    configurations = [
        {
            "input": (32, 32),
            "kernel": (3, 3),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
        },
        {
            "input": (32, 48),
            "kernel": (3, 5),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
        },
        {
            "input": (28, 40),
            "kernel": (3, 3),
            "stride": (2, 2),
            "padding": (1, 1),
            "dilation": (1, 1),
        },
        {
            "input": (28, 40),
            "kernel": (3, 5),
            "stride": (1, 2),
            "padding": (2, 1),
            "dilation": (2, 1),
        },
    ]

    for config in configurations:
        expected_shape = calculate_output_shape(
            input_shape=config["input"],
            kernel_size=config["kernel"],
            stride=config["stride"],
            padding=config["padding"],
            dilation=config["dilation"],
        )

        input_tensor = torch.randn(
            2, 3, *config["input"]
        )

        layer = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=config["kernel"],
            stride=config["stride"],
            padding=config["padding"],
            dilation=config["dilation"],
        )

        with torch.no_grad():
            output_tensor = layer(input_tensor)

        actual_shape = tuple(output_tensor.shape[-2:])

        assert actual_shape == expected_shape, (
            f"Expected {expected_shape}, got {actual_shape}"
        )

        print(
            f"Passed: input={config['input']}, "
            f"kernel={config['kernel']}, "
            f"predicted={expected_shape}, "
            f"actual={actual_shape}"
        )


def test_edge_cases() -> None:
    """Validate special cases and invalid configurations."""
    print("\n=== Edge Case Tests ===")

    # A 1x1 kernel with stride 1 preserves spatial dimensions.
    assert calculate_output_shape(
        input_shape=(16, 20),
        kernel_size=(1, 1),
    ) == (16, 20)

    # A 3x3 kernel with padding 1 preserves dimensions at stride 1.
    assert calculate_output_shape(
        input_shape=(16, 20),
        kernel_size=(3, 3),
        padding=(1, 1),
    ) == (16, 20)

    # Dilation 2 makes a 3x3 kernel effectively 5x5.
    assert calculate_output_shape(
        input_shape=(10, 10),
        kernel_size=(3, 3),
        dilation=(2, 2),
    ) == (6, 6)

    # A kernel larger than the input cannot produce a valid output.
    try:
        calculate_output_shape(
            input_shape=(3, 3),
            kernel_size=(5, 5),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for an invalid output.")

    # Invalid stride must be rejected.
    try:
        calculate_output_dimension(
            input_size=10,
            kernel_size=3,
            stride=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for stride=0.")

    print("All edge case tests passed.")


def main() -> None:
    experiment_basic_dimensions()
    experiment_rectangular_shapes()
    verify_against_pytorch()
    test_edge_cases()

    print("\nAll convolution dimension experiments completed successfully.")


if __name__ == "__main__":
    main()
