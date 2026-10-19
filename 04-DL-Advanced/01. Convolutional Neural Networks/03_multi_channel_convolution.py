"""Explore multi-channel 2D convolution and trainable parameters.

Experiments:
1. Implement convolution across multiple input channels using NumPy.
2. Apply multiple filters to produce multiple output feature maps.
3. Understand convolution weights and bias terms.
4. Calculate trainable parameter counts.
5. Verify the implementation against PyTorch.
"""

import numpy as np
import torch
import torch.nn.functional as F


def calculate_output_shape(
    input_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
) -> tuple[int, int]:
    """Calculate the spatial dimensions of a 2D convolution output."""
    input_height, input_width = input_shape
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride
    padding_height, padding_width = padding

    if min(input_height, input_width, kernel_height, kernel_width) <= 0:
        raise ValueError("Input and kernel dimensions must be positive.")

    if min(stride_height, stride_width) <= 0:
        raise ValueError("Stride values must be positive.")

    if min(padding_height, padding_width) < 0:
        raise ValueError("Padding values cannot be negative.")

    output_height = (
        input_height + 2 * padding_height - kernel_height
    ) // stride_height + 1

    output_width = (
        input_width + 2 * padding_width - kernel_width
    ) // stride_width + 1

    if output_height <= 0 or output_width <= 0:
        raise ValueError("Kernel configuration produces an invalid output.")

    return output_height, output_width


def count_parameters(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    use_bias: bool = True,
) -> int:
    """Count trainable parameters in a standard convolutional layer."""
    kernel_height, kernel_width = kernel_size

    if min(in_channels, out_channels, kernel_height, kernel_width) <= 0:
        raise ValueError("Channel counts and kernel dimensions must be positive.")

    weights = (
        out_channels
        * in_channels
        * kernel_height
        * kernel_width
    )

    biases = out_channels if use_bias else 0

    return weights + biases


def conv2d_multi_channel_numpy(
    input_tensor: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None = None,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Apply multi-channel 2D cross-correlation using NumPy.

    Args:
        input_tensor: Shape (in_channels, height, width).
        weights: Shape (out_channels, in_channels, kernel_height, kernel_width).
        bias: Optional array with shape (out_channels,).
        stride: Vertical and horizontal stride.
        padding: Vertical and horizontal zero padding.

    Returns:
        Output tensor with shape (out_channels, out_height, out_width).
    """
    if input_tensor.ndim != 3:
        raise ValueError("Input must have shape (channels, height, width).")

    if weights.ndim != 4:
        raise ValueError(
            "Weights must have shape "
            "(out_channels, in_channels, kernel_height, kernel_width)."
        )

    if not np.issubdtype(input_tensor.dtype, np.number):
        raise TypeError("Input must contain numeric values.")

    if not np.issubdtype(weights.dtype, np.number):
        raise TypeError("Weights must contain numeric values.")

    in_channels, input_height, input_width = input_tensor.shape
    out_channels, weight_channels, kernel_height, kernel_width = weights.shape

    if in_channels != weight_channels:
        raise ValueError(
            f"Input has {in_channels} channels, "
            f"but weights expect {weight_channels}."
        )

    if bias is not None and bias.shape != (out_channels,):
        raise ValueError("Bias must have shape (out_channels,).")

    stride_height, stride_width = stride
    padding_height, padding_width = padding

    output_height, output_width = calculate_output_shape(
        input_shape=(input_height, input_width),
        kernel_size=(kernel_height, kernel_width),
        stride=stride,
        padding=padding,
    )

    padded_input = np.pad(
        input_tensor,
        pad_width=(
            (0, 0),
            (padding_height, padding_height),
            (padding_width, padding_width),
        ),
        mode="constant",
    )

    dtype = np.result_type(input_tensor, weights, np.float64)
    output = np.zeros(
        (out_channels, output_height, output_width),
        dtype=dtype,
    )

    if bias is not None:
        output += np.asarray(bias, dtype=dtype)[:, None, None]

    for out_channel in range(out_channels):
        for i in range(output_height):
            for j in range(output_width):
                start_i = i * stride_height
                start_j = j * stride_width

                patch = padded_input[
                    :,
                    start_i : start_i + kernel_height,
                    start_j : start_j + kernel_width,
                ]

                output[out_channel, i, j] += np.sum(
                    patch * weights[out_channel]
                )

    return output


def demonstrate_channel_contributions() -> None:
    """Show how input channels combine to produce one feature map."""
    print("=== Multi-Channel Convolution ===")

    # Two input channels, each with spatial dimensions 3x3.
    input_tensor = np.array(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [9, 8, 7],
                [6, 5, 4],
                [3, 2, 1],
            ],
        ],
        dtype=np.float64,
    )

    # One output filter with two channel-specific 2x2 kernels.
    weights = np.array(
        [
            [
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
            ]
        ],
        dtype=np.float64,
    )

    bias = np.array([0.5], dtype=np.float64)

    output = conv2d_multi_channel_numpy(
        input_tensor,
        weights,
        bias=bias,
    )

    print(f"Input shape:   {input_tensor.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Bias shape:    {bias.shape}")
    print(f"Output shape:  {output.shape}")
    print("Output feature map:")
    print(output[0])


def demonstrate_multiple_filters() -> None:
    """Show how multiple filters produce multiple feature maps."""
    print("\n=== Multiple Filters ===")

    rng = np.random.default_rng(42)

    in_channels = 3
    out_channels = 4
    input_height, input_width = 8, 10
    kernel_size = (3, 3)

    input_tensor = rng.normal(
        size=(in_channels, input_height, input_width)
    )

    weights = rng.normal(
        size=(
            out_channels,
            in_channels,
            kernel_size[0],
            kernel_size[1],
        )
    )

    bias = rng.normal(size=out_channels)

    output = conv2d_multi_channel_numpy(
        input_tensor,
        weights,
        bias=bias,
        stride=(1, 1),
        padding=(1, 1),
    )

    print(f"Input shape:   {input_tensor.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Bias shape:    {bias.shape}")
    print(f"Output shape:  {output.shape}")

    print(
        f"Input channels: {in_channels} | "
        f"Output feature maps: {out_channels}"
    )


def demonstrate_parameter_count() -> None:
    """Calculate weights and biases for a convolutional layer."""
    print("\n=== Trainable Parameter Count ===")

    in_channels = 3
    out_channels = 16
    kernel_size = (3, 3)

    weight_count = (
        in_channels
        * out_channels
        * kernel_size[0]
        * kernel_size[1]
    )

    bias_count = out_channels

    total_with_bias = count_parameters(
        in_channels,
        out_channels,
        kernel_size,
        use_bias=True,
    )

    total_without_bias = count_parameters(
        in_channels,
        out_channels,
        kernel_size,
        use_bias=False,
    )

    print(f"Weight parameters: {weight_count}")
    print(f"Bias parameters:   {bias_count}")
    print(f"Total with bias:   {total_with_bias}")
    print(f"Total without bias: {total_without_bias}")


def verify_against_pytorch() -> None:
    """Verify multi-channel NumPy outputs against PyTorch."""
    print("\n=== PyTorch Verification ===")

    rng = np.random.default_rng(123)

    configurations = [
        {
            "input_shape": (1, 7, 9),
            "out_channels": 2,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (0, 0),
        },
        {
            "input_shape": (3, 8, 10),
            "out_channels": 4,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
        },
        {
            "input_shape": (2, 12, 14),
            "out_channels": 3,
            "kernel_size": (3, 5),
            "stride": (2, 2),
            "padding": (1, 2),
        },
    ]

    for config in configurations:
        input_tensor = rng.normal(size=config["input_shape"])

        in_channels = config["input_shape"][0]
        kernel_height, kernel_width = config["kernel_size"]
        out_channels = config["out_channels"]

        weights = rng.normal(
            size=(
                out_channels,
                in_channels,
                kernel_height,
                kernel_width,
            )
        )
        bias = rng.normal(size=out_channels)

        numpy_output = conv2d_multi_channel_numpy(
            input_tensor,
            weights,
            bias=bias,
            stride=config["stride"],
            padding=config["padding"],
        )

        torch_input = torch.as_tensor(
            input_tensor[None, ...],
            dtype=torch.float64,
        )
        torch_weights = torch.as_tensor(weights, dtype=torch.float64)
        torch_bias = torch.as_tensor(bias, dtype=torch.float64)

        with torch.no_grad():
            torch_output = F.conv2d(
                torch_input,
                torch_weights,
                bias=torch_bias,
                stride=config["stride"],
                padding=config["padding"],
            )

        expected = torch_output[0].numpy()

        np.testing.assert_allclose(
            numpy_output,
            expected,
            rtol=1e-10,
            atol=1e-10,
        )

        print(
            f"Passed: input={config['input_shape']}, "
            f"output={numpy_output.shape}"
        )

    print("All NumPy and PyTorch outputs match.")


def main() -> None:
    demonstrate_channel_contributions()
    demonstrate_multiple_filters()
    demonstrate_parameter_count()
    verify_against_pytorch()

    print("\nAll multi-channel convolution experiments completed successfully.")


if __name__ == "__main__":
    main()
