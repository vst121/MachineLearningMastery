"""Systematically verify convolution operations against PyTorch.

Experiments:
1. Verify single-channel and multi-channel convolutions.
2. Test rectangular kernels, stride, padding, and dilation.
3. Verify configurations with and without bias.
4. Validate output dimensions and parameter counts.
5. Measure numerical differences and report a test summary.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Conv2DConfig:
    """Configuration for one convolution verification case."""

    name: str
    in_channels: int
    out_channels: int
    input_height: int
    input_width: int
    kernel_height: int
    kernel_width: int
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    dilation: tuple[int, int] = (1, 1)
    use_bias: bool = True


def calculate_output_dimension(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Calculate one spatial dimension of a convolution output."""
    if input_size <= 0 or kernel_size <= 0:
        raise ValueError("Input and kernel sizes must be positive.")
    if stride <= 0 or dilation <= 0:
        raise ValueError("Stride and dilation must be positive.")
    if padding < 0:
        raise ValueError("Padding cannot be negative.")

    effective_kernel = dilation * (kernel_size - 1) + 1

    output_size = (
        input_size + 2 * padding - effective_kernel
    ) // stride + 1

    if output_size <= 0:
        raise ValueError("Configuration produces an invalid output.")

    return output_size


def calculate_output_shape(
    config: Conv2DConfig,
) -> tuple[int, int]:
    """Calculate expected output height and width."""
    output_height = calculate_output_dimension(
        config.input_height,
        config.kernel_height,
        config.stride[0],
        config.padding[0],
        config.dilation[0],
    )

    output_width = calculate_output_dimension(
        config.input_width,
        config.kernel_width,
        config.stride[1],
        config.padding[1],
        config.dilation[1],
    )

    return output_height, output_width


def calculate_parameter_count(config: Conv2DConfig) -> int:
    """Calculate the number of convolution parameters."""
    weights = (
        config.out_channels
        * config.in_channels
        * config.kernel_height
        * config.kernel_width
    )

    biases = config.out_channels if config.use_bias else 0

    return weights + biases


def numpy_conv2d(
    input_tensor: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> np.ndarray:
    """Apply multi-channel 2D cross-correlation using NumPy.

    Input shape:   (in_channels, height, width)
    Weights shape: (out_channels, in_channels, kernel_height, kernel_width)
    Output shape:  (out_channels, output_height, output_width)
    """
    if input_tensor.ndim != 3 or weights.ndim != 4:
        raise ValueError("Unexpected input or weight dimensions.")

    in_channels, input_height, input_width = input_tensor.shape
    out_channels, weight_channels, kernel_height, kernel_width = (
        weights.shape
    )

    if in_channels != weight_channels:
        raise ValueError("Input and weight channel counts must match.")

    if bias is not None and bias.shape != (out_channels,):
        raise ValueError("Bias shape must match the output channel count.")

    stride_height, stride_width = stride
    padding_height, padding_width = padding
    dilation_height, dilation_width = dilation

    output_height = calculate_output_dimension(
        input_height,
        kernel_height,
        stride_height,
        padding_height,
        dilation_height,
    )

    output_width = calculate_output_dimension(
        input_width,
        kernel_width,
        stride_width,
        padding_width,
        dilation_width,
    )

    padded_input = np.pad(
        input_tensor,
        (
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
                    start_i : start_i
                    + dilation_height * (kernel_height - 1) + 1
                    : dilation_height,
                    start_j : start_j
                    + dilation_width * (kernel_width - 1) + 1
                    : dilation_width,
                ]

                output[out_channel, i, j] += np.sum(
                    patch * weights[out_channel]
                )

    return output


def verify_case(
    config: Conv2DConfig,
    rng: np.random.Generator,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> dict[str, object]:
    """Verify one configuration against PyTorch."""
    input_tensor = rng.normal(
        size=(
            config.in_channels,
            config.input_height,
            config.input_width,
        )
    )

    weights = rng.normal(
        size=(
            config.out_channels,
            config.in_channels,
            config.kernel_height,
            config.kernel_width,
        )
    )

    bias = (
        rng.normal(size=config.out_channels)
        if config.use_bias
        else None
    )

    numpy_output = numpy_conv2d(
        input_tensor,
        weights,
        bias,
        config.stride,
        config.padding,
        config.dilation,
    )

    torch_input = torch.as_tensor(
        input_tensor[None, ...],
        dtype=torch.float64,
    )

    torch_weights = torch.as_tensor(
        weights,
        dtype=torch.float64,
    )

    torch_bias = (
        torch.as_tensor(bias, dtype=torch.float64)
        if bias is not None
        else None
    )

    with torch.no_grad():
        torch_output = F.conv2d(
            torch_input,
            torch_weights,
            bias=torch_bias,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
        )

    expected_shape = (
        config.out_channels,
        *calculate_output_shape(config),
    )

    if numpy_output.shape != expected_shape:
        raise AssertionError(
            f"{config.name}: NumPy shape {numpy_output.shape}, "
            f"expected {expected_shape}"
        )

    if tuple(torch_output.shape[1:]) != expected_shape:
        raise AssertionError(
            f"{config.name}: PyTorch shape {tuple(torch_output.shape[1:])}, "
            f"expected {expected_shape}"
        )

    torch_result = torch_output[0].numpy()

    np.testing.assert_allclose(
        numpy_output,
        torch_result,
        rtol=rtol,
        atol=atol,
    )

    max_absolute_error = float(
        np.max(np.abs(numpy_output - torch_result))
    )

    torch_layer = torch.nn.Conv2d(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        kernel_size=(config.kernel_height, config.kernel_width),
        stride=config.stride,
        padding=config.padding,
        dilation=config.dilation,
        bias=config.use_bias,
    )

    actual_parameter_count = sum(
        parameter.numel() for parameter in torch_layer.parameters()
    )

    expected_parameter_count = calculate_parameter_count(config)

    if actual_parameter_count != expected_parameter_count:
        raise AssertionError(
            f"{config.name}: parameter count "
            f"{actual_parameter_count} != {expected_parameter_count}"
        )

    return {
        "name": config.name,
        "output_shape": expected_shape,
        "parameters": expected_parameter_count,
        "max_absolute_error": max_absolute_error,
    }


def main() -> None:
    """Run all convolution verification experiments."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    configurations = [
        Conv2DConfig(
            name="Single channel, valid convolution",
            in_channels=1,
            out_channels=1,
            input_height=8,
            input_width=8,
            kernel_height=3,
            kernel_width=3,
        ),
        Conv2DConfig(
            name="RGB input, same spatial dimensions",
            in_channels=3,
            out_channels=8,
            input_height=32,
            input_width=32,
            kernel_height=3,
            kernel_width=3,
            padding=(1, 1),
        ),
        Conv2DConfig(
            name="Rectangular kernel",
            in_channels=2,
            out_channels=4,
            input_height=16,
            input_width=24,
            kernel_height=3,
            kernel_width=5,
            padding=(1, 2),
        ),
        Conv2DConfig(
            name="Strided convolution",
            in_channels=3,
            out_channels=6,
            input_height=31,
            input_width=29,
            kernel_height=3,
            kernel_width=3,
            stride=(2, 2),
            padding=(1, 1),
        ),
        Conv2DConfig(
            name="Dilated convolution",
            in_channels=2,
            out_channels=5,
            input_height=20,
            input_width=18,
            kernel_height=3,
            kernel_width=3,
            dilation=(2, 2),
            padding=(2, 2),
        ),
        Conv2DConfig(
            name="Asymmetric stride and dilation",
            in_channels=2,
            out_channels=3,
            input_height=17,
            input_width=23,
            kernel_height=3,
            kernel_width=5,
            stride=(2, 1),
            padding=(2, 2),
            dilation=(2, 1),
            use_bias=False,
        ),
    ]

    print("=== Convolution Verification Suite ===\n")

    results = []

    for config in configurations:
        result = verify_case(config, rng)
        results.append(result)

        print(f"PASS: {result['name']}")
        print(f"  Output shape: {result['output_shape']}")
        print(f"  Parameters:   {result['parameters']}")
        print(
            "  Maximum absolute error: "
            f"{result['max_absolute_error']:.3e}"
        )

    print("\n=== Summary ===")
    print(f"Configurations passed: {len(results)}/{len(configurations)}")
    print(
        "Largest absolute error: "
        f"{max(r['max_absolute_error'] for r in results):.3e}"
    )
    print("All convolution verification tests passed.")


if __name__ == "__main__":
    main()
