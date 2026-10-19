"""
PyTorch Device Management
=========================

Experiments:
1. Detect available CPU and GPU devices.
2. Inspect tensor and model device placement.
3. Move tensors between CPU and GPU.
4. Move neural network models and inputs to the same device.
5. Diagnose and resolve device mismatch errors.
6. Compare execution times across available devices.
7. Understand synchronization when benchmarking CUDA operations.
"""

import time

import torch
import torch.nn as nn


def section(title):
    print(f"\n{'=' * 65}")
    print(title)
    print("=" * 65)


# --------------------------------------------------
# 1. Detect available devices
# --------------------------------------------------

def inspect_devices():
    section("1. Available Devices")

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Number of CUDA devices:", torch.cuda.device_count())

        for index in range(torch.cuda.device_count()):
            print(
                f"GPU {index}:",
                torch.cuda.get_device_name(index),
            )

    else:
        print("CUDA is unavailable. CPU execution will be used.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Selected device:", device)

    return device


# --------------------------------------------------
# 2. Inspect tensor device placement
# --------------------------------------------------

def inspect_tensor_placement(device):
    section("2. Tensor Device Placement")

    cpu_tensor = torch.randn(3, 4)

    print("CPU tensor device:", cpu_tensor.device)
    print("CPU tensor shape:", cpu_tensor.shape)

    device_tensor = cpu_tensor.to(device)

    print("Selected device tensor:", device_tensor.device)
    print("Tensor shape:", device_tensor.shape)

    # .to(device) returns a tensor on the requested device.
    # The original tensor remains on its original device.
    print("Original tensor device:", cpu_tensor.device)


# --------------------------------------------------
# 3. Move tensors between devices
# --------------------------------------------------

def transfer_tensors(device):
    section("3. Moving Tensors Between Devices")

    tensor = torch.arange(1, 7, dtype=torch.float32)

    print("Original device:", tensor.device)

    device_tensor = tensor.to(device)
    print("After transfer:", device_tensor.device)

    cpu_tensor = device_tensor.to("cpu")
    print("Returned to CPU:", cpu_tensor.device)

    print("Values preserved:", torch.equal(tensor, cpu_tensor))


# --------------------------------------------------
# 4. Move models and inputs to the same device
# --------------------------------------------------

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, inputs):
        return self.network(inputs)


def run_model_on_device(device):
    section("4. Model and Input Device Placement")

    model = SimpleNetwork().to(device)
    inputs = torch.randn(16, 10).to(device)

    print("Model parameter device:", next(model.parameters()).device)
    print("Input device:", inputs.device)

    with torch.no_grad():
        outputs = model(inputs)

    print("Output device:", outputs.device)
    print("Output shape:", outputs.shape)


# --------------------------------------------------
# 5. Demonstrate a device mismatch
# --------------------------------------------------

def demonstrate_device_mismatch(device):
    section("5. Device Mismatch Errors")

    model = nn.Linear(4, 2).to(device)

    # This tensor is explicitly created on the CPU.
    cpu_inputs = torch.randn(3, 4, device="cpu")

    print("Model device:", next(model.parameters()).device)
    print("Input device:", cpu_inputs.device)

    if device.type != "cpu":
        try:
            model(cpu_inputs)
        except RuntimeError as error:
            print("\nExpected device mismatch:")
            print(error)

    # Correct approach: move inputs to the model's device.
    model_device = next(model.parameters()).device
    inputs = cpu_inputs.to(model_device)

    with torch.no_grad():
        outputs = model(inputs)

    print("\nAfter fixing the mismatch:")
    print("Input device:", inputs.device)
    print("Output shape:", outputs.shape)


# --------------------------------------------------
# 6. Benchmark tensor operations
# --------------------------------------------------

def benchmark_device(device, size=1500, repetitions=10):
    """
    Benchmark repeated matrix multiplications.

    CUDA operations are asynchronous, so synchronize before
    measuring elapsed time.
    """
    matrix_a = torch.randn(size, size, device=device)
    matrix_b = torch.randn(size, size, device=device)

    # Warm up the device before measuring.
    for _ in range(2):
        _ = matrix_a @ matrix_b

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()

    for _ in range(repetitions):
        result = matrix_a @ matrix_b

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    # Keep the result alive and verify the operation completed.
    print(
        f"{device}: {repetitions} matrix multiplications "
        f"took {elapsed:.4f} seconds "
        f"(result shape: {tuple(result.shape)})"
    )

    return elapsed


def compare_execution_times():
    section("6. Device Performance Comparison")

    cpu = torch.device("cpu")
    benchmark_device(cpu)

    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        benchmark_device(gpu)
    else:
        print("GPU benchmark skipped because CUDA is unavailable.")

    print(
        "\nBenchmark results depend on hardware, tensor size, "
        "memory transfers, and system load."
    )


# --------------------------------------------------
# 7. Main
# --------------------------------------------------

def main():
    device = inspect_devices()

    inspect_tensor_placement(device)
    transfer_tensors(device)
    run_model_on_device(device)
    demonstrate_device_mismatch(device)
    compare_execution_times()


if __name__ == "__main__":
    main()
