"""
PyTorch Autograd
================

Experiments:
1. Calculate gradients using backward().
2. Inspect gradients in a computational graph.
3. Compare autograd with analytical derivatives.
4. Demonstrate gradient accumulation and zeroing.
5. Disable gradient tracking during inference.
"""

import torch


def basic_autograd():
    """Calculate the gradient of a simple function."""
    x = torch.tensor(3.0, requires_grad=True)

    y = x**2 + 2 * x + 1
    y.backward()

    print("Basic Autograd")
    print("-" * 40)
    print(f"x: {x.item()}")
    print(f"y: {y.item()}")
    print(f"dy/dx: {x.grad.item()}")

    # Analytical derivative: dy/dx = 2x + 2
    analytical = 2 * x.item() + 2
    print(f"Analytical gradient: {analytical:.4f}")


def computational_graph():
    """Inspect gradients in a computational graph."""
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    z = x**2 + x * y + y**2
    z.backward()

    print("\nComputational Graph")
    print("-" * 40)
    print(f"z: {z.item()}")
    print(f"dz/dx: {x.grad.item()}")  # 2x + y = 7
    print(f"dz/dy: {y.grad.item()}")  # x + 2y = 8


def tensor_gradient_check():
    """Compare PyTorch gradients with analytical derivatives."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    loss = (x**2).sum()
    loss.backward()

    analytical = 2 * x.detach()

    print("\nTensor Gradient Check")
    print("-" * 40)
    print("Autograd gradients: ", x.grad.tolist())
    print("Analytical gradients:", analytical.tolist())
    print("Gradients match:", torch.allclose(x.grad, analytical))


def gradient_accumulation():
    """Demonstrate gradient accumulation between backward calls."""
    x = torch.tensor(2.0, requires_grad=True)

    first_loss = x**2
    first_loss.backward()
    first_gradient = x.grad.item()

    second_loss = 3 * x
    second_loss.backward()
    accumulated_gradient = x.grad.item()

    x.grad.zero_()

    third_loss = x**2
    third_loss.backward()
    reset_gradient = x.grad.item()

    print("\nGradient Accumulation")
    print("-" * 40)
    print(f"After first backward(): {first_gradient:.1f}")
    print(f"After second backward(): {accumulated_gradient:.1f}")
    print(f"After zeroing and backward(): {reset_gradient:.1f}")


def no_gradient_inference():
    """Disable gradient tracking during inference."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    with torch.no_grad():
        predictions = x**2

    print("\nInference Without Gradient Tracking")
    print("-" * 40)
    print("Predictions:", predictions.tolist())
    print("Requires gradient:", predictions.requires_grad)


def main():
    print("PyTorch Version:", torch.__version__)
    print()

    basic_autograd()
    computational_graph()
    tensor_gradient_check()
    gradient_accumulation()
    no_gradient_inference()


if __name__ == "__main__":
    main()
