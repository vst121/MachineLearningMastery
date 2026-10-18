"""
Gradient Accumulation
=====================

Experiments:
1. Demonstrate gradient accumulation across backward calls.
2. Reset gradients before a new optimization step.
3. Accumulate gradients across mini-batches.
4. Compare summed and averaged gradients.
5. Simulate a larger effective batch using gradient accumulation.
"""

import torch
import torch.nn as nn


def basic_gradient_accumulation():
    """Show that gradients accumulate across backward calls."""
    x = torch.tensor(2.0, requires_grad=True)

    (x**2).backward()
    first_gradient = x.grad.item()

    (3 * x).backward()
    accumulated_gradient = x.grad.item()

    print("Basic Gradient Accumulation")
    print("-" * 40)
    print(f"First gradient:       {first_gradient:.2f}")
    print(f"Accumulated gradient:  {accumulated_gradient:.2f}")

    x.grad.zero_()
    (x**2).backward()

    print(f"After zeroing:         {x.grad.item():.2f}")


def compare_sum_and_average():
    """Compare summed gradients with averaged gradients."""
    gradients = torch.tensor([2.0, 4.0, 6.0, 8.0])

    summed_gradient = gradients.sum()
    averaged_gradient = gradients.mean()

    print("\nSummed vs Averaged Gradients")
    print("-" * 40)
    print(f"Individual gradients: {gradients.tolist()}")
    print(f"Summed gradient:      {summed_gradient.item():.2f}")
    print(f"Averaged gradient:    {averaged_gradient.item():.2f}")


def mini_batch_accumulation():
    """Accumulate gradients over mini-batches before updating parameters."""
    torch.manual_seed(42)

    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)

    criterion = nn.MSELoss()
    learning_rate = 0.1

    mini_batches = [
        (
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([[2.0], [4.0]]),
        ),
        (
            torch.tensor([[3.0], [4.0]]),
            torch.tensor([[6.0], [8.0]]),
        ),
    ]

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    print("\nMini-Batch Gradient Accumulation")
    print("-" * 40)
    print(f"Initial weight: {model.weight.item():.4f}")

    for index, (inputs, targets) in enumerate(mini_batches, start=1):
        predictions = model(inputs)
        loss = criterion(predictions, targets)

        # Scale each mean loss to account for equal-sized mini-batches.
        scaled_loss = loss / len(mini_batches)
        scaled_loss.backward()

        print(
            f"Mini-batch {index}: "
            f"loss={loss.item():.4f}, "
            f"accumulated gradient={model.weight.grad.item():.4f}"
        )

    optimizer.step()

    print(f"Updated weight: {model.weight.item():.4f}")


def compare_with_full_batch():
    """Verify that accumulation matches a full-batch gradient."""
    torch.manual_seed(42)

    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

    accumulated_model = nn.Linear(1, 1, bias=False)
    full_batch_model = nn.Linear(1, 1, bias=False)

    with torch.no_grad():
        accumulated_model.weight.fill_(1.0)
        full_batch_model.weight.fill_(1.0)

    criterion = nn.MSELoss()

    # Accumulate gradients over two mini-batches.
    accumulated_model.zero_grad()

    for start in (0, 2):
        batch_X = X[start:start + 2]
        batch_y = y[start:start + 2]

        batch_loss = criterion(accumulated_model(batch_X), batch_y)
        (batch_loss / 2).backward()

    accumulated_gradient = accumulated_model.weight.grad.detach().clone()

    # Calculate the gradient using the entire batch.
    full_batch_loss = criterion(full_batch_model(X), y)
    full_batch_loss.backward()

    full_batch_gradient = full_batch_model.weight.grad.detach().clone()

    print("\nAccumulation vs Full Batch")
    print("-" * 40)
    print(f"Accumulated gradient: {accumulated_gradient.item():.6f}")
    print(f"Full-batch gradient:  {full_batch_gradient.item():.6f}")
    print(
        "Gradients match:",
        torch.allclose(
            accumulated_gradient,
            full_batch_gradient,
            atol=1e-6,
        ),
    )


if __name__ == "__main__":
    print("PyTorch Version:", torch.__version__)
    print()

    basic_gradient_accumulation()
    compare_sum_and_average()
    mini_batch_accumulation()
    compare_with_full_batch()
