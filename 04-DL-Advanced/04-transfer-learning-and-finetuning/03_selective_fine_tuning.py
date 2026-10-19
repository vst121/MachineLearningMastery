"""
Explore selective fine-tuning of a pretrained convolutional neural network.

This experiment covers:
1. Loading pretrained ResNet-18 weights.
2. Freezing the backbone and selectively unfreezing later layers.
3. Applying discriminative learning rates.
4. Using a learning-rate scheduler.
5. Verifying gradients and parameter updates across model sections.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def snapshot_parameters(module: nn.Module) -> list[torch.Tensor]:
    """Copy parameters for later update verification."""
    return [
        parameter.detach().clone()
        for parameter in module.parameters()
    ]


def parameters_changed(
    before: list[torch.Tensor],
    module: nn.Module,
) -> bool:
    """Return whether at least one parameter changed."""
    after = list(module.parameters())
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, after)
    )


def main() -> None:
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    batch_size = 4

    print(f"Device: {device}")

    # Load pretrained ImageNet weights.
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Freeze every pretrained parameter first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze the final residual block.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    # Replace the original ImageNet classifier.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)

    total, trainable = count_parameters(model)
    print("\nParameter counts")
    print(f"Total: {total:,}")
    print(f"Trainable: {trainable:,}")
    print(f"Frozen: {total - trainable:,}")

    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3."))
    )
    assert all(
        parameter.requires_grad
        for parameter in model.layer4.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.fc.parameters()
    )

    # Keep frozen backbone BatchNorm statistics unchanged.
    # Only layer4 and the classifier use training mode.
    model.eval()
    model.layer4.train()
    model.fc.train()

    # Different learning rates for pretrained and new parameters.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.layer4.parameters(),
                "lr": 1e-5,
            },
            {
                "params": model.fc.parameters(),
                "lr": 1e-3,
            },
        ],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.1,
    )

    loss_function = nn.CrossEntropyLoss()

    # Synthetic inputs keep this experiment independent of a dataset download.
    images = torch.rand(batch_size, 3, 224, 224)
    images = weights.transforms()(images).to(device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    frozen_before = snapshot_parameters(model.layer3)
    layer4_before = snapshot_parameters(model.layer4)
    head_before = snapshot_parameters(model.fc)

    for epoch in range(3):
        optimizer.zero_grad()

        logits = model(images)
        loss = loss_function(logits, labels)

        loss.backward()

        assert all(
            parameter.grad is None
            for parameter in model.layer3.parameters()
        )
        assert all(
            parameter.grad is not None
            for parameter in model.layer4.parameters()
        )
        assert all(
            parameter.grad is not None
            for parameter in model.fc.parameters()
        )

        optimizer.step()
        scheduler.step()

        learning_rates = [
            group["lr"] for group in optimizer.param_groups
        ]
        print(
            f"Epoch {epoch + 1}: loss = {loss.item():.4f}, "
            f"layer4_lr = {learning_rates[0]:.1e}, "
            f"head_lr = {learning_rates[1]:.1e}"
        )

    assert not parameters_changed(frozen_before, model.layer3)
    assert parameters_changed(layer4_before, model.layer4)
    assert parameters_changed(head_before, model.fc)

    print("\nVerification passed:")
    print("- Earlier backbone layers remained frozen.")
    print("- The final residual block was updated.")
    print("- The new classification head was updated.")
    print("- The scheduler adjusted both learning rates.")


if __name__ == "__main__":
    main()
