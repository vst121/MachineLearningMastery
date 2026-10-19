"""
Explore feature extraction with a pretrained convolutional neural network.

This experiment covers:
1. Loading a pretrained ResNet-18 model.
2. Freezing pretrained backbone parameters.
3. Extracting image representations without gradient computation.
4. Replacing the original classification head.
5. Training and verifying the new classification head.
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


def main() -> None:
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    batch_size = 4

    print(f"Device: {device}")

    # Load a model with pretrained ImageNet weights.
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)

    # Replace the original ImageNet classifier with an identity layer.
    # ResNet-18 now returns its 512-dimensional feature representations.
    backbone.fc = nn.Identity()

    # Freeze all pretrained backbone parameters.
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    backbone = backbone.to(device)
    backbone.eval()

    # Create a new, randomly initialized classification head.
    classifier = nn.Linear(512, num_classes).to(device)

    total, trainable = count_parameters(backbone)
    print("\nPretrained backbone")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    assert trainable == 0

    # Synthetic images demonstrate the pipeline without downloading a dataset.
    # The pretrained model's recommended transform resizes and normalizes inputs.
    images = torch.rand(batch_size, 3, 224, 224)
    images = weights.transforms()(images).to(device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # Extract features without building a gradient graph.
    with torch.no_grad():
        features = backbone(images)

    print(f"\nInput shape: {tuple(images.shape)}")
    print(f"Extracted feature shape: {tuple(features.shape)}")

    assert features.shape == (batch_size, 512)
    assert not features.requires_grad

    # Train only the new classification head.
    classifier.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    head_before = [
        parameter.detach().clone()
        for parameter in classifier.parameters()
    ]

    for epoch in range(3):
        optimizer.zero_grad()

        logits = classifier(features)
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")

    head_after = list(classifier.parameters())

    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert all(parameter.grad is not None for parameter in head_after)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(head_before, head_after)
    )

    print("\nVerification passed:")
    print("- Pretrained backbone parameters remained frozen.")
    print("- Feature extraction did not track gradients.")
    print("- Extracted representations have 512 features per image.")
    print("- The new classification head was updated successfully.")


if __name__ == "__main__":
    main()
