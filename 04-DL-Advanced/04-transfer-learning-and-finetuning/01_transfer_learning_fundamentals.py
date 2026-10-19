"""
Explore the fundamental mechanics of transfer learning.

This experiment covers:
1. Building a pretrained-style feature extractor and classification head.
2. Understanding frozen and trainable parameters.
3. Comparing feature extraction with full fine-tuning.
4. Inspecting gradient flow and optimizer parameter selection.
5. Verifying parameter counts and training behavior.
"""

import torch
import torch.nn as nn


class SmallImageClassifier(nn.Module):
    """Create a compact model to demonstrate transfer learning mechanics."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(12, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        return self.classifier(features)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def set_feature_extraction_mode(model: SmallImageClassifier) -> None:
    """Freeze the backbone and leave the classifier trainable."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def set_fine_tuning_mode(model: SmallImageClassifier) -> None:
    """Allow all model parameters to be updated."""
    for parameter in model.parameters():
        parameter.requires_grad = True


def main() -> None:
    torch.manual_seed(42)

    model = SmallImageClassifier(num_classes=3)

    # This model is randomly initialized to demonstrate the mechanics.
    # In a real transfer learning workflow, load pretrained backbone weights.
    print("=== Initial Model ===")
    total, trainable = count_parameters(model)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    inputs = torch.randn(6, 8)
    labels = torch.randint(0, 3, (6,))
    loss_function = nn.CrossEntropyLoss()

    # 1. Feature extraction: only the classification head is trainable.
    set_feature_extraction_mode(model)

    total, trainable = count_parameters(model)

    print("\n=== Feature Extraction Mode ===")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {total - trainable:,}")

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=1e-3,
    )

    backbone_before = [
        parameter.detach().clone()
        for parameter in model.backbone.parameters()
    ]

    classifier_before = [
        parameter.detach().clone()
        for parameter in model.classifier.parameters()
    ]

    model.train()
    optimizer.zero_grad()

    logits = model(inputs)
    loss = loss_function(logits, labels)
    loss.backward()
    optimizer.step()

    backbone_after = list(model.backbone.parameters())
    classifier_after = list(model.classifier.parameters())

    assert all(parameter.grad is None for parameter in backbone_after)
    assert all(parameter.grad is not None for parameter in classifier_after)

    assert all(
        torch.equal(before, after)
        for before, after in zip(backbone_before, backbone_after)
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(classifier_before, classifier_after)
    )

    print(f"Training loss: {loss.item():.4f}")
    print("Backbone parameters unchanged: True")
    print("Classifier parameters updated: True")

    # 2. Fine-tuning: allow gradients through the entire model.
    set_fine_tuning_mode(model)

    total, trainable = count_parameters(model)

    print("\n=== Fine-Tuning Mode ===")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    assert trainable == total

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    backbone_before = [
        parameter.detach().clone()
        for parameter in model.backbone.parameters()
    ]

    optimizer.zero_grad()
    logits = model(inputs)
    loss = loss_function(logits, labels)
    loss.backward()

    assert all(
        parameter.grad is not None
        for parameter in model.backbone.parameters()
    )

    optimizer.step()

    backbone_after = list(model.backbone.parameters())

    assert any(
        not torch.equal(before, after)
        for before, after in zip(backbone_before, backbone_after)
    )

    print(f"Training loss: {loss.item():.4f}")
    print("Backbone gradients available: True")
    print("Backbone parameters updated: True")

    print("\nAll transfer learning fundamentals checks passed.")


if __name__ == "__main__":
    main()
