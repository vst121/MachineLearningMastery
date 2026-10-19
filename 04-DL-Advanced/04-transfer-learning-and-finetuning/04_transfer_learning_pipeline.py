"""
Build an end-to-end transfer learning pipeline.

This experiment covers:
1. Loading and preprocessing a real image dataset.
2. Training a frozen pretrained feature extractor.
3. Selectively fine-tuning the final residual block.
4. Evaluating models on a validation dataset.
5. Saving the best model checkpoints and comparing results.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18


DATA_DIR = Path("data/cifar10")
CHECKPOINT_DIR = Path("checkpoints/transfer_learning")
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 3
SEED = 42


def get_device() -> torch.device:
    """Select an available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(
    weights: ResNet18_Weights,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 and create training and validation loaders."""
    preprocessing = weights.transforms()

    dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=preprocessing,
    )

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )

    # The validation subset must not use random training augmentation.
    validation_dataset.dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=preprocessing,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    return train_loader, validation_loader


def build_model(
    num_classes: int,
    mode: str,
    device: torch.device,
) -> nn.Module:
    """Build a pretrained ResNet-18 for the selected training strategy."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for parameter in model.parameters():
        parameter.requires_grad = False

    if mode == "fine_tuning":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    elif mode != "feature_extraction":
        raise ValueError(f"Unsupported training mode: {mode}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def configure_training_mode(model: nn.Module, mode: str) -> None:
    """Keep frozen BatchNorm statistics fixed while training selected layers."""
    model.eval()
    model.fc.train()

    if mode == "fine_tuning":
        model.layer4.train()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Return average cross-entropy loss and accuracy."""
    model.eval()

    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_function(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    mode: str,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Train one transfer learning strategy and save its best checkpoint."""
    model = build_model(NUM_CLASSES, mode, device)

    parameter_groups = []

    if mode == "fine_tuning":
        parameter_groups.append(
            {
                "params": model.layer4.parameters(),
                "lr": 1e-5,
            }
        )

    parameter_groups.append(
        {
            "params": model.fc.parameters(),
            "lr": 1e-3,
        }
    )

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.1,
    )
    loss_function = nn.CrossEntropyLoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{mode}_best.pt"

    best_accuracy = -1.0
    best_validation_loss = float("inf")

    print(f"\n{'=' * 60}\nTraining strategy: {mode}\n{'=' * 60}")

    for epoch in range(EPOCHS):
        configure_training_mode(model, mode)

        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Avoid building a backbone gradient graph during feature extraction.
            if mode == "feature_extraction":
                with torch.no_grad():
                    features = model.conv1(images)
                    features = model.bn1(features)
                    features = model.relu(features)
                    features = model.maxpool(features)
                    features = model.layer1(features)
                    features = model.layer2(features)
                    features = model.layer3(features)
                    features = model.layer4(features)
                    features = model.avgpool(features)
                    features = torch.flatten(features, 1)

                logits = model.fc(features)
            else:
                logits = model(images)

            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

        scheduler.step()

        train_loss = running_loss / total_samples
        train_accuracy = total_correct / total_samples
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.2%} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.2%}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_validation_loss = validation_loss

            torch.save(
                {
                    "mode": mode,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                    "classes": list(train_loader.dataset.dataset.classes),
                },
                checkpoint_path,
            )

    print(f"Best checkpoint: {checkpoint_path}")

    return {
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_validation_loss,
    }


def main() -> None:
    torch.manual_seed(SEED)

    device = get_device()
    weights = ResNet18_Weights.DEFAULT

    print(f"Device: {device}")
    print(f"Dataset directory: {DATA_DIR.resolve()}")

    train_loader, validation_loader = build_dataloaders(weights, device)

    results = {}

    for mode in ("feature_extraction", "fine_tuning"):
        results[mode] = train_model(
            mode,
            train_loader,
            validation_loader,
            device,
        )

    print("\nFinal comparison")
    print("-" * 60)

    for mode, metrics in results.items():
        print(
            f"{mode:20s} | "
            f"best validation accuracy: "
            f"{metrics['best_validation_accuracy']:.2%} | "
            f"validation loss: {metrics['best_validation_loss']:.4f}"
        )

    print("\nTraining complete. Checkpoints are saved for both strategies.")


if __name__ == "__main__":
    main()
