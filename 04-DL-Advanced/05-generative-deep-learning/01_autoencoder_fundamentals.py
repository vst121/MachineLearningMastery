"""
Explore the fundamental mechanics of autoencoders.

This experiment covers:
1. Building an encoder-decoder neural network.
2. Compressing images into low-dimensional latent representations.
3. Optimizing mean squared reconstruction error.
4. Evaluating reconstruction quality on held-out data.
5. Verifying tensor dimensions, gradient flow, and training progress.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 128
LATENT_DIM = 16
EPOCHS = 5
LEARNING_RATE = 1e-3

DATA_DIR = Path("data/mnist")
OUTPUT_DIR = Path("outputs/autoencoder")


class Autoencoder(nn.Module):
    """Encode images into latent vectors and reconstruct them."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Map input images to latent representations."""
        return self.encoder(images)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct images from latent representations."""
        return self.decoder(latent)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))


def build_dataloaders(
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible MNIST training and validation loaders."""
    dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_size = int(0.9 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(SEED),
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


def run_epoch(
    model: Autoencoder,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one training or validation epoch and return average MSE."""
    training = optimizer is not None
    model.train(training)

    total_squared_error = 0.0
    total_pixels = 0

    for images, _ in loader:
        images = images.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            reconstructed = model(images)
            loss = loss_function(reconstructed, images)

            if training:
                loss.backward()
                optimizer.step()

        # Accumulate squared error across all pixels rather than averaging
        # batch-level means, so the metric is correct for uneven batch sizes.
        squared_error = (reconstructed.detach() - images).square().sum().item()
        total_squared_error += squared_error
        total_pixels += images.numel()

    return total_squared_error / total_pixels


def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    train_loader, validation_loader = build_dataloaders(device)

    model = Autoencoder(latent_dim=LATENT_DIM).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print(f"Total parameters: {total_parameters:,}")
    print("Input dimensions: 784")
    print(f"Latent dimensions: {LATENT_DIM}")
    print(f"Input-to-latent dimensionality ratio: {784 / LATENT_DIM:.1f}:1")

    # Validate the model's dimensions and output range.
    sample_images, _ = next(iter(train_loader))
    sample_images = sample_images[:8].to(device)

    with torch.no_grad():
        latent = model.encode(sample_images)
        reconstructed = model.decode(latent)

    assert latent.shape == (8, LATENT_DIM)
    assert reconstructed.shape == sample_images.shape
    assert torch.all(reconstructed >= 0)
    assert torch.all(reconstructed <= 1)

    print(f"Sample input shape: {tuple(sample_images.shape)}")
    print(f"Latent shape: {tuple(latent.shape)}")
    print(f"Reconstructed shape: {tuple(reconstructed.shape)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "autoencoder_best.pt"

    best_validation_loss = float("inf")
    initial_validation_loss = run_epoch(
        model,
        validation_loader,
        loss_function,
        device,
    )

    print(f"\nInitial validation MSE: {initial_validation_loss:.6f}")

    for epoch in range(EPOCHS):
        train_loss = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )

        validation_loss = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train MSE: {train_loss:.6f} | "
            f"validation MSE: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "latent_dim": LATENT_DIM,
                    "epoch": epoch + 1,
                    "validation_mse": validation_loss,
                },
                checkpoint_path,
            )

    # Verify gradient flow through both parts of the network.
    model.zero_grad(set_to_none=True)
    model.eval()

    images, _ = next(iter(validation_loader))
    images = images[:8].to(device)

    reconstructed = model(images)
    final_loss = loss_function(reconstructed, images)
    final_loss.backward()

    assert all(
        parameter.grad is not None
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    )
    assert all(
        parameter.grad is not None
        for parameter in model.decoder.parameters()
        if parameter.requires_grad
    )

    print("\nVerification passed:")
    print("- Latent representations have the expected dimensions.")
    print("- Reconstructed images match the input shape.")
    print("- Decoder outputs lie in the [0, 1] range.")
    print("- Gradients flow through encoder and decoder.")
    print(f"Best validation MSE: {best_validation_loss:.6f}")
    print(f"Checkpoint: {checkpoint_path}")

    if best_validation_loss < initial_validation_loss:
        print("Training reduced the validation reconstruction error.")
    else:
        print(
            "Validation MSE did not improve; consider more epochs, "
            "learning-rate adjustments, or inspecting the training run."
        )


if __name__ == "__main__":
    main()
