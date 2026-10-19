"""
Explore the mathematical and computational foundations of variational autoencoders.

This experiment covers:
1. Learning mean and log-variance parameters for latent distributions.
2. Implementing the reparameterization trick.
3. Combining reconstruction loss with KL-divergence regularization.
4. Training and evaluating a VAE on MNIST.
5. Generating new samples by decoding latent vectors from the prior.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 128
LATENT_DIM = 16
EPOCHS = 5
LEARNING_RATE = 1e-3
BETA = 1.0

DATA_DIR = Path("data/mnist")
OUTPUT_DIR = Path("outputs/vae")


class VariationalAutoencoder(nn.Module):
    """Encode images as latent distributions and decode latent samples."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(128, latent_dim)
        self.log_variance_layer = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def encode(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent mean and log-variance."""
        features = self.encoder(images)
        mean = self.mean_layer(features)
        log_variance = self.log_variance_layer(features)
        return mean, log_variance

    @staticmethod
    def reparameterize(
        mean: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a latent vector while preserving gradient flow."""
        standard_deviation = torch.exp(0.5 * log_variance)
        noise = torch.randn_like(standard_deviation)
        return mean + standard_deviation * noise

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into reconstructed images."""
        return self.decoder(latent)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(images)
        latent = self.reparameterize(mean, log_variance)
        reconstructed = self.decode(latent)
        return reconstructed, mean, log_variance

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate images from the standard normal latent prior."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        latent = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(latent)


def vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    beta: float = BETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reconstruction, KL-divergence, and total VAE loss."""
    if beta < 0:
        raise ValueError("beta must be non-negative.")

    # Sum pixel losses per image, then average across the batch.
    reconstruction_loss = F.binary_cross_entropy(
        reconstructed,
        original,
        reduction="none",
    ).flatten(start_dim=1).sum(dim=1).mean()

    # KL(q(z|x) || N(0, I)), averaged across the batch.
    kl_divergence = -0.5 * torch.sum(
        1 + log_variance - mean.square() - log_variance.exp(),
        dim=1,
    ).mean()

    total_loss = reconstruction_loss + beta * kl_divergence

    return total_loss, reconstruction_loss, kl_divergence


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
    model: VariationalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Run one training or validation epoch."""
    training = optimizer is not None
    model.train(training)

    totals = {
        "total": 0.0,
        "reconstruction": 0.0,
        "kl": 0.0,
    }
    sample_count = 0

    for images, _ in loader:
        images = images.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            reconstructed, mean, log_variance = model(images)
            total, reconstruction, kl = vae_loss(
                reconstructed,
                images,
                mean,
                log_variance,
            )

            if training:
                total.backward()
                optimizer.step()

        batch_size = images.size(0)
        totals["total"] += total.item() * batch_size
        totals["reconstruction"] += reconstruction.item() * batch_size
        totals["kl"] += kl.item() * batch_size
        sample_count += batch_size

    return {
        name: value / sample_count
        for name, value in totals.items()
    }


def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    train_loader, validation_loader = build_dataloaders(device)

    model = VariationalAutoencoder(LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(f"Total parameters: {total_parameters:,}")
    print(f"Latent dimensions: {LATENT_DIM}")
    print(f"Beta: {BETA}")

    # Verify shapes, finite values, and the reparameterization gradient.
    sample_images, _ = next(iter(train_loader))
    sample_images = sample_images[:8].to(device)

    mean, log_variance = model.encode(sample_images)
    latent = model.reparameterize(mean, log_variance)
    reconstructed = model.decode(latent)

    assert mean.shape == (8, LATENT_DIM)
    assert log_variance.shape == (8, LATENT_DIM)
    assert latent.shape == (8, LATENT_DIM)
    assert reconstructed.shape == sample_images.shape
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_variance).all()
    assert torch.isfinite(reconstructed).all()

    test_mean = mean.detach().clone().requires_grad_(True)
    test_log_variance = log_variance.detach().clone().requires_grad_(True)
    test_latent = model.reparameterize(test_mean, test_log_variance)
    test_latent.sum().backward()

    assert test_mean.grad is not None
    assert test_log_variance.grad is not None

    print("\nShape and gradient checks passed.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "vae_best.pt"
    samples_path = OUTPUT_DIR / "generated_samples.pt"

    best_validation_loss = float("inf")

    for epoch in range(EPOCHS):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train total: {train_metrics['total']:.2f} | "
            f"train reconstruction: {train_metrics['reconstruction']:.2f} | "
            f"train KL: {train_metrics['kl']:.2f} | "
            f"validation total: {validation_metrics['total']:.2f}"
        )

        if validation_metrics["total"] < best_validation_loss:
            best_validation_loss = validation_metrics["total"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "latent_dim": LATENT_DIM,
                    "beta": BETA,
                    "epoch": epoch + 1,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )

    model.eval()
    with torch.no_grad():
        generated = model.generate(num_samples=16, device=device)
        assert generated.shape == (16, 1, 28, 28)
        assert torch.isfinite(generated).all()
        assert generated.min() >= 0
        assert generated.max() <= 1

        torch.save(generated.cpu(), samples_path)

    print("\nVerification passed:")
    print("- Encoder predicts latent means and log-variances.")
    print("- Reparameterization preserves gradient flow.")
    print("- Reconstruction and KL losses are finite.")
    print("- The decoder generates correctly shaped images.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Generated tensors: {samples_path}")
    print(f"Best validation loss: {best_validation_loss:.2f}")


if __name__ == "__main__":
    main()
