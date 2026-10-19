"""
Explore the fundamental mechanics of generative adversarial networks.

This experiment covers:
1. Building a generator and discriminator.
2. Training the discriminator to distinguish real and generated images.
3. Training the generator to fool the discriminator.
4. Implementing the non-saturating GAN objective.
5. Verifying adversarial gradient flow and generated sample shapes.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 128
LATENT_DIM = 64
HIDDEN_DIM = 256
EPOCHS = 5
LEARNING_RATE = 2e-4

DATA_DIR = Path("data/mnist")
OUTPUT_DIR = Path("outputs/gan")


class Generator(nn.Module):
    """Map random latent vectors to synthetic MNIST-like images."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        generated = self.network(latent)
        return generated.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """Estimate whether an image comes from the real dataset."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Return logits, not probabilities. BCEWithLogitsLoss is numerically stable.
        return self.network(images)


def build_dataloader(device: torch.device) -> DataLoader:
    """Load MNIST images normalized to the generator's output range."""
    dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )


def set_requires_grad(model: nn.Module, enabled: bool) -> None:
    """Enable or disable parameter gradients."""
    for parameter in model.parameters():
        parameter.requires_grad = enabled


def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    dataloader = build_dataloader(device)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    adversarial_loss = nn.BCEWithLogitsLoss()

    generator_parameters = sum(
        parameter.numel() for parameter in generator.parameters()
    )
    discriminator_parameters = sum(
        parameter.numel() for parameter in discriminator.parameters()
    )

    print(f"Generator parameters: {generator_parameters:,}")
    print(f"Discriminator parameters: {discriminator_parameters:,}")

    # Verify the generator output and discriminator output dimensions.
    test_noise = torch.randn(8, LATENT_DIM, device=device)

    with torch.no_grad():
        test_images = generator(test_noise)
        test_logits = discriminator(test_images)

    assert test_images.shape == (8, 1, 28, 28)
    assert test_logits.shape == (8, 1)
    assert torch.isfinite(test_images).all()
    assert torch.isfinite(test_logits).all()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "gan_best.pt"
    samples_path = OUTPUT_DIR / "generated_samples.pt"

    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()

        generator_loss_total = 0.0
        discriminator_loss_total = 0.0
        sample_count = 0

        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            # ---------------------------------------------------------
            # 1. Train the discriminator.
            # ---------------------------------------------------------
            set_requires_grad(discriminator, True)
            discriminator_optimizer.zero_grad(set_to_none=True)

            real_logits = discriminator(real_images)
            real_loss = adversarial_loss(real_logits, real_targets)

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            with torch.no_grad():
                fake_images = generator(noise)

            fake_logits = discriminator(fake_images)
            fake_loss = adversarial_loss(fake_logits, fake_targets)

            discriminator_loss = 0.5 * (real_loss + fake_loss)
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # ---------------------------------------------------------
            # 2. Train the generator to fool the discriminator.
            # ---------------------------------------------------------
            set_requires_grad(discriminator, False)
            generator_optimizer.zero_grad(set_to_none=True)

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            generated_images = generator(noise)
            generated_logits = discriminator(generated_images)

            # The generator wants generated images to be classified as real.
            generator_loss = adversarial_loss(
                generated_logits,
                real_targets,
            )

            generator_loss.backward()
            generator_optimizer.step()

            set_requires_grad(discriminator, True)

            generator_loss_total += generator_loss.item() * batch_size
            discriminator_loss_total += discriminator_loss.item() * batch_size
            sample_count += batch_size

        average_generator_loss = generator_loss_total / sample_count
        average_discriminator_loss = discriminator_loss_total / sample_count

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"generator loss: {average_generator_loss:.4f} | "
            f"discriminator loss: {average_discriminator_loss:.4f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "generator_loss": average_generator_loss,
                "discriminator_loss": average_discriminator_loss,
                "latent_dim": LATENT_DIM,
            },
            checkpoint_path,
        )

    # Generate samples from the trained generator.
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(64, LATENT_DIM, device=device)
        generated_images = generator(noise)

        assert generated_images.shape == (64, 1, 28, 28)
        assert torch.isfinite(generated_images).all()
        assert generated_images.min() >= -1
        assert generated_images.max() <= 1

        torch.save(generated_images.cpu(), samples_path)

    print("\nVerification passed:")
    print("- Generator maps latent vectors to image tensors.")
    print("- Discriminator returns one logit per image.")
    print("- Discriminator and generator have separate optimizers.")
    print("- Generated samples have the expected shape and range.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Generated samples: {samples_path}")


if __name__ == "__main__":
    main()
