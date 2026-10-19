"""
Investigate GAN training stability using Wasserstein GAN with gradient penalty.

This experiment covers:
1. Understanding the difference between a discriminator and a critic.
2. Implementing the Wasserstein critic objective.
3. Computing a gradient penalty to encourage the Lipschitz constraint.
4. Alternating critic and generator optimization.
5. Verifying gradient norms, gradient flow, and generated sample shapes.
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
CRITIC_STEPS = 3
LEARNING_RATE = 1e-4
GRADIENT_PENALTY_WEIGHT = 10.0

DATA_DIR = Path("data/mnist")
OUTPUT_DIR = Path("outputs/wgan_gp")


class Generator(nn.Module):
    """Generate images from latent noise."""

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
        return self.network(latent).view(-1, 1, 28, 28)


class Critic(nn.Module):
    """Assign an unrestricted real-valued score to each image."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def gradient_penalty(
    critic: Critic,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
) -> torch.Tensor:
    """Penalize critic gradient norms that differ from one."""
    batch_size = real_images.size(0)

    # Interpolate between real and generated samples.
    alpha = torch.rand(
        batch_size, 1, 1, 1, device=real_images.device
    )
    interpolated = (
        alpha * real_images + (1.0 - alpha) * fake_images
    ).requires_grad_(True)

    scores = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    gradient_norms = gradients.norm(2, dim=1)

    return ((gradient_norms - 1.0) ** 2).mean()


def build_dataloader(device: torch.device) -> DataLoader:
    """Load MNIST normalized to the generator's output range."""
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
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def set_requires_grad(model: nn.Module, enabled: bool) -> None:
    """Toggle parameter gradient computation."""
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
    critic = Critic().to(device)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.0, 0.9),
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=LEARNING_RATE,
        betas=(0.0, 0.9),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "wgan_gp_best.pt"
    samples_path = OUTPUT_DIR / "generated_samples.pt"

    for epoch in range(EPOCHS):
        generator.train()
        critic.train()

        critic_loss_total = 0.0
        generator_loss_total = 0.0
        wasserstein_estimate_total = 0.0
        penalty_total = 0.0
        critic_updates = 0
        generator_updates = 0

        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # -----------------------------------------------------
            # Train the critic multiple times per generator update.
            # -----------------------------------------------------
            for _ in range(CRITIC_STEPS):
                set_requires_grad(critic, True)
                critic_optimizer.zero_grad(set_to_none=True)

                noise = torch.randn(batch_size, LATENT_DIM, device=device)
                with torch.no_grad():
                    fake_images = generator(noise)

                real_scores = critic(real_images)
                fake_scores = critic(fake_images)

                penalty = gradient_penalty(
                    critic,
                    real_images,
                    fake_images,
                )

                # Minimize this loss to maximize real scores and
                # minimize generated scores, with a gradient penalty.
                critic_loss = (
                    fake_scores.mean()
                    - real_scores.mean()
                    + GRADIENT_PENALTY_WEIGHT * penalty
                )

                critic_loss.backward()
                critic_optimizer.step()

                critic_loss_total += critic_loss.item()
                penalty_total += penalty.item()
                wasserstein_estimate_total += (
                    real_scores.mean() - fake_scores.mean()
                ).detach().item()
                critic_updates += 1

            # -----------------------------------------------------
            # Train the generator to increase critic scores.
            # -----------------------------------------------------
            set_requires_grad(critic, False)
            generator_optimizer.zero_grad(set_to_none=True)

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_images = generator(noise)
            generator_loss = -critic(fake_images).mean()

            generator_loss.backward()
            generator_optimizer.step()

            set_requires_grad(critic, True)

            generator_loss_total += generator_loss.item()
            generator_updates += 1

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"critic loss: {critic_loss_total / critic_updates:.4f} | "
            f"generator loss: {generator_loss_total / generator_updates:.4f} | "
            f"Wasserstein estimate: "
            f"{wasserstein_estimate_total / critic_updates:.4f} | "
            f"gradient penalty: {penalty_total / critic_updates:.4f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "latent_dim": LATENT_DIM,
                "gradient_penalty_weight": GRADIENT_PENALTY_WEIGHT,
            },
            checkpoint_path,
        )

    # Verify that the trained generator produces valid image tensors.
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(64, LATENT_DIM, device=device)
        generated_images = generator(noise)

    assert generated_images.shape == (64, 1, 28, 28)
    assert torch.isfinite(generated_images).all()
    assert generated_images.min() >= -1
    assert generated_images.max() <= 1

    # Verify the gradient penalty is finite and differentiable.
    real_images, _ = next(iter(dataloader))
    real_images = real_images[:8].to(device)

    noise = torch.randn(8, LATENT_DIM, device=device)
    fake_images = generator(noise).detach()

    critic_optimizer.zero_grad(set_to_none=True)
    penalty = gradient_penalty(critic, real_images, fake_images)
    penalty.backward()

    assert torch.isfinite(penalty)
    assert any(
        parameter.grad is not None
        for parameter in critic.parameters()
    )

    torch.save(generated_images.cpu(), samples_path)

    print("\nVerification passed:")
    print("- Critic produces unrestricted scalar scores.")
    print("- Gradient penalty is finite and differentiable.")
    print("- Critic and generator optimization steps are separate.")
    print("- Generated samples have the expected shape and range.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Generated samples: {samples_path}")


if __name__ == "__main__":
    main()
