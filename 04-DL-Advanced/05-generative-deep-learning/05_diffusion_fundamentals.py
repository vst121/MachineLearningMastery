"""
Explore the forward diffusion process and train a noise-prediction model.

This experiment covers:
1. Constructing a linear diffusion noise schedule.
2. Adding noise to images at arbitrary timesteps.
3. Training a neural network to predict the added noise.
4. Understanding the simplified DDPM training objective.
5. Validating tensor shapes, noise schedules, and gradient flow.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3

TIMESTEPS = 200
BETA_START = 1e-4
BETA_END = 0.02

HIDDEN_DIM = 256
DATA_DIR = Path("data/mnist")
OUTPUT_DIR = Path("outputs/diffusion")


class DiffusionSchedule:
    """Precompute coefficients for the forward diffusion process."""

    def __init__(
        self,
        timesteps: int = TIMESTEPS,
        beta_start: float = BETA_START,
        beta_end: float = BETA_END,
        device: torch.device | None = None,
    ) -> None:
        self.timesteps = timesteps

        self.betas = torch.linspace(
            beta_start, beta_end, timesteps, device=device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        assert torch.all(self.betas > 0)
        assert torch.all(self.betas < 1)
        assert torch.all(self.alpha_bars[1:] <= self.alpha_bars[:-1])

    def add_noise(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample noisy images directly from the closed-form distribution."""
        if noise is None:
            noise = torch.randn_like(clean_images)

        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1, 1)

        noisy_images = (
            alpha_bar.sqrt() * clean_images
            + (1.0 - alpha_bar).sqrt() * noise
        )

        return noisy_images, noise


class SinusoidalTimeEmbedding(nn.Module):
    """Encode integer diffusion timesteps as continuous feature vectors."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even")

        half_dim = embedding_dim // 2
        frequencies = torch.exp(
            -torch.log(torch.tensor(10000.0))
            * torch.arange(half_dim, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        angles = timesteps.float().unsqueeze(1) * self.frequencies.unsqueeze(0)

        return torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=1,
        )


class NoisePredictor(nn.Module):
    """Predict injected noise from a noisy image and its timestep."""

    def __init__(
        self,
        image_dim: int = 28 * 28,
        hidden_dim: int = HIDDEN_DIM,
        time_dim: int = 64,
    ) -> None:
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        self.network = nn.Sequential(
            nn.Linear(image_dim + time_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, image_dim),
        )

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_images.size(0)

        image_features = noisy_images.flatten(start_dim=1)
        time_features = self.time_embedding(timesteps)

        combined = torch.cat(
            [image_features, time_features],
            dim=1,
        )

        predicted_noise = self.network(combined)

        return predicted_noise.view(batch_size, 1, 28, 28)


def build_dataloader(device: torch.device) -> DataLoader:
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


def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    dataloader = build_dataloader(device)

    schedule = DiffusionSchedule(device=device)
    model = NoisePredictor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "diffusion_noise_predictor.pt"

    for epoch in range(EPOCHS):
        model.train()
        loss_total = 0.0
        batches = 0

        for clean_images, _ in dataloader:
            clean_images = clean_images.to(device)
            batch_size = clean_images.size(0)

            # Each image receives a randomly sampled diffusion timestep.
            timesteps = torch.randint(
                low=0,
                high=TIMESTEPS,
                size=(batch_size,),
                device=device,
            )

            noisy_images, true_noise = schedule.add_noise(
                clean_images,
                timesteps,
            )

            predicted_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(predicted_noise, true_noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            loss_total += loss.item()
            batches += 1

        average_loss = loss_total / batches
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"noise prediction MSE: {average_loss:.6f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "timesteps": TIMESTEPS,
                "betas": schedule.betas.cpu(),
                "alphas": schedule.alphas.cpu(),
                "alpha_bars": schedule.alpha_bars.cpu(),
            },
            checkpoint_path,
        )

    # -------------------------------------------------------------
    # Verify the forward process and trained model.
    # -------------------------------------------------------------
    model.eval()

    clean_images, _ = next(iter(dataloader))
    clean_images = clean_images[:8].to(device)

    fixed_timesteps = torch.tensor(
        [0, 10, 25, 50, 75, 100, 150, 199],
        device=device,
    )

    noisy_images, true_noise = schedule.add_noise(
        clean_images,
        fixed_timesteps,
    )

    with torch.no_grad():
        predicted_noise = model(noisy_images, fixed_timesteps)

    assert noisy_images.shape == clean_images.shape
    assert predicted_noise.shape == true_noise.shape
    assert torch.isfinite(noisy_images).all()
    assert torch.isfinite(predicted_noise).all()

    torch.save(
        {
            "clean_images": clean_images.cpu(),
            "noisy_images": noisy_images.cpu(),
            "timesteps": fixed_timesteps.cpu(),
            "true_noise": true_noise.cpu(),
            "predicted_noise": predicted_noise.cpu(),
        },
        OUTPUT_DIR / "forward_process_examples.pt",
    )

    print("\nVerification passed:")
    print("- Noisy images have the expected shape.")
    print("- Noise predictions are finite.")
    print("- The model accepts both images and timesteps.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Examples: {OUTPUT_DIR / 'forward_process_examples.pt'}")


if __name__ == "__main__":
    main()
