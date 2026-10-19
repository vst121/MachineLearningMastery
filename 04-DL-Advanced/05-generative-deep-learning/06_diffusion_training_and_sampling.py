"""
Train a diffusion noise predictor and generate images with reverse sampling.

This experiment covers:
1. Training a timestep-conditioned denoising model.
2. Computing the DDPM reverse-process mean and variance.
3. Generating images by iteratively removing predicted noise.
4. Saving generated images and a model checkpoint.
5. Validating the sampling process and model outputs.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3

TIMESTEPS = 200
BETA_START = 1e-4
BETA_END = 0.02

HIDDEN_DIM = 256
TIME_DIM = 64
NUM_GENERATED_IMAGES = 64

DATA_DIR = Path("data/mnist")
OUTPUT_DIR = Path("outputs/diffusion_sampling")


class DiffusionSchedule:
    """Store the coefficients needed for forward and reverse diffusion."""

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

        # alpha_bar at the timestep before the first timestep is defined as 1.
        self.alpha_bars_previous = torch.cat(
            [
                torch.ones(1, device=device),
                self.alpha_bars[:-1],
            ]
        )

        self.posterior_variance = (
            self.betas
            * (1.0 - self.alpha_bars_previous)
            / (1.0 - self.alpha_bars)
        )

    def add_noise(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean_images)
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1, 1)

        noisy_images = (
            alpha_bar.sqrt() * clean_images
            + (1.0 - alpha_bar).sqrt() * noise
        )

        return noisy_images, noise


class SinusoidalTimeEmbedding(nn.Module):
    """Encode diffusion timesteps as feature vectors."""

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
        angles = (
            timesteps.float().unsqueeze(1)
            * self.frequencies.unsqueeze(0)
        )

        return torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=1,
        )


class NoisePredictor(nn.Module):
    """Predict the noise in an image at a given diffusion timestep."""

    def __init__(self) -> None:
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(TIME_DIM)

        self.network = nn.Sequential(
            nn.Linear(28 * 28 + TIME_DIM, HIDDEN_DIM * 2),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM * 2),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM * 2, 28 * 28),
        )

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_images.size(0)

        image_features = noisy_images.flatten(start_dim=1)
        time_features = self.time_embedding(timesteps)

        features = torch.cat(
            [image_features, time_features],
            dim=1,
        )

        predicted_noise = self.network(features)

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


def train_model(
    model: NoisePredictor,
    schedule: DiffusionSchedule,
    dataloader: DataLoader,
    device: torch.device,
) -> None:
    """Train the noise predictor with the simplified DDPM objective."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        batches = 0

        for clean_images, _ in dataloader:
            clean_images = clean_images.to(device)
            batch_size = clean_images.size(0)

            timesteps = torch.randint(
                0,
                TIMESTEPS,
                (batch_size,),
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

            total_loss += loss.item()
            batches += 1

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"noise prediction MSE: {total_loss / batches:.6f}"
        )


@torch.no_grad()
def sample_images(
    model: NoisePredictor,
    schedule: DiffusionSchedule,
    num_images: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate images by iteratively reversing the diffusion process."""
    model.eval()

    # Begin with samples from a standard Gaussian distribution.
    images = torch.randn(
        num_images, 1, 28, 28, device=device
    )

    for timestep in reversed(range(schedule.timesteps)):
        timesteps = torch.full(
            (num_images,),
            timestep,
            device=device,
            dtype=torch.long,
        )

        predicted_noise = model(images, timesteps)

        beta_t = schedule.betas[timestep]
        alpha_t = schedule.alphas[timestep]
        alpha_bar_t = schedule.alpha_bars[timestep]

        # Mean of the DDPM reverse transition.
        mean = (
            images
            - beta_t
            / torch.sqrt(1.0 - alpha_bar_t)
            * predicted_noise
        ) / torch.sqrt(alpha_t)

        if timestep > 0:
            variance = schedule.posterior_variance[timestep]
            noise = torch.randn_like(images)
            images = mean + torch.sqrt(variance) * noise
        else:
            # No additional noise at the final reverse step.
            images = mean

    return images.clamp(-1.0, 1.0)


def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataloader = build_dataloader(device)
    schedule = DiffusionSchedule(device=device)
    model = NoisePredictor().to(device)

    train_model(model, schedule, dataloader, device)

    checkpoint_path = OUTPUT_DIR / "diffusion_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "timesteps": TIMESTEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "time_dim": TIME_DIM,
            "hidden_dim": HIDDEN_DIM,
        },
        checkpoint_path,
    )

    generated_images = sample_images(
        model=model,
        schedule=schedule,
        num_images=NUM_GENERATED_IMAGES,
        device=device,
    )

    assert generated_images.shape == (
        NUM_GENERATED_IMAGES, 1, 28, 28
    )
    assert torch.isfinite(generated_images).all()
    assert generated_images.min() >= -1.0
    assert generated_images.max() <= 1.0

    samples_path = OUTPUT_DIR / "generated_samples.png"
    save_image(
        generated_images.cpu(),
        samples_path,
        nrow=8,
        normalize=True,
        value_range=(-1.0, 1.0),
    )

    torch.save(
        generated_images.cpu(),
        OUTPUT_DIR / "generated_samples.pt",
    )

    print("\nVerification passed:")
    print("- Reverse sampling completed.")
    print("- Generated images have the expected shape and range.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Generated image grid: {samples_path}")


if __name__ == "__main__":
    main()
