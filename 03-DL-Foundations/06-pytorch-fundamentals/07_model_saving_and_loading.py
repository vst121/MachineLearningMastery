"""
PyTorch Model Saving and Loading
================================

Experiments:
1. Save trained model weights using state_dict.
2. Recreate a model and load its saved weights.
3. Save and restore optimizer state in a training checkpoint.
4. Resume training from a checkpoint.
5. Save model configuration alongside its weights.
6. Compare predictions before and after loading.
7. Run inference with a restored model.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ARTIFACT_DIR = Path("artifacts")
WEIGHTS_PATH = ARTIFACT_DIR / "moon_classifier_weights.pth"
CHECKPOINT_PATH = ARTIFACT_DIR / "moon_classifier_checkpoint.pth"


class MoonClassifier(nn.Module):
    def __init__(self, input_features: int = 2, hidden_features: int = 32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def prepare_data():
    """Generate data and fit preprocessing only on the training split."""
    features, labels = make_moons(
        n_samples=1500,
        noise=0.25,
        random_state=42,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )

    test_features = torch.tensor(x_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return train_dataset, test_features, test_labels, scaler


def train_one_epoch(model, loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(x_batch)
        loss = loss_function(logits, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def predict(model, features, device):
    """Return probabilities while keeping gradient tracking disabled."""
    model.eval()

    with torch.inference_mode():
        logits = model(features.to(device))
        probabilities = torch.sigmoid(logits)

    return probabilities.cpu()


def save_weights(model):
    """Save only the model parameters and buffers."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Model weights saved to: {WEIGHTS_PATH}")


def load_weights(model, device):
    """Load saved weights into an already-created model."""
    state_dict = torch.load(
        WEIGHTS_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model weights loaded from: {WEIGHTS_PATH}")
    return model


def save_checkpoint(model, optimizer, epoch, loss, model_config):
    """Save enough state to continue a training run."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_config": model_config,
    }

    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Training checkpoint saved to: {CHECKPOINT_PATH}")


def load_checkpoint(model, optimizer, device):
    """Restore model parameters, optimizer state, and training metadata."""
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.to(device)

    print(f"Checkpoint loaded from: {CHECKPOINT_PATH}")
    print(f"Saved epoch: {checkpoint['epoch']}")
    print(f"Saved loss: {checkpoint['loss']:.4f}")
    print(f"Model configuration: {checkpoint['model_config']}")

    return checkpoint


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    train_dataset, test_features, _, _ = prepare_data()

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    model_config = {
        "input_features": 2,
        "hidden_features": 32,
    }

    model = MoonClassifier(**model_config).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\n1. Training the model")
    last_loss = 0.0

    for epoch in range(10):
        last_loss = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:02d} | Loss: {last_loss:.4f}")

    print("\n2. Saving model weights")
    original_probabilities = predict(model, test_features, device)
    save_weights(model)

    print("\n3. Loading weights into a new model")
    restored_model = MoonClassifier(**model_config)
    restored_model = load_weights(restored_model, device)

    restored_probabilities = predict(
        restored_model,
        test_features,
        device,
    )

    max_difference = (
        original_probabilities - restored_probabilities
    ).abs().max().item()

    print(f"Maximum prediction difference: {max_difference:.10f}")
    print(
        "Predictions match:",
        torch.allclose(
            original_probabilities,
            restored_probabilities,
            atol=1e-7,
        ),
    )

    print("\n4. Saving a training checkpoint")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=10,
        loss=last_loss,
        model_config=model_config,
    )

    print("\n5. Restoring checkpoint and continuing training")
    resumed_model = MoonClassifier(**model_config).to(device)
    resumed_optimizer = torch.optim.Adam(
        resumed_model.parameters(),
        lr=0.01,
    )

    checkpoint = load_checkpoint(
        resumed_model,
        resumed_optimizer,
        device,
    )

    next_epoch = checkpoint["epoch"] + 1

    resumed_loss = train_one_epoch(
        resumed_model,
        train_loader,
        loss_function,
        resumed_optimizer,
        device,
    )

    print(
        f"Resumed training: epoch {next_epoch + 1} | "
        f"Loss: {resumed_loss:.4f}"
    )

    print("\n6. Inference using the restored model")
    probabilities = predict(
        resumed_model,
        test_features[:5],
        device,
    )

    for index, probability in enumerate(probabilities.flatten()):
        predicted_class = int(probability >= 0.5)
        print(
            f"Sample {index + 1}: "
            f"probability={probability.item():.4f}, "
            f"class={predicted_class}"
        )


if __name__ == "__main__":
    main()
