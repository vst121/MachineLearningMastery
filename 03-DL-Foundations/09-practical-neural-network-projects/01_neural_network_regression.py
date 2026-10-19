"""
Neural Network Regression with PyTorch
======================================

Experiments:
1. Generate a nonlinear regression dataset with noise and feature interactions.
2. Split data into training, validation, and test sets without data leakage.
3. Train a baseline multilayer perceptron (MLP).
4. Train a regularized MLP using dropout and AdamW weight decay.
5. Apply learning-rate scheduling and early stopping based on validation loss.
6. Compare models using MAE, RMSE, and R².
7. Visualize learning curves, predictions, and residuals.
8. Save the trained model and preprocessing objects for inference.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
N_SAMPLES = 3000
N_FEATURES = 8
BATCH_SIZE = 64
MAX_EPOCHS = 250
PATIENCE = 25
LEARNING_RATE = 1e-3

OUTPUT_DIR = Path("artifacts/neural_network_regression")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------

def generate_dataset(
    n_samples: int = N_SAMPLES,
    n_features: int = N_FEATURES,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a nonlinear regression dataset with noisy targets."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, n_features))

    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    y = (
        3.0 * np.sin(x0)
        + 1.5 * x1**2
        - 2.0 * x2
        + 1.2 * x0 * x3
        + 0.8 * np.cos(x1 * x2)
        + rng.normal(0.0, 0.6, size=n_samples)
    )

    return X.astype(np.float32), y.reshape(-1, 1).astype(np.float32)


def prepare_data() -> dict:
    """
    Split data into train, validation, and test sets.

    Fit scalers only on training data to avoid data leakage.
    """
    X, y = generate_dataset()

    # 70% training, 15% validation, 15% test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = x_scaler.transform(X_val).astype(np.float32)
    X_test_scaled = x_scaler.transform(X_test).astype(np.float32)

    y_train_scaled = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val).astype(np.float32)
    y_test_scaled = y_scaler.transform(y_test).astype(np.float32)

    def make_loader(
        features: np.ndarray,
        targets: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(targets),
        )

        generator = torch.Generator().manual_seed(SEED)

        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=0,
        )

    return {
        "train_loader": make_loader(
            X_train_scaled, y_train_scaled, shuffle=True
        ),
        "val_loader": make_loader(
            X_val_scaled, y_val_scaled, shuffle=False
        ),
        "X_test": X_test_scaled,
        "y_test": y_test,
        "y_test_scaled": y_test_scaled,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "n_features": X.shape[1],
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
    }


# ---------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------

class RegressionMLP(nn.Module):
    """Multilayer perceptron for continuous-value prediction."""

    def __init__(
        self,
        n_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


# ---------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one training or validation epoch and return sample-weighted loss."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            predictions = model(features)
            loss = loss_function(predictions, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def train_model(
    model_name: str,
    dropout: float,
    weight_decay: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_features: int,
) -> tuple[nn.Module, dict]:
    """Train a model with a scheduler and validation-based early stopping."""
    # Reset initialization so both experiments start comparably.
    set_seed(SEED)

    model = RegressionMLP(
        n_features=n_features,
        dropout=dropout,
    ).to(DEVICE)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(
            model, train_loader, loss_function, optimizer
        )
        val_loss = run_epoch(
            model, val_loader, loss_function
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(
            optimizer.param_groups[0]["lr"]
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"{model_name:12s} | Epoch {epoch:03d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"{model_name}: early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError(f"{model_name} did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["epochs_trained"] = len(history["train_loss"])

    return model, history


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def predict(
    model: nn.Module,
    X: np.ndarray,
    y_scaler: StandardScaler,
) -> np.ndarray:
    """Predict values in the original target scale."""
    model.eval()

    with torch.no_grad():
        features = torch.from_numpy(X).to(DEVICE)
        predictions_scaled = model(features).cpu().numpy()

    return y_scaler.inverse_transform(predictions_scaled).ravel()


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics in the original target scale."""
    y_true = np.asarray(y_true).ravel()

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_scaler: StandardScaler,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate predictions on the untouched test set."""
    predictions = predict(model, X_test, y_scaler)
    metrics = calculate_metrics(y_test, predictions)

    return predictions, metrics


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def plot_learning_curves(histories: dict[str, dict]) -> None:
    """Plot training and validation loss for both experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(
            epochs, history["train_loss"], label=name
        )
        axes[1].plot(
            epochs, history["val_loss"], label=name
        )

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE Loss (standardized target)")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.suptitle("Neural Network Regression: Learning Curves")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)


def plot_predictions_and_residuals(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    """Compare predicted values and residual distributions."""
    n_models = len(predictions)
    fig, axes = plt.subplots(
        2, n_models, figsize=(6 * n_models, 9), squeeze=False
    )

    for column, (name, y_pred) in enumerate(predictions.items()):
        residuals = y_true.ravel() - y_pred

        axes[0, column].scatter(
            y_true.ravel(), y_pred, alpha=0.5, s=18
        )

        lower = min(y_true.min(), y_pred.min())
        upper = max(y_true.max(), y_pred.max())

        axes[0, column].plot(
            [lower, upper], [lower, upper], linestyle="--"
        )
        axes[0, column].set_title(f"{name}: Actual vs. Predicted")
        axes[0, column].set_xlabel("Actual")
        axes[0, column].set_ylabel("Predicted")
        axes[0, column].grid(True, alpha=0.3)

        axes[1, column].scatter(
            y_pred, residuals, alpha=0.5, s=18
        )
        axes[1, column].axhline(0, linestyle="--")
        axes[1, column].set_title(f"{name}: Residuals")
        axes[1, column].set_xlabel("Predicted")
        axes[1, column].set_ylabel("Actual - Predicted")
        axes[1, column].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predictions_and_residuals.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Persistence and inference
# ---------------------------------------------------------------------

def save_model_artifacts(
    model: nn.Module,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    n_features: int,
) -> None:
    """Save model weights, architecture configuration, and scalers."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "hidden_layers": [128, 64, 32],
        "dropout": 0.15,
    }

    torch.save(
        checkpoint,
        OUTPUT_DIR / "regression_model.pt",
    )

    # Store scaler parameters in the checkpoint for portable inference.
    checkpoint["x_scaler_mean"] = x_scaler.mean_.tolist()
    checkpoint["x_scaler_scale"] = x_scaler.scale_.tolist()
    checkpoint["y_scaler_mean"] = y_scaler.mean_.tolist()
    checkpoint["y_scaler_scale"] = y_scaler.scale_.tolist()

    torch.save(
        checkpoint,
        OUTPUT_DIR / "regression_model_with_preprocessing.pt",
    )


def load_model_for_inference(
    checkpoint_path: Path,
) -> tuple[nn.Module, StandardScaler, StandardScaler]:
    """Reconstruct the model and preprocessing scalers from a checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model = RegressionMLP(
        n_features=checkpoint["n_features"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def reconstruct_scaler(
        mean: list[float],
        scale: list[float],
    ) -> StandardScaler:
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(mean, dtype=np.float64)
        scaler.scale_ = np.asarray(scale, dtype=np.float64)
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = len(mean)
        scaler.n_samples_seen_ = 1
        return scaler

    x_scaler = reconstruct_scaler(
        checkpoint["x_scaler_mean"],
        checkpoint["x_scaler_scale"],
    )
    y_scaler = reconstruct_scaler(
        checkpoint["y_scaler_mean"],
        checkpoint["y_scaler_scale"],
    )

    return model, x_scaler, y_scaler


def demonstrate_saved_model_inference(
    checkpoint_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Reload the saved model and run inference on one test sample."""
    model, x_scaler, y_scaler = load_model_for_inference(
        checkpoint_path
    )

    raw_sample = x_scaler.inverse_transform(X_test[:1])
    scaled_sample = x_scaler.transform(raw_sample).astype(np.float32)

    prediction = predict(model, scaled_sample, y_scaler)[0]
    actual = float(y_test.ravel()[0])

    print("\nSaved-model inference check")
    print(f"Actual target:    {actual:.3f}")
    print(f"Predicted target: {prediction:.3f}")


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    data = prepare_data()

    print("\nDataset split sizes:")
    for split, size in data["split_sizes"].items():
        print(f"  {split.capitalize():10s}: {size}")

    configurations = {
        "Baseline": {
            "dropout": 0.0,
            "weight_decay": 0.0,
        },
        "Regularized": {
            "dropout": 0.15,
            "weight_decay": 1e-3,
        },
    }

    models = {}
    histories = {}
    predictions = {}
    metrics = {}

    for name, config in configurations.items():
        print(f"\n{'=' * 60}")
        print(f"Training {name} model")
        print(f"Configuration: {config}")
        print(f"{'=' * 60}")

        model, history = train_model(
            model_name=name,
            dropout=config["dropout"],
            weight_decay=config["weight_decay"],
            train_loader=data["train_loader"],
            val_loader=data["val_loader"],
            n_features=data["n_features"],
        )

        y_pred, test_metrics = evaluate_model(
            model=model,
            X_test=data["X_test"],
            y_test=data["y_test"],
            y_scaler=data["y_scaler"],
        )

        models[name] = model
        histories[name] = history
        predictions[name] = y_pred
        metrics[name] = test_metrics

    print("\nTest-set results")
    print("-" * 72)
    print(f"{'Model':<15}{'MAE':>12}{'RMSE':>12}{'R²':>12}{'Best epoch':>15}")
    print("-" * 72)

    for name, result in metrics.items():
        print(
            f"{name:<15}"
            f"{result['mae']:>12.4f}"
            f"{result['rmse']:>12.4f}"
            f"{result['r2']:>12.4f}"
            f"{histories[name]['best_epoch']:>15}"
        )

    # Choose the final model using validation loss, not test performance.
    selected_name = min(
        histories,
        key=lambda name: histories[name]["best_val_loss"],
    )
    selected_model = models[selected_name]

    print(f"\nSelected model based on validation loss: {selected_name}")

    plot_learning_curves(histories)
    plot_predictions_and_residuals(
        data["y_test"],
        predictions,
    )

    save_model_artifacts(
        model=selected_model,
        x_scaler=data["x_scaler"],
        y_scaler=data["y_scaler"],
        n_features=data["n_features"],
    )

    report = {
        "device": str(DEVICE),
        "dataset_split_sizes": data["split_sizes"],
        "selected_model": selected_name,
        "test_metrics": metrics,
        "training": {
            name: {
                "best_epoch": history["best_epoch"],
                "best_validation_loss": history["best_val_loss"],
                "epochs_trained": history["epochs_trained"],
            }
            for name, history in histories.items()
        },
    }

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    demonstrate_saved_model_inference(
        OUTPUT_DIR / "regression_model_with_preprocessing.pt",
        data["X_test"],
        data["y_test"],
    )

    print("\nGenerated artifacts:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
