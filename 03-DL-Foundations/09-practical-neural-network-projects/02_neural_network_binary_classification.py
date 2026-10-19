"""
Neural Network Binary Classification with PyTorch
=================================================

Experiments:
1. Generate a nonlinear binary classification dataset.
2. Create stratified training, validation, and test splits.
3. Standardize features without leaking information from validation or test data.
4. Train an MLP using BCEWithLogitsLoss and AdamW.
5. Compare an unregularized baseline with a regularized model.
6. Apply learning-rate scheduling and early stopping.
7. Evaluate accuracy, precision, recall, F1, ROC-AUC, and confusion matrices.
8. Select a decision threshold using validation data only.
9. Save model weights, preprocessing parameters, and the selected threshold.
10. Reload the model and demonstrate inference.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
N_SAMPLES = 4000
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 1e-3

OUTPUT_DIR = Path("artifacts/neural_network_binary_classification")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------

def prepare_data() -> dict:
    """Generate data and create stratified train, validation, and test splits."""
    X, y = make_moons(
        n_samples=N_SAMPLES,
        noise=0.25,
        random_state=SEED,
    )

    # Add two less-informative features to make the problem more realistic.
    rng = np.random.default_rng(SEED)
    noise_features = rng.normal(size=(N_SAMPLES, 2))
    X = np.column_stack([X, noise_features]).astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=SEED,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    def make_loader(
        features: np.ndarray,
        targets: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(targets).float().reshape(-1, 1),
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
            X_train_scaled, y_train, shuffle=True
        ),
        "val_loader": make_loader(
            X_val_scaled, y_val, shuffle=False
        ),
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler,
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

class BinaryClassificationMLP(nn.Module):
    """MLP that returns raw logits rather than probabilities."""

    def __init__(
        self,
        n_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

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
    """Run one epoch and return sample-weighted binary cross-entropy loss."""
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
            logits = model(features)
            loss = loss_function(logits, targets)

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
    """Train using validation loss, a learning-rate scheduler, and early stopping."""
    set_seed(SEED)

    model = BinaryClassificationMLP(
        n_features=n_features,
        dropout=dropout,
    ).to(DEVICE)

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=6,
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

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 20 == 0:
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
# Prediction and evaluation
# ---------------------------------------------------------------------

def predict_probabilities(
    model: nn.Module,
    X: np.ndarray,
) -> np.ndarray:
    """Convert model logits to positive-class probabilities."""
    model.eval()

    with torch.no_grad():
        features = torch.from_numpy(X).to(DEVICE)
        logits = model(features)
        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy().ravel()


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """
    Select a threshold maximizing validation F1.

    Threshold selection must happen on validation data, never on test data.
    """
    thresholds = np.linspace(0.10, 0.90, 81)

    scores = [
        f1_score(
            y_true,
            (probabilities >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in thresholds
    ]

    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate classification metrics using a fixed decision threshold."""
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def plot_learning_curves(histories: dict[str, dict]) -> None:
    """Visualize training and validation loss."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"], label=name)
        axes[1].plot(epochs, history["val_loss"], label=name)

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("BCE Loss")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> None:
    """Plot a confusion matrix for each model."""
    n_models = len(probabilities)
    fig, axes = plt.subplots(
        1, n_models, figsize=(6 * n_models, 5), squeeze=False
    )

    for column, (name, probs) in enumerate(probabilities.items()):
        predictions = (probs >= thresholds[name]).astype(int)
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

        axis = axes[0, column]
        image = axis.imshow(matrix, interpolation="nearest")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        axis.set_title(f"{name}: Confusion Matrix")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1], labels=["Class 0", "Class 1"])
        axis.set_yticks([0, 1], labels=["Class 0", "Class 1"])

        for row in range(2):
            for col in range(2):
                axis.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> None:
    """Compare ROC curves on the test set."""
    fig, axis = plt.subplots(figsize=(7, 6))

    for name, probs in probabilities.items():
        RocCurveDisplay.from_predictions(
            y_true,
            probs,
            name=name,
            ax=axis,
        )

    axis.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    axis.set_title("ROC Curves on the Test Set")
    axis.grid(True, alpha=0.3)
    axis.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------

def save_model_artifacts(
    model: nn.Module,
    scaler: StandardScaler,
    n_features: int,
    dropout: float,
    threshold: float,
) -> Path:
    """Save model parameters, preprocessing information, and threshold."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = OUTPUT_DIR / "binary_classifier.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "hidden_layers": [64, 32],
        "dropout": dropout,
        "threshold": threshold,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_model_for_inference(
    checkpoint_path: Path,
) -> tuple[nn.Module, StandardScaler, float]:
    """Reload the classifier, scaler, and selected decision threshold."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model = BinaryClassificationMLP(
        n_features=checkpoint["n_features"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.n_samples_seen_ = 1

    return model, scaler, float(checkpoint["threshold"])


def demonstrate_saved_model_inference(
    checkpoint_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Load the saved artifact and predict one held-out test sample."""
    model, scaler, threshold = load_model_for_inference(checkpoint_path)

    # Convert the standardized sample back to raw feature values, then
    # apply the scaler loaded from the checkpoint.
    raw_sample = scaler.inverse_transform(X_test[:1])
    sample_scaled = scaler.transform(raw_sample).astype(np.float32)

    probability = float(
        predict_probabilities(model, sample_scaled)[0]
    )
    prediction = int(probability >= threshold)

    print("\nSaved-model inference check")
    print(f"Actual label:           {int(y_test[0])}")
    print(f"Positive-class prob.:   {probability:.4f}")
    print(f"Decision threshold:     {threshold:.2f}")
    print(f"Predicted label:        {prediction}")


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
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
            "dropout": 0.20,
            "weight_decay": 1e-3,
        },
    }

    models = {}
    histories = {}
    val_probabilities = {}
    test_probabilities = {}
    thresholds = {}
    metrics = {}

    for name, config in configurations.items():
        print(f"\n{'=' * 60}")
        print(f"Training {name} model: {config}")
        print(f"{'=' * 60}")

        model, history = train_model(
            model_name=name,
            dropout=config["dropout"],
            weight_decay=config["weight_decay"],
            train_loader=data["train_loader"],
            val_loader=data["val_loader"],
            n_features=data["n_features"],
        )

        val_probs = predict_probabilities(model, data["X_val"])
        threshold, val_f1 = select_threshold(data["y_val"], val_probs)

        test_probs = predict_probabilities(model, data["X_test"])
        test_metrics = calculate_metrics(
            data["y_test"],
            test_probs,
            threshold,
        )

        models[name] = model
        histories[name] = history
        val_probabilities[name] = val_probs
        test_probabilities[name] = test_probs
        thresholds[name] = threshold
        metrics[name] = test_metrics

        print(
            f"{name}: validation-selected threshold={threshold:.2f}, "
            f"validation F1={val_f1:.4f}"
        )

    print("\nTest-set results")
    print("-" * 88)
    print(
        f"{'Model':<15}{'Threshold':>12}{'Accuracy':>12}"
        f"{'Precision':>12}{'Recall':>12}{'F1':>10}{'ROC-AUC':>12}"
    )
    print("-" * 88)

    for name, result in metrics.items():
        print(
            f"{name:<15}"
            f"{thresholds[name]:>12.2f}"
            f"{result['accuracy']:>12.4f}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['f1']:>10.4f}"
            f"{result['roc_auc']:>12.4f}"
        )

    # Select the final model using validation loss, not test metrics.
    selected_name = min(
        histories,
        key=lambda name: histories[name]["best_val_loss"],
    )
    selected_model = models[selected_name]
    selected_threshold = thresholds[selected_name]

    selected_config = configurations[selected_name]

    checkpoint_path = save_model_artifacts(
        model=selected_model,
        scaler=data["scaler"],
        n_features=data["n_features"],
        dropout=selected_config["dropout"],
        threshold=selected_threshold,
    )

    plot_learning_curves(histories)
    plot_confusion_matrices(
        data["y_test"],
        test_probabilities,
        thresholds,
    )
    plot_roc_curves(
        data["y_test"],
        test_probabilities,
    )

    report = {
        "device": str(DEVICE),
        "split_sizes": data["split_sizes"],
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "models": {
            name: {
                "best_epoch": histories[name]["best_epoch"],
                "best_validation_loss": histories[name]["best_val_loss"],
                "epochs_trained": histories[name]["epochs_trained"],
                "decision_threshold": thresholds[name],
                "test_metrics": metrics[name],
            }
            for name in histories
        },
    }

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    demonstrate_saved_model_inference(
        checkpoint_path,
        data["X_test"],
        data["y_test"],
    )

    print("\nGenerated artifacts:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {path}")


if __name__ == "__main__":
    main()
