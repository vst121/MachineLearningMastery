"""
Neural Network Multiclass Classification with PyTorch
=====================================================

Experiments:
1. Generate a nonlinear multiclass classification dataset.
2. Create stratified training, validation, and test splits.
3. Standardize features using training data only.
4. Train an MLP using CrossEntropyLoss and AdamW.
5. Compare baseline and regularized neural networks.
6. Apply learning-rate scheduling and early stopping.
7. Evaluate accuracy, macro F1, weighted F1, and per-class metrics.
8. Inspect a confusion matrix and one-vs-rest ROC-AUC.
9. Select the final model using validation loss.
10. Save model weights, preprocessing parameters, and class labels.
11. Reload the saved model and run inference.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
N_SAMPLES = 5000
N_CLASSES = 4
N_FEATURES = 12
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 1e-3

CLASS_NAMES = [
    "Class A",
    "Class B",
    "Class C",
    "Class D",
]

OUTPUT_DIR = Path("artifacts/neural_network_multiclass_classification")
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
    """Generate multiclass data and create stratified dataset splits."""
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=8,
        n_redundant=2,
        n_repeated=0,
        n_classes=N_CLASSES,
        n_clusters_per_class=2,
        weights=[0.50, 0.25, 0.15, 0.10],
        class_sep=1.2,
        flip_y=0.02,
        random_state=SEED,
    )

    X = X.astype(np.float32)
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
            torch.from_numpy(targets).long(),
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
        "n_classes": len(np.unique(y)),
        "class_names": CLASS_NAMES,
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "class_distribution": {
            "train": np.bincount(y_train, minlength=N_CLASSES).tolist(),
            "validation": np.bincount(y_val, minlength=N_CLASSES).tolist(),
            "test": np.bincount(y_test, minlength=N_CLASSES).tolist(),
        },
    }


# ---------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------

class MulticlassClassificationMLP(nn.Module):
    """MLP returning one raw logit per class."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
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

            nn.Linear(32, n_classes),
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
    """Run one epoch and return sample-weighted cross-entropy loss."""
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
    n_classes: int,
) -> tuple[nn.Module, dict]:
    """Train with validation-based checkpointing and early stopping."""
    set_seed(SEED)

    model = MulticlassClassificationMLP(
        n_features=n_features,
        n_classes=n_classes,
        dropout=dropout,
    ).to(DEVICE)

    loss_function = nn.CrossEntropyLoss()

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
    """Convert class logits into probability distributions."""
    model.eval()

    with torch.no_grad():
        features = torch.from_numpy(X).to(DEVICE)
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)

    return probabilities.cpu().numpy()


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_classes: int,
) -> dict:
    """Calculate overall and class-sensitive classification metrics."""
    predictions = probabilities.argmax(axis=1)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "roc_auc_ovr_macro": float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=np.arange(n_classes),
                multi_class="ovr",
                average="macro",
            )
        ),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=np.arange(n_classes),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=np.arange(n_classes),
        ).tolist(),
    }


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def plot_learning_curves(histories: dict[str, dict]) -> None:
    """Plot training and validation loss."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=name)
        axes[1].plot(epochs, history["val_loss"], label=name)

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-Entropy Loss")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    filename: str = "confusion_matrix.png",
) -> None:
    """Plot a confusion matrix with actual class labels."""
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=np.arange(len(class_names)),
    )

    fig, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=axis)

    axis.set_title("Multiclass Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(
        np.arange(len(class_names)),
        labels=class_names,
        rotation=35,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(class_names)),
        labels=class_names,
    )

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def plot_per_class_f1(
    metrics: dict,
    class_names: list[str],
) -> None:
    """Visualize the F1-score of each class."""
    report = metrics["classification_report"]
    scores = [report[name]["f1-score"] for name in class_names]

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(class_names, scores)
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1-score")
    axis.set_title("Per-Class F1 on the Test Set")
    axis.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_class_f1.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------

def save_model_artifacts(
    model: nn.Module,
    scaler: StandardScaler,
    n_features: int,
    n_classes: int,
    dropout: float,
    class_names: list[str],
) -> Path:
    """Save model parameters, class labels, and preprocessing information."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = OUTPUT_DIR / "multiclass_classifier.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "n_classes": n_classes,
        "hidden_layers": [128, 64, 32],
        "dropout": dropout,
        "class_names": class_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_model_for_inference(
    checkpoint_path: Path,
) -> tuple[nn.Module, StandardScaler, list[str]]:
    """Reload the model, scaler, and class names."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model = MulticlassClassificationMLP(
        n_features=checkpoint["n_features"],
        n_classes=checkpoint["n_classes"],
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

    return model, scaler, checkpoint["class_names"]


def demonstrate_saved_model_inference(
    checkpoint_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Reload the saved artifact and predict one test sample."""
    model, scaler, class_names = load_model_for_inference(checkpoint_path)

    raw_sample = scaler.inverse_transform(X_test[:1])
    sample_scaled = scaler.transform(raw_sample).astype(np.float32)

    probabilities = predict_probabilities(model, sample_scaled)[0]
    predicted_index = int(np.argmax(probabilities))

    print("\nSaved-model inference check")
    print(f"Actual class:      {class_names[int(y_test[0])]}")
    print(f"Predicted class:   {class_names[predicted_index]}")
    print(f"Confidence:        {probabilities[predicted_index]:.4f}")

    print("Class probabilities:")
    for name, probability in zip(class_names, probabilities):
        print(f"  {name}: {probability:.4f}")


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

    print("\nClass counts by split:")
    for split, counts in data["class_distribution"].items():
        print(f"  {split.capitalize():10s}: {counts}")

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
    probabilities = {}
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
            n_classes=data["n_classes"],
        )

        test_probabilities = predict_probabilities(
            model,
            data["X_test"],
        )

        test_metrics = calculate_metrics(
            data["y_test"],
            test_probabilities,
            data["n_classes"],
        )

        models[name] = model
        histories[name] = history
        probabilities[name] = test_probabilities
        metrics[name] = test_metrics

    print("\nTest-set results")
    print("-" * 72)
    print(
        f"{'Model':<15}{'Accuracy':>12}{'Macro F1':>12}"
        f"{'Weighted F1':>15}{'ROC-AUC OVR':>15}"
    )
    print("-" * 72)

    for name, result in metrics.items():
        print(
            f"{name:<15}"
            f"{result['accuracy']:>12.4f}"
            f"{result['macro_f1']:>12.4f}"
            f"{result['weighted_f1']:>15.4f}"
            f"{result['roc_auc_ovr_macro']:>15.4f}"
        )

        print(f"\n{name}: per-class classification report")
        print(
            classification_report(
                data["y_test"],
                probabilities[name].argmax(axis=1),
                labels=np.arange(data["n_classes"]),
                target_names=data["class_names"],
                zero_division=0,
            )
        )

    # Choose the model using validation loss, not test metrics.
    selected_name = min(
        histories,
        key=lambda name: histories[name]["best_val_loss"],
    )
    selected_model = models[selected_name]
    selected_metrics = metrics[selected_name]

    print(f"\nSelected model based on validation loss: {selected_name}")

    plot_learning_curves(histories)
    plot_confusion_matrix(
        data["y_test"],
        probabilities[selected_name],
        data["class_names"],
    )
    plot_per_class_f1(
        selected_metrics,
        data["class_names"],
    )

    selected_config = configurations[selected_name]

    checkpoint_path = save_model_artifacts(
        model=selected_model,
        scaler=data["scaler"],
        n_features=data["n_features"],
        n_classes=data["n_classes"],
        dropout=selected_config["dropout"],
        class_names=data["class_names"],
    )

    report = {
        "device": str(DEVICE),
        "split_sizes": data["split_sizes"],
        "class_distribution": data["class_distribution"],
        "selected_model": selected_name,
        "models": {
            name: {
                "best_epoch": histories[name]["best_epoch"],
                "best_validation_loss": histories[name]["best_val_loss"],
                "epochs_trained": histories[name]["epochs_trained"],
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
