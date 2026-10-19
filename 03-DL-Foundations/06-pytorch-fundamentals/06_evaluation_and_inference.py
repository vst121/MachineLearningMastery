"""
PyTorch Evaluation and Inference
================================

Experiments:
1. Train a classifier with explicit training and evaluation modes.
2. Compare model.train() and model.eval() behavior with Dropout.
3. Distinguish evaluation mode from gradient disabling.
4. Convert logits into probabilities and class predictions.
5. Evaluate accuracy, precision, recall, and the confusion matrix.
6. Perform inference with torch.inference_mode().
7. Predict classes for new, unseen samples.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_moons
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def section(title: str) -> None:
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")


def prepare_data():
    """Create reproducible data and avoid preprocessing leakage."""
    features, labels = make_moons(
        n_samples=2000,
        noise=0.25,
        random_state=42,
    )

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    def to_tensor_dataset(x, y):
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )

    train_dataset = to_tensor_dataset(x_train, y_train)
    val_dataset = to_tensor_dataset(x_val, y_val)
    test_dataset = to_tensor_dataset(x_test, y_test)

    return train_dataset, val_dataset, test_dataset, scaler


class MoonClassifier(nn.Module):
    def __init__(self, input_features: int = 2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return logits. Apply sigmoid when probabilities are needed.
        return self.network(x)


def train_model(model, train_loader, val_loader, device, epochs=40):
    """Train using gradients and evaluate validation loss without them."""
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        # Training mode enables Dropout and training behavior in BatchNorm.
        model.train()
        total_train_loss = 0.0
        train_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(x_batch)
            loss = loss_function(logits, y_batch)

            loss.backward()
            optimizer.step()

            batch_size = x_batch.size(0)
            total_train_loss += loss.item() * batch_size
            train_samples += batch_size

        # Evaluation mode changes layer behavior, not gradient tracking.
        model.eval()
        total_val_loss = 0.0
        val_samples = 0

        with torch.inference_mode():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
                loss = loss_function(logits, y_batch)

                batch_size = x_batch.size(0)
                total_val_loss += loss.item() * batch_size
                val_samples += batch_size

        if (epoch + 1) % 10 == 0:
            train_loss = total_train_loss / train_samples
            val_loss = total_val_loss / val_samples

            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Validation Loss: {val_loss:.4f}"
            )



def demonstrate_train_eval_modes(model, device):
    section("1. Training Mode vs Evaluation Mode")

    # BatchNorm needs more than one sample per channel in training mode.
    sample = torch.randn(4, 2, device=device)

    model.train()
    train_output_1 = model(sample)
    train_output_2 = model(sample)

    model.eval()
    with torch.inference_mode():
        eval_output_1 = model(sample)
        eval_output_2 = model(sample)

    print("Training mode outputs 1:", train_output_1.detach().flatten())
    print("Training mode outputs 2:", train_output_2.detach().flatten())
    print("Evaluation mode outputs 1:", eval_output_1.flatten())
    print("Evaluation mode outputs 2:", eval_output_2.flatten())

    print("\nDropout may cause training outputs to differ.")
    print("In evaluation mode, Dropout is disabled.")
    print("BatchNorm uses stored running statistics in eval mode.")


def demonstrate_gradient_tracking(model, device):
    section("2. Evaluation Mode Does Not Disable Gradients")

    sample = torch.randn(4, 2, device=device)

    model.eval()

    # eval() does not disable gradient computation.
    output_with_gradients = model(sample)
    print(
        "Gradients enabled after model.eval():",
        output_with_gradients.requires_grad,
    )

    # inference_mode() disables gradient tracking for inference.
    with torch.inference_mode():
        output_without_gradients = model(sample)

    print(
        "Gradients enabled inside inference_mode():",
        output_without_gradients.requires_grad,
    )


def evaluate_model(model, data_loader, device):
    """Return predictions, probabilities, labels, and evaluation metrics."""
    model.eval()

    all_probabilities = []
    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)

            logits = model(x_batch)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).long()

            all_probabilities.extend(
                probabilities.cpu().numpy().ravel()
            )
            all_predictions.extend(
                predictions.cpu().numpy().ravel()
            )
            all_labels.extend(
                y_batch.numpy().ravel().astype(int)
            )

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    y_prob = np.asarray(all_probabilities)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    return y_true, y_pred, y_prob, metrics


def show_metrics(metrics):
    section("3. Classification Metrics")

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("\nRows = actual classes; columns = predicted classes.")


def predict_new_samples(model, scaler, device):
    section("4. Inference on New Samples")

    # These coordinates are raw, unscaled feature values.
    new_samples = np.array([
        [-0.5, 0.5],
        [1.5, -0.5],
        [0.0, 1.0],
    ])

    # Apply the same preprocessing fitted on the training set.
    scaled_samples = scaler.transform(new_samples)
    x_new = torch.tensor(
        scaled_samples,
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    with torch.inference_mode():
        logits = model(x_new)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()

    for index, sample in enumerate(new_samples):
        print(
            f"Sample {sample} | "
            f"Probability of class 1: "
            f"{probabilities[index].item():.4f} | "
            f"Predicted class: {predictions[index].item()}"
        )


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    train_dataset, val_dataset, test_dataset, scaler = prepare_data()

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
    )

    model = MoonClassifier().to(device)

    section("5. Model Training")
    train_model(model, train_loader, val_loader, device)

    demonstrate_train_eval_modes(model, device)
    demonstrate_gradient_tracking(model, device)

    section("6. Final Test Evaluation")
    _, _, _, metrics = evaluate_model(model, test_loader, device)
    show_metrics(metrics)

    predict_new_samples(model, scaler, device)


if __name__ == "__main__":
    main()
