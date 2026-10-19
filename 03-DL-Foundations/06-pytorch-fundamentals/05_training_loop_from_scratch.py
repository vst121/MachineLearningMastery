"""
PyTorch Training Loop From Scratch
==================================

Experiments:
1. Prepare a synthetic classification dataset for training and validation.
2. Build a neural network using nn.Module.
3. Implement the forward pass and calculate classification loss.
4. Perform backpropagation and update parameters with an optimizer.
5. Track training and validation loss across epochs.
6. Evaluate classification accuracy.
7. Compare model parameters before and after training.
8. Visualize convergence and model performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --------------------------------------------------
# 1. Prepare the dataset
# --------------------------------------------------

def prepare_data():
    X, y = make_moons(
        n_samples=1500,
        noise=0.25,
        random_state=RANDOM_SEED,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Fit preprocessing only on the training data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )

    val_X = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    return train_loader, val_X, val_y


# --------------------------------------------------
# 2. Define the neural network
# --------------------------------------------------

class BinaryClassifier(nn.Module):
    def __init__(self, input_features=2, hidden_features=32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, inputs):
        return self.network(inputs)


# --------------------------------------------------
# 3. Evaluate the model
# --------------------------------------------------

def evaluate_model(model, X, y, criterion):
    model.eval()

    with torch.no_grad():
        logits = model(X)
        loss = criterion(logits, y)

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        accuracy = (predictions == y).float().mean().item()

    return loss.item(), accuracy


# --------------------------------------------------
# 4. Train the model
# --------------------------------------------------

def train_model(model, train_loader, val_X, val_y):
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            # 1. Clear gradients from the previous update.
            optimizer.zero_grad(set_to_none=True)

            # 2. Forward pass: compute model predictions.
            logits = model(X_batch)

            # 3. Calculate the loss.
            loss = criterion(logits, y_batch)

            # 4. Backward pass: calculate gradients.
            loss.backward()

            # 5. Update model parameters.
            optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_train_loss = total_loss / total_samples

        # Calculate metrics for the entire training dataset.
        train_X = train_loader.dataset.tensors[0]
        train_y = train_loader.dataset.tensors[1]

        _, train_accuracy = evaluate_model(
            model,
            train_X,
            train_y,
            criterion,
        )

        val_loss, val_accuracy = evaluate_model(
            model,
            val_X,
            val_y,
            criterion,
        )

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1:3d}/{EPOCHS}] "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.2%} | "
                f"Val Accuracy: {val_accuracy:.2%}"
            )

    return history


# --------------------------------------------------
# 5. Visualize training history
# --------------------------------------------------

def plot_history(history):
    epochs = range(1, EPOCHS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Training Loss",
    )
    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )
    axes[0].set_title("Loss Across Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy",
    )
    axes[1].plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )
    axes[1].set_title("Accuracy Across Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 6. Main
# --------------------------------------------------

def main():
    train_loader, val_X, val_y = prepare_data()

    model = BinaryClassifier()

    # Save initial parameters to compare before and after training.
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    print("Model architecture:\n")
    print(model)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nTrainable parameters:", parameter_count)

    print("\nTraining started...")
    history = train_model(
        model,
        train_loader,
        val_X,
        val_y,
    )

    # Compare parameter values before and after optimization.
    print("\nParameter Changes")

    for name, parameter in model.named_parameters():
        change = (
            parameter.detach() - initial_parameters[name]
        ).norm().item()

        print(f"{name:<30} L2 change: {change:.6f}")

    # Report final validation performance.
    criterion = nn.BCEWithLogitsLoss()

    final_val_loss, final_val_accuracy = evaluate_model(
        model,
        val_X,
        val_y,
        criterion,
    )

    print("\nFinal Validation Results")
    print(f"Validation Loss: {final_val_loss:.4f}")
    print(f"Validation Accuracy: {final_val_accuracy:.2%}")

    plot_history(history)


if __name__ == "__main__":
    main()
