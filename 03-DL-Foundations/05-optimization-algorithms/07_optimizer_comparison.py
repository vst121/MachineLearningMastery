"""
Optimizer Comparison
====================

Experiments:
1. Compare SGD, SGD with momentum, RMSprop, Adam, and AdamW.
2. Train identical neural networks using consistent initial parameters.
3. Compare convergence speed and training stability.
4. Evaluate training loss, validation loss, and classification accuracy.
5. Visualize optimizer performance and convergence behavior.
6. Analyze the trade-offs between optimization speed, stability, and generalization.
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
EPOCHS = 80
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )

    val_X = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    return train_dataset, val_X, val_y


# --------------------------------------------------
# 2. Define the neural network
# --------------------------------------------------

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, X):
        return self.network(X)


# --------------------------------------------------
# 3. Create optimizers
# --------------------------------------------------

def create_optimizer(name, model):
    if name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
        )

    if name == "SGD + Momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=0.9,
        )

    if name == "RMSprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=LEARNING_RATE,
        )

    if name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

    if name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.01,
        )

    raise ValueError(f"Unknown optimizer: {name}")


# --------------------------------------------------
# 4. Train and evaluate one optimizer
# --------------------------------------------------

def train_model(
    optimizer_name,
    initial_state,
    train_dataset,
    val_X,
    val_y,
):
    model = BinaryClassifier()
    model.load_state_dict(initial_state)

    optimizer = create_optimizer(optimizer_name, model)
    criterion = nn.BCEWithLogitsLoss()

    # Reset the generator so every optimizer sees the same
    # shuffled mini-batch order during each epoch.
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_train_loss = total_loss / total_samples
        history["train_loss"].append(epoch_train_loss)

        model.eval()

        with torch.no_grad():
            val_logits = model(val_X)
            val_loss = criterion(val_logits, val_y).item()

            probabilities = torch.sigmoid(val_logits)
            predictions = (probabilities >= 0.5).float()

            val_accuracy = (
                predictions == val_y
            ).float().mean().item()

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

    return history


# --------------------------------------------------
# 5. Run all optimizer experiments
# --------------------------------------------------

def main():
    train_dataset, val_X, val_y = prepare_data()

    # Initialize one reference model and reuse its weights.
    reference_model = BinaryClassifier()

    initial_state = {
        name: parameter.detach().clone()
        for name, parameter in reference_model.state_dict().items()
    }

    optimizer_names = [
        "SGD",
        "SGD + Momentum",
        "RMSprop",
        "Adam",
        "AdamW",
    ]

    results = {}

    for name in optimizer_names:
        print(f"Training with {name}...")

        results[name] = train_model(
            optimizer_name=name,
            initial_state=initial_state,
            train_dataset=train_dataset,
            val_X=val_X,
            val_y=val_y,
        )

    # --------------------------------------------------
    # 6. Print final metrics
    # --------------------------------------------------

    print("\nFinal Results")
    print("-" * 78)
    print(
        f"{'Optimizer':<20}"
        f"{'Train Loss':>14}"
        f"{'Val Loss':>14}"
        f"{'Val Accuracy':>16}"
    )

    for name, history in results.items():
        print(
            f"{name:<20}"
            f"{history['train_loss'][-1]:>14.4f}"
            f"{history['val_loss'][-1]:>14.4f}"
            f"{history['val_accuracy'][-1]:>15.2%}"
        )

    # --------------------------------------------------
    # 7. Plot training and validation loss
    # --------------------------------------------------

    epochs = range(1, EPOCHS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for name, history in results.items():
        axes[0].plot(
            epochs,
            history["train_loss"],
            label=name,
        )

        axes[1].plot(
            epochs,
            history["val_loss"],
            label=name,
        )

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary Cross-Entropy Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 8. Compare validation accuracy
    # --------------------------------------------------

    final_accuracies = [
        results[name]["val_accuracy"][-1]
        for name in optimizer_names
    ]

    plt.figure(figsize=(10, 5))

    plt.bar(optimizer_names, final_accuracies)

    plt.title("Final Validation Accuracy by Optimizer")
    plt.xlabel("Optimizer")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
