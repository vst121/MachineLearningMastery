"""
Learning Rate Schedulers
========================

Experiments:
1. Compare a constant learning rate with StepLR, ExponentialLR, and CosineAnnealingLR.
2. Train identical neural networks using the same optimizer and initial parameters.
3. Track training and validation loss throughout training.
4. Measure final validation accuracy.
5. Visualize learning-rate schedules and convergence behavior.
6. Analyze how learning-rate decay affects optimization and generalization.
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
INITIAL_LR = 0.01

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

    # Fit the scaler only on the training data.
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
# 3. Create a learning-rate scheduler
# --------------------------------------------------

def create_scheduler(name, optimizer):
    if name == "Constant":
        return None

    if name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.5,
        )

    if name == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.97,
        )

    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=1e-4,
        )

    raise ValueError(f"Unknown scheduler: {name}")


# --------------------------------------------------
# 4. Train and evaluate one configuration
# --------------------------------------------------

def train_model(
    scheduler_name,
    initial_state,
    train_dataset,
    val_X,
    val_y,
):
    model = BinaryClassifier()
    model.load_state_dict(initial_state)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=INITIAL_LR,
    )

    scheduler = create_scheduler(scheduler_name, optimizer)
    criterion = nn.BCEWithLogitsLoss()

    # Reuse the same shuffled batch order for every experiment.
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
        "learning_rate": [],
    }

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_samples = 0

        # Record the learning rate used during this epoch.
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rate"].append(current_lr)

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        history["train_loss"].append(
            total_loss / total_samples
        )

        # Evaluate on validation data without calculating gradients.
        model.eval()

        with torch.no_grad():
            val_logits = model(val_X)
            val_loss = criterion(val_logits, val_y).item()

            probabilities = torch.sigmoid(val_logits)
            predictions = (probabilities >= 0.5).float()

            accuracy = (
                predictions == val_y
            ).float().mean().item()

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(accuracy)

        # Update the learning rate for the next epoch.
        if scheduler is not None:
            scheduler.step()

    return history


# --------------------------------------------------
# 5. Run all experiments
# --------------------------------------------------

def main():
    train_dataset, val_X, val_y = prepare_data()

    # Use identical initial parameters in every experiment.
    reference_model = BinaryClassifier()
    initial_state = {
        name: parameter.detach().clone()
        for name, parameter in reference_model.state_dict().items()
    }

    scheduler_names = [
        "Constant",
        "StepLR",
        "ExponentialLR",
        "CosineAnnealingLR",
    ]

    results = {}

    for name in scheduler_names:
        print(f"Training with {name}...")

        results[name] = train_model(
            scheduler_name=name,
            initial_state=initial_state,
            train_dataset=train_dataset,
            val_X=val_X,
            val_y=val_y,
        )

    # --------------------------------------------------
    # 6. Report final metrics
    # --------------------------------------------------

    print("\nFinal Results")
    print("-" * 90)
    print(
        f"{'Scheduler':<22}"
        f"{'Train Loss':>14}"
        f"{'Val Loss':>14}"
        f"{'Val Accuracy':>16}"
        f"{'Final LR':>16}"
    )

    for name, history in results.items():
        print(
            f"{name:<22}"
            f"{history['train_loss'][-1]:>14.4f}"
            f"{history['val_loss'][-1]:>14.4f}"
            f"{history['val_accuracy'][-1]:>15.2%}"
            f"{history['learning_rate'][-1]:>16.6f}"
        )

    # --------------------------------------------------
    # 7. Plot training and validation losses
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
    # 8. Plot learning-rate schedules
    # --------------------------------------------------

    plt.figure(figsize=(10, 5))

    for name, history in results.items():
        plt.plot(
            epochs,
            history["learning_rate"],
            label=name,
        )

    plt.title("Learning Rate Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
