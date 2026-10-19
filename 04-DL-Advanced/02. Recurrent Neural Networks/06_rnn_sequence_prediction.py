"""
Train a Recurrent Neural Network for time-series prediction.

This experiment covers:
1. Generating a synthetic time series.
2. Creating input sequences and next-step targets.
3. Building an RNN using PyTorch.
4. Training with Mean Squared Error and Adam.
5. Evaluating predictions on unseen data.
6. Comparing predictions against actual values.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(
    series: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows and their next-step targets."""
    inputs = []
    targets = []

    for i in range(len(series) - sequence_length):
        inputs.append(series[i : i + sequence_length])
        targets.append(series[i + sequence_length])

    return np.array(inputs), np.array(targets)


class RNNPredictor(nn.Module):
    """Predict the next time-series value from an input sequence."""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one prediction for each input sequence."""
        output, _ = self.rnn(x)

        # Use the hidden representation at the final time step.
        last_hidden_state = output[:, -1, :]

        return self.output_layer(last_hidden_state)


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    sequence_length = 20
    train_ratio = 0.8
    batch_size = 32
    epochs = 20
    learning_rate = 0.001

    # Generate a noisy, periodic time series.
    time = np.linspace(0, 80, 1200)
    series = (
        np.sin(time)
        + 0.3 * np.sin(2 * time)
        + 0.05 * np.random.randn(len(time))
    ).astype(np.float32)

    # Split chronologically to avoid training on future observations.
    split_index = int(len(series) * train_ratio)

    train_series = series[:split_index]
    test_series = series[split_index:]

    X_train, y_train = create_sequences(
        train_series, sequence_length
    )
    X_test, y_test = create_sequences(
        test_series, sequence_length
    )

    # RNN input shape: (samples, sequence_length, input_size).
    X_train = torch.from_numpy(X_train).unsqueeze(-1)
    y_train = torch.from_numpy(y_train).unsqueeze(-1)
    X_test = torch.from_numpy(X_test).unsqueeze(-1)
    y_test = torch.from_numpy(y_test).unsqueeze(-1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = RNNPredictor(hidden_size=16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    print("RNN Sequence Prediction")
    print("=" * 55)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Input shape:      {tuple(X_train.shape)}")
    print(f"Target shape:     {tuple(y_train.shape)}")
    print()

    # Training loop.
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()

            # Limit excessively large gradient norms.
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        average_loss = total_loss / len(train_dataset)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} "
                f"| Training MSE: {average_loss:.6f}"
            )

    # Evaluate on the unseen test segment.
    model.eval()

    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test).item()

        # Mean absolute error in the original series units.
        test_mae = torch.mean(
            torch.abs(predictions - y_test)
        ).item()

    print("\nEvaluation")
    print("-" * 55)
    print(f"Test MSE:  {test_loss:.6f}")
    print(f"Test MAE:  {test_mae:.6f}")

    print("\nSample Predictions")
    print("-" * 55)
    print(f"{'Actual':>12} {'Predicted':>12} {'Absolute Error':>16}")

    for actual, predicted in zip(
        y_test[:10].squeeze(-1),
        predictions[:10].squeeze(-1),
    ):
        error = abs(actual.item() - predicted.item())

        print(
            f"{actual.item():12.4f} "
            f"{predicted.item():12.4f} "
            f"{error:16.4f}"
        )

    # Verify model output dimensions and numerical stability.
    assert predictions.shape == y_test.shape
    assert torch.isfinite(predictions).all()
    assert np.isfinite(test_loss)
    assert np.isfinite(test_mae)

    print("\nAll shape and numerical checks passed.")


if __name__ == "__main__":
    main()
