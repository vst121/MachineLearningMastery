"""
Manual Backpropagation
======================

Experiments:
1. Implement a two-layer neural network using NumPy.
2. Perform a forward pass and calculate binary cross-entropy loss.
3. Derive and implement gradients using the chain rule.
4. Update weights and biases using gradient descent.
5. Train the network on a small binary classification dataset.
6. Visualize the loss across training epochs.
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """A small two-layer neural network with manual backpropagation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        self.W1 = rng.normal(
            0.0, np.sqrt(2.0 / input_size), (input_size, hidden_size)
        )
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = rng.normal(
            0.0, np.sqrt(2.0 / hidden_size), (hidden_size, output_size)
        )
        self.b2 = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid activation."""
        result = np.empty_like(z, dtype=np.float64)
        positive = z >= 0

        result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

        exp_z = np.exp(z[~positive])
        result[~positive] = exp_z / (1.0 + exp_z)

        return result

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Calculate predictions and cache intermediate values."""
        self.X = X

        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.maximum(0.0, self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    @staticmethod
    def binary_cross_entropy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epsilon: float = 1e-12,
    ) -> float:
        """Calculate mean binary cross-entropy."""
        probabilities = np.clip(y_pred, epsilon, 1.0 - epsilon)

        return float(
            -np.mean(
                y_true * np.log(probabilities)
                + (1.0 - y_true) * np.log(1.0 - probabilities)
            )
        )

    def backward(self, y_true: np.ndarray) -> dict[str, np.ndarray]:
        """Calculate gradients manually using the chain rule."""
        sample_count = y_true.shape[0]

        # Sigmoid combined with binary cross-entropy.
        dZ2 = (self.A2 - y_true) / sample_count

        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Propagate gradients through the second layer.
        dA1 = dZ2 @ self.W2.T

        # Derivative of ReLU.
        dZ1 = dA1 * (self.Z1 > 0.0)

        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }

    def update_parameters(
        self,
        gradients: dict[str, np.ndarray],
        learning_rate: float,
    ) -> None:
        """Update parameters using gradient descent."""
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels."""
        probabilities = self.forward(X)
        return (probabilities >= 0.5).astype(int)


def main() -> None:
    # Experiment 1: Small binary classification dataset.
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    # XOR targets.
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    network = NeuralNetwork(
        input_size=2,
        hidden_size=8,
        output_size=1,
    )

    # Experiment 2: Initial forward pass.
    predictions = network.forward(X)
    initial_loss = network.binary_cross_entropy(y, predictions)

    print("Experiment 1: Initial forward pass")
    print(f"Initial loss: {initial_loss:.4f}")
    print("Initial probabilities:")
    print(np.round(predictions, 4))
    print()

    # Experiment 3: Inspect manually calculated gradients.
    gradients = network.backward(y)

    print("Experiment 2: Gradient shapes")
    for name, gradient in gradients.items():
        print(f"{name}: shape={gradient.shape}, norm={np.linalg.norm(gradient):.6f}")
    print()

    # Experiment 4: Train using gradient descent.
    learning_rate = 0.1
    epochs = 10_000
    losses = []

    for epoch in range(epochs):
        predictions = network.forward(X)
        loss = network.binary_cross_entropy(y, predictions)
        losses.append(loss)

        gradients = network.backward(y)
        network.update_parameters(gradients, learning_rate)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

    # Experiment 5: Evaluate final predictions.
    probabilities = network.forward(X)
    predictions = (probabilities >= 0.5).astype(int)
    final_loss = network.binary_cross_entropy(y, probabilities)
    accuracy = np.mean(predictions == y)

    print("\nExperiment 3: Final results")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Accuracy:     {accuracy:.2%}")
    print("\nPredictions:")
    for inputs, probability, prediction, target in zip(
        X, probabilities, predictions, y
    ):
        print(
            f"Input={inputs.astype(int)} | "
            f"Probability={probability[0]:.4f} | "
            f"Predicted={prediction[0]} | "
            f"Target={int(target[0])}"
        )

    # Experiment 6: Visualize training loss.
    plt.figure(figsize=(9, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Manual Backpropagation: Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
