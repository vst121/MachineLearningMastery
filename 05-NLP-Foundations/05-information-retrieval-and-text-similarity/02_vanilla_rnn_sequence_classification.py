"""
Implement text classification with a vanilla recurrent neural network.

This experiment covers:
1. Preparing a labeled text dataset.
2. Building a vocabulary and encoding token sequences.
3. Padding sequences and creating mini-batches.
4. Learning token embeddings and recurrent hidden states.
5. Training an RNN classifier with cross-entropy loss.
6. Evaluating predictions and inspecting classification errors.
7. Understanding how token order affects recurrent representations.
"""

import random
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


SEED = 42
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase whitespace-separated tokens."""
    return text.lower().split()


def build_vocabulary(
    texts: list[str],
    min_frequency: int = 1,
) -> dict[str, int]:
    """Build a deterministic vocabulary from training documents."""
    counts = Counter(
        token
        for text in texts
        for token in tokenize(text)
    )

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token in sorted(
        token
        for token, count in counts.items()
        if count >= min_frequency
    ):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    return vocabulary


def encode_text(
    text: str,
    vocabulary: dict[str, int],
    max_length: int,
) -> list[int]:
    """Encode, truncate, and pad a document."""
    tokens = tokenize(text)
    unknown_id = vocabulary[UNK_TOKEN]

    token_ids = [
        vocabulary.get(token, unknown_id)
        for token in tokens[:max_length]
    ]

    token_ids.extend([vocabulary[PAD_TOKEN]] * (max_length - len(token_ids)))
    return token_ids


class TextDataset(Dataset):
    """Dataset of encoded documents and integer labels."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocabulary: dict[str, int],
        max_length: int,
    ) -> None:
        self.features = torch.tensor(
            [
                encode_text(text, vocabulary, max_length)
                for text in texts
            ],
            dtype=torch.long,
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class VanillaRNNClassifier(nn.Module):
    """Classify a token sequence using its final valid hidden state."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            nonlinearity="tanh",
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return class logits and final hidden states.

        Packing prevents right-padding positions from being processed
        as ordinary tokens by the recurrent layer.
        """
        embedded = self.embedding(token_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.rnn(packed)
        final_hidden = hidden[-1]
        logits = self.classifier(final_hidden)

        return logits, final_hidden


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Train for one epoch and return mean loss and accuracy."""
    model.train()

    total_loss = 0.0
    total_examples = 0
    all_predictions = []
    all_labels = []

    for token_ids, labels, lengths in loader:
        token_ids = token_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits, _ = model(token_ids, lengths)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        predictions = logits.argmax(dim=1)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    mean_loss = total_loss / total_examples
    accuracy = accuracy_score(all_labels, all_predictions)

    return mean_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, list[int], list[int], list[float]]:
    """Evaluate the model and return loss, labels, predictions, confidences."""
    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_predictions = []
    all_labels = []
    all_confidences = []

    for token_ids, labels, lengths in loader:
        token_ids = token_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        logits, _ = model(token_ids, lengths)
        loss = criterion(logits, labels)

        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_confidences.extend(confidences.cpu().tolist())

    return (
        total_loss / total_examples,
        all_labels,
        all_predictions,
        all_confidences,
    )


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a batch and calculate each sequence's non-padding length."""
    token_ids, labels = zip(*batch)

    token_tensor = torch.stack(token_ids)
    label_tensor = torch.stack(labels)

    lengths = (token_tensor != 0).sum(dim=1).clamp(min=1)

    return token_tensor, label_tensor, lengths


def main() -> None:
    texts = [
        "team won the football match",
        "player scored a winning goal",
        "coach prepared the team for the match",
        "the striker scored two goals",
        "fans celebrated the championship win",
        "the goalkeeper saved the final shot",
        "the basketball team won the game",
        "the player trained before the tournament",
        "the software fixed a critical bug",
        "the computer processed the data",
        "developers released a software update",
        "the database stores application records",
        "the programmer debugged the application",
        "the server handled incoming requests",
        "the processor executed the program",
        "the application improved system performance",
        "the team won the final game",
        "the coach praised the winning players",
        "the software update fixed the system",
        "the developer tested the new application",
        "players prepared for the final tournament",
        "the server processed application data",
        "the goalkeeper trained before the match",
        "the programmer fixed a database bug",
    ]

    # 0 = sports, 1 = technology
    labels = [
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 0, 1, 0, 1,
    ]

    label_names = ["sports", "technology"]
    max_length = 12
    batch_size = 4
    embedding_dim = 32
    hidden_dim = 48
    learning_rate = 0.01
    epochs = 30

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=SEED,
        stratify=labels,
    )

    # Fit the vocabulary only on training documents.
    vocabulary = build_vocabulary(train_texts)

    train_dataset = TextDataset(
        train_texts, train_labels, vocabulary, max_length
    )
    test_dataset = TextDataset(
        test_texts, test_labels, vocabulary, max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = VanillaRNNClassifier(
        vocabulary_size=len(vocabulary),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=len(label_names),
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    print("=" * 72)
    print("DATASET AND MODEL")
    print("=" * 72)
    print(f"Training documents: {len(train_dataset)}")
    print(f"Test documents: {len(test_dataset)}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"RNN hidden dimension: {hidden_dim}")
    print(f"Device: {DEVICE}")

    print("\n" + "=" * 72)
    print("TRAINING")
    print("=" * 72)

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"loss={train_loss:.4f} | "
                f"accuracy={train_accuracy:.3f}"
            )

    print("\n" + "=" * 72)
    print("HELD-OUT TEST EVALUATION")
    print("=" * 72)

    test_loss, y_true, y_pred, confidences = evaluate(
        model, test_loader, criterion
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=label_names,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\n" + "=" * 72)
    print("PREDICTION ERROR ANALYSIS")
    print("=" * 72)

    for text, actual, predicted, confidence in zip(
        test_texts, test_labels, y_pred, confidences
    ):
        status = "CORRECT" if actual == predicted else "ERROR"

        print(f"[{status}] {text}")
        print(f"  Actual: {label_names[actual]}")
        print(f"  Predicted: {label_names[predicted]}")
        print(f"  Confidence: {confidence:.3f}")

    print("\n" + "=" * 72)
    print("INSPECT HIDDEN STATES")
    print("=" * 72)

    sample_texts = [
        "team won the match",
        "software fixed the bug",
    ]

    model.eval()

    for text in sample_texts:
        encoded = encode_text(text, vocabulary, max_length)
        token_tensor = torch.tensor(
            [encoded], dtype=torch.long, device=DEVICE
        )
        lengths = (token_tensor != 0).sum(dim=1).clamp(min=1)

        logits, hidden = model(token_tensor, lengths)
        probabilities = torch.softmax(logits, dim=1)[0]

        predicted = int(probabilities.argmax().item())

        print(f"\nText: {text}")
        print(f"Hidden state shape: {tuple(hidden.shape)}")
        print(f"Hidden state sample: {hidden[0, :8].cpu().numpy().round(3)}")
        print(f"Probabilities: {probabilities.cpu().numpy().round(3)}")
        print(f"Prediction: {label_names[predicted]}")

    print("\n" + "=" * 72)
    print("IMPORTANT LIMITATIONS")
    print("=" * 72)
    print("- This tiny dataset is for demonstrating the training process.")
    print("- Test scores are unstable and do not establish generalization.")
    print("- Whitespace tokenization does not handle punctuation robustly.")
    print("- A vanilla RNN may struggle with long-range dependencies.")
    print("- Learned embeddings are task-specific and randomly initialized.")


if __name__ == "__main__":
    main()
