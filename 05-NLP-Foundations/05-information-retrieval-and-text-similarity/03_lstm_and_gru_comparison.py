"""
Compare LSTM and GRU networks for text sequence classification.

This experiment covers:
1. Preparing a labeled text dataset with a shared train-test split.
2. Building a vocabulary from training documents only.
3. Encoding and padding variable-length token sequences.
4. Training LSTM and GRU classifiers with identical data and settings.
5. Comparing training loss, accuracy, parameter counts, and inference time.
6. Evaluating held-out predictions using classification metrics.
7. Understanding how gated recurrent networks manage sequential information.
"""

import random
import time
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


SEED = 42
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase whitespace-separated tokens."""
    return text.lower().split()


def build_vocabulary(texts: list[str]) -> dict[str, int]:
    """Build a deterministic vocabulary from training documents."""
    tokens = Counter(
        token
        for text in texts
        for token in tokenize(text)
    )

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token in sorted(tokens):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    return vocabulary


def encode_text(
    text: str,
    vocabulary: dict[str, int],
    max_length: int,
) -> list[int]:
    """Encode a document, truncate it, and pad it to max_length."""
    token_ids = [
        vocabulary.get(token, vocabulary[UNK_TOKEN])
        for token in tokenize(text)[:max_length]
    ]

    token_ids.extend(
        [vocabulary[PAD_TOKEN]] * (max_length - len(token_ids))
    )

    return token_ids


class TextDataset(Dataset):
    """Store token ID sequences and their labels."""

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

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def collate_batch(batch):
    """Return token IDs, labels, and non-padding sequence lengths."""
    token_ids, labels = zip(*batch)

    token_tensor = torch.stack(token_ids)
    label_tensor = torch.stack(labels)
    lengths = (token_tensor != 0).sum(dim=1).clamp(min=1)

    return token_tensor, label_tensor, lengths


class GatedRNNClassifier(nn.Module):
    """Classify text using either an LSTM or a GRU encoder."""

    def __init__(
        self,
        cell_type: str,
        vocabulary_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=0,
        )

        if cell_type == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
        elif cell_type == "GRU":
            self.recurrent = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
        else:
            raise ValueError("cell_type must be 'LSTM' or 'GRU'.")

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.cell_type = cell_type

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(token_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, state = self.recurrent(packed)

        # LSTM returns (hidden_state, cell_state); GRU returns hidden_state.
        hidden_state = state[0] if self.cell_type == "LSTM" else state
        final_hidden = hidden_state[-1]

        logits = self.classifier(final_hidden)

        return logits, final_hidden


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Train one epoch and return average loss and accuracy."""
    model.train()

    total_loss = 0.0
    total_examples = 0
    predictions_all = []
    labels_all = []

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

        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())

    return (
        total_loss / total_examples,
        accuracy_score(labels_all, predictions_all),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> dict:
    """Evaluate a model and return its loss, predictions, and metrics."""
    model.eval()

    total_loss = 0.0
    total_examples = 0
    predictions_all = []
    labels_all = []

    start_time = time.perf_counter()

    for token_ids, labels, lengths in loader:
        token_ids = token_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        logits, _ = model(token_ids, lengths)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    elapsed = time.perf_counter() - start_time

    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_score(labels_all, predictions_all),
        "macro_f1": f1_score(
            labels_all,
            predictions_all,
            average="macro",
            zero_division=0,
        ),
        "predictions": predictions_all,
        "labels": labels_all,
        "inference_seconds": elapsed,
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


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
    epochs = 25

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=SEED,
        stratify=labels,
    )

    vocabulary = build_vocabulary(train_texts)

    train_dataset = TextDataset(
        train_texts, train_labels, vocabulary, max_length
    )
    test_dataset = TextDataset(
        test_texts, test_labels, vocabulary, max_length
    )

    # Use the same data order for each model's training configuration.
    def make_train_loader() -> DataLoader:
        generator = torch.Generator().manual_seed(SEED)
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate_batch,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    results = []

    print("=" * 72)
    print("LSTM AND GRU COMPARISON")
    print("=" * 72)
    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Device: {DEVICE}")
    print(f"Epochs per model: {epochs}")

    for cell_type in ["LSTM", "GRU"]:
        # Reset the random seed so initialization is reproducible.
        torch.manual_seed(SEED)

        model = GatedRNNClassifier(
            cell_type=cell_type,
            vocabulary_size=len(vocabulary),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=len(label_names),
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
        criterion = nn.CrossEntropyLoss()
        train_loader = make_train_loader()

        print(f"\n{'-' * 72}")
        print(f"Training {cell_type}")
        print(f"Trainable parameters: {count_parameters(model):,}")

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

        metrics = evaluate(model, test_loader, criterion)

        print(f"\n{cell_type} held-out results")
        print(f"Test loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Macro F1: {metrics['macro_f1']:.3f}")
        print(
            "Inference time for the test set: "
            f"{metrics['inference_seconds']:.6f} seconds"
        )

        print("\nClassification report:")
        print(
            classification_report(
                metrics["labels"],
                metrics["predictions"],
                labels=[0, 1],
                target_names=label_names,
                zero_division=0,
            )
        )

        results.append({
            "model": cell_type,
            "parameters": count_parameters(model),
            "test_loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "inference_seconds": metrics["inference_seconds"],
        })

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"{'Model':<10} {'Parameters':>12} {'Test Loss':>12} "
        f"{'Accuracy':>10} {'Macro F1':>10} {'Inference (s)':>15}"
    )

    for result in results:
        print(
            f"{result['model']:<10} "
            f"{result['parameters']:>12,} "
            f"{result['test_loss']:>12.4f} "
            f"{result['accuracy']:>10.3f} "
            f"{result['macro_f1']:>10.3f} "
            f"{result['inference_seconds']:>15.6f}"
        )

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print("- LSTMs use a cell state and three gates to control information flow.")
    print("- GRUs use update and reset gates with a simpler state structure.")
    print("- LSTM parameter counts are generally higher for matching dimensions.")
    print("- Runtime depends on hardware, batch size, and implementation details.")
    print("- A tiny test set cannot establish which architecture is generally better.")
    print("- Use repeated runs or cross-validation on a larger dataset for robust conclusions.")


if __name__ == "__main__":
    main()
