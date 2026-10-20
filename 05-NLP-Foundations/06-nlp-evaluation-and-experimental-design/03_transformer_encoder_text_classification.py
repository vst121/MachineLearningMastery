"""
Build a Transformer encoder for text classification.

This experiment covers:
1. Preparing a small text classification dataset.
2. Building a vocabulary from training data only.
3. Converting tokens into embeddings and adding positional encodings.
4. Applying Transformer encoder layers with padding masks.
5. Training a classifier with cross-entropy loss.
6. Evaluating predictions using accuracy and a classification report.
7. Understanding how self-attention supports contextual text representations.
"""

import math
import random
import re
import time
from collections import Counter

import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
TEST_SIZE = 0.25
EMBEDDING_DIM = 32
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
FEEDFORWARD_DIM = 64
DROPOUT = 0.1
MAX_SEQUENCE_LENGTH = 24
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.003

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def set_seed(seed: int = SEED) -> None:
    """Make the experiment reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)


def tokenize(text: str) -> list[str]:
    """Lowercase text and extract simple word and number tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def build_vocabulary(
    texts: list[str],
    min_frequency: int = 1,
) -> dict[str, int]:
    """Build a vocabulary using only the supplied training texts."""
    counts = Counter(
        token
        for text in texts
        for token in tokenize(text)
    )

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, count in sorted(counts.items()):
        if count >= min_frequency and token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    return vocabulary


def encode_text(
    text: str,
    vocabulary: dict[str, int],
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> list[int]:
    """Convert a text into token IDs, truncating to a fixed maximum length."""
    tokens = tokenize(text)[:max_length]
    token_ids = [
        vocabulary.get(token, vocabulary[UNK_TOKEN])
        for token in tokens
    ]

    if not token_ids:
        token_ids = [vocabulary[UNK_TOKEN]]

    return token_ids


class TextDataset(Dataset):
    """Store token sequences and their classification labels."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocabulary: dict[str, int],
    ) -> None:
        self.sequences = [
            encode_text(text, vocabulary)
            for text in texts
        ]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.sequences[index], self.labels[index]


def collate_batch(
    batch: list[tuple[list[int], int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences within a batch and return their original lengths."""
    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long,
    )

    max_length = max(lengths).item()
    padded_sequences = torch.full(
        (len(sequences), max_length),
        fill_value=0,
        dtype=torch.long,
    )

    for index, sequence in enumerate(sequences):
        padded_sequences[index, : len(sequence)] = torch.tensor(
            sequence,
            dtype=torch.long,
        )

    return (
        padded_sequences,
        torch.tensor(labels, dtype=torch.long),
        lengths,
    )


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal positional information to token embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        max_length: int,
        dropout: float,
    ) -> None:
        super().__init__()

        positions = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        dimensions = torch.arange(
            embedding_dim,
            dtype=torch.float32,
        )

        frequencies = torch.exp(
            (2 * torch.floor(dimensions / 2) / embedding_dim)
            * (-math.log(10000.0))
        )

        angles = positions * frequencies.unsqueeze(0)

        encoding = torch.zeros(
            max_length,
            embedding_dim,
            dtype=torch.float32,
        )

        encoding[:, 0::2] = torch.sin(angles[:, 0::2])
        encoding[:, 1::2] = torch.cos(angles[:, 1::2])

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        sequence_length = embeddings.size(1)

        if sequence_length > self.encoding.size(1):
            raise ValueError("Input sequence exceeds the positional encoding limit.")

        positions = self.encoding[:, :sequence_length, :]
        return self.dropout(embeddings + positions)


class TransformerTextClassifier(nn.Module):
    """Classify text using Transformer encoder representations."""

    def __init__(
        self,
        vocabulary_size: int,
        num_classes: int,
        embedding_dim: int = EMBEDDING_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_ENCODER_LAYERS,
        feedforward_dim: int = FEEDFORWARD_DIM,
        dropout: float = DROPOUT,
        max_length: int = MAX_SEQUENCE_LENGTH,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.token_embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=0,
        )

        self.position_encoding = SinusoidalPositionalEncoding(
            embedding_dim=embedding_dim,
            max_length=max_length,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return class logits.

        Args:
            token_ids: Tensor shaped (batch_size, sequence_length).
                Padding tokens must use ID 0.
        """
        padding_mask = token_ids.eq(0)

        embeddings = self.token_embedding(token_ids)
        embeddings = embeddings * math.sqrt(embeddings.size(-1))
        embeddings = self.position_encoding(embeddings)

        encoded = self.encoder(
            embeddings,
            src_key_padding_mask=padding_mask,
        )

        # Exclude padded positions from the sequence representation.
        valid_tokens = (~padding_mask).unsqueeze(-1).to(encoded.dtype)

        pooled = (encoded * valid_tokens).sum(dim=1)
        pooled = pooled / valid_tokens.sum(dim=1).clamp(min=1.0)

        return self.classifier(pooled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return mean loss and accuracy."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for token_ids, labels, _lengths in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(token_ids)
        loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Return ground-truth labels and predicted labels."""
    model.eval()

    all_labels: list[int] = []
    all_predictions: list[int] = []

    for token_ids, labels, _lengths in loader:
        token_ids = token_ids.to(device)

        logits = model(token_ids)
        predictions = logits.argmax(dim=1).cpu().tolist()

        all_predictions.extend(predictions)
        all_labels.extend(labels.tolist())

    return all_labels, all_predictions


def main() -> None:
    set_seed()

    # Educational dataset: sports (0) versus technology (1).
    # The dataset is deliberately small and is not suitable for reliable
    # estimates of real-world generalization.
    examples = [
        ("The team won the football match with a late goal", 0),
        ("The coach prepared players for the final game", 0),
        ("The basketball team improved its defensive strategy", 0),
        ("The striker scored twice during the tournament", 0),
        ("The tennis player practiced before the championship", 0),
        ("The athlete trained daily to improve running speed", 0),
        ("The referee stopped the match after an injury", 0),
        ("The team celebrated its victory in the stadium", 0),
        ("The goalkeeper saved a penalty in the final minute", 0),
        ("The club signed a new player for the season", 0),
        ("The runner finished the marathon with a personal record", 0),
        ("The players discussed tactics before the important match", 0),
        ("The software engineer deployed a new application", 1),
        ("The database stores customer records securely", 1),
        ("The machine learning model predicts future demand", 1),
        ("The developer fixed a bug in the web service", 1),
        ("The cloud platform supports distributed applications", 1),
        ("The neural network learned patterns from training data", 1),
        ("The server processes requests through an API", 1),
        ("The programmer wrote tests for the new software", 1),
        ("The computer uses memory to store information", 1),
        ("The team improved the performance of its database", 1),
        ("The application uses an algorithm to rank documents", 1),
        ("The engineer configured containers for deployment", 1),
    ]

    texts = [text for text, _ in examples]
    labels = [label for _, label in examples]

    indices = list(range(len(examples)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    train_texts = [texts[index] for index in train_indices]
    test_texts = [texts[index] for index in test_indices]
    train_labels = [labels[index] for index in train_indices]
    test_labels = [labels[index] for index in test_indices]

    # Build vocabulary only from training data to prevent test-data leakage.
    vocabulary = build_vocabulary(train_texts)

    train_dataset = TextDataset(
        train_texts,
        train_labels,
        vocabulary,
    )
    test_dataset = TextDataset(
        test_texts,
        test_labels,
        vocabulary,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TransformerTextClassifier(
        vocabulary_size=len(vocabulary),
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    print("TRANSFORMER ENCODER TEXT CLASSIFICATION")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Attention heads: {NUM_HEADS}")
    print(f"Encoder layers: {NUM_ENCODER_LAYERS}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"loss={train_loss:.4f} | "
                f"accuracy={train_accuracy:.3f}"
            )

    training_seconds = time.perf_counter() - start_time

    true_labels, predictions = evaluate(
        model,
        test_loader,
        device,
    )

    accuracy = sum(
        actual == predicted
        for actual, predicted in zip(true_labels, predictions)
    ) / len(true_labels)

    print("\nTEST RESULTS")
    print("-" * 65)
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Test accuracy: {accuracy:.3f}")
    print("Class labels: 0 = sports, 1 = technology")

    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predictions,
            labels=[0, 1],
            target_names=["sports", "technology"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(true_labels, predictions, labels=[0, 1]))

    print("\nINDIVIDUAL TEST PREDICTIONS")
    print("-" * 65)

    for text, actual, predicted in zip(
        test_texts,
        true_labels,
        predictions,
    ):
        actual_name = "sports" if actual == 0 else "technology"
        predicted_name = "sports" if predicted == 0 else "technology"
        status = "correct" if actual == predicted else "incorrect"

        print(f"Text: {text}")
        print(
            f"Actual: {actual_name} | "
            f"Predicted: {predicted_name} | "
            f"Result: {status}\n"
        )

    print("LIMITATIONS")
    print("-" * 65)
    print(
        "This is a small educational dataset with only six test examples. "
        "Metrics can change substantially with a different split or seed."
    )
    print(
        "The model starts with randomly initialized token embeddings and "
        "Transformer weights. It does not use pretrained language-model "
        "knowledge."
    )


if __name__ == "__main__":
    main()
