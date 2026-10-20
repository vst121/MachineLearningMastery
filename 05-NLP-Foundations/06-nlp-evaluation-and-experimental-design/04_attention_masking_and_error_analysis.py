"""
Investigate Transformer attention masking and classification errors.

This experiment covers:
1. Implementing scaled dot-product attention with a padding mask.
2. Verifying that padded key positions receive zero attention weight.
3. Comparing attention distributions for different sequence lengths.
4. Training a small Transformer encoder text classifier.
5. Inspecting prediction confidence and classification errors.
6. Analyzing unknown tokens, padding, truncation, and token order.
7. Understanding the limitations of interpreting attention weights.
"""

import math
import random
import re
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
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def build_vocabulary(texts: list[str]) -> dict[str, int]:
    """Build a vocabulary using training texts only."""
    counts = Counter(
        token
        for text in texts
        for token in tokenize(text)
    )

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token in sorted(counts):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    return vocabulary


def encode_text(
    text: str,
    vocabulary: dict[str, int],
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> tuple[list[int], int, int]:
    """
    Encode text and report its effective length and number of unknown tokens.

    Truncation occurs before padding.
    """
    tokens = tokenize(text)
    truncated_tokens = tokens[:max_length]
    effective_length = len(truncated_tokens)

    unknown_count = sum(
        token not in vocabulary
        for token in truncated_tokens
    )

    token_ids = [
        vocabulary.get(token, vocabulary[UNK_TOKEN])
        for token in truncated_tokens
    ]

    if not token_ids:
        token_ids = [vocabulary[UNK_TOKEN]]
        effective_length = 1

    return token_ids, effective_length, unknown_count


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention with an optional padding mask.

    Args:
        queries: Tensor shaped (batch, query_length, head_dim).
        keys: Tensor shaped (batch, key_length, head_dim).
        values: Tensor shaped (batch, key_length, head_dim).
        key_padding_mask: Boolean tensor shaped (batch, key_length).
            True indicates a padded key position.

    Returns:
        Contextual representations and attention weights.
    """
    scores = queries @ keys.transpose(-2, -1)
    scores = scores / math.sqrt(queries.size(-1))

    if key_padding_mask is not None:
        scores = scores.masked_fill(
            key_padding_mask.unsqueeze(1),
            float("-inf"),
        )

    weights = torch.softmax(scores, dim=-1)
    contextual = weights @ values

    return contextual, weights


class TextDataset(Dataset):
    """Store encoded sequences and labels."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocabulary: dict[str, int],
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.encoded = [
            encode_text(text, vocabulary)
            for text in texts
        ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[list[int], int]:
        token_ids, _, _ = self.encoded[index]
        return token_ids, self.labels[index]


def collate_batch(
    batch: list[tuple[list[int], int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch and preserve each sequence's effective length."""
    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long,
    )

    max_length = max(lengths).item()
    padded = torch.full(
        (len(sequences), max_length),
        fill_value=0,
        dtype=torch.long,
    )

    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = torch.tensor(sequence)

    return (
        padded,
        torch.tensor(labels, dtype=torch.long),
        lengths,
    )


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal positional information."""

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

        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        sequence_length = embeddings.size(1)

        if sequence_length > self.encoding.size(1):
            raise ValueError("Input sequence exceeds the positional encoding limit.")

        return self.dropout(
            embeddings + self.encoding[:, :sequence_length, :]
        )


class TransformerTextClassifier(nn.Module):
    """Classify sequences with a Transformer encoder."""

    def __init__(
        self,
        vocabulary_size: int,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocabulary_size,
            EMBEDDING_DIM,
            padding_idx=0,
        )

        self.position_encoding = SinusoidalPositionalEncoding(
            embedding_dim=EMBEDDING_DIM,
            max_length=MAX_SEQUENCE_LENGTH,
            dropout=DROPOUT,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=DROPOUT,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_ENCODER_LAYERS,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(EMBEDDING_DIM, num_classes),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        padding_mask = token_ids.eq(0)

        embeddings = self.token_embedding(token_ids)
        embeddings = embeddings * math.sqrt(EMBEDDING_DIM)
        embeddings = self.position_encoding(embeddings)

        encoded = self.encoder(
            embeddings,
            src_key_padding_mask=padding_mask,
        )

        # Mask padding positions during pooling as well as in attention.
        valid_tokens = (~padding_mask).unsqueeze(-1).to(encoded.dtype)

        pooled = (encoded * valid_tokens).sum(dim=1)
        pooled = pooled / valid_tokens.sum(dim=1).clamp(min=1.0)

        return self.classifier(pooled)


def demonstrate_attention_masking() -> None:
    """Show that padded key positions receive zero attention."""
    print("ATTENTION MASKING DEMONSTRATION")
    print("=" * 65)

    torch.manual_seed(SEED)

    queries = torch.randn(1, 3, 4)
    keys = torch.randn(1, 4, 4)
    values = torch.randn(1, 4, 4)

    # The fourth key/value is padding.
    padding_mask = torch.tensor([[False, False, False, True]])

    _, weights_without_mask = scaled_dot_product_attention(
        queries,
        keys,
        values,
    )

    contextual_masked, weights_with_mask = scaled_dot_product_attention(
        queries,
        keys,
        values,
        key_padding_mask=padding_mask,
    )

    print("\nAttention weights without masking:")
    print(weights_without_mask[0].numpy().round(3))

    print("\nAttention weights with masking:")
    print(weights_with_mask[0].numpy().round(3))

    padded_weights = weights_with_mask[..., -1]

    print(
        "\nMaximum attention weight assigned to the padded key: "
        f"{padded_weights.max().item():.6f}"
    )
    print(
        "All padded-key weights are zero: "
        f"{bool(torch.allclose(padded_weights, torch.zeros_like(padded_weights)))}"
    )
    print(
        "All masked contextual values are finite: "
        f"{bool(torch.isfinite(contextual_masked).all())}"
    )

    assert torch.allclose(
        padded_weights,
        torch.zeros_like(padded_weights),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
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
def predict(
    model: nn.Module,
    token_ids: torch.Tensor,
    device: torch.device,
) -> tuple[int, float]:
    """Predict a class and return its softmax confidence."""
    model.eval()

    logits = model(token_ids.unsqueeze(0).to(device))
    probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_class = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_class].item())

    return predicted_class, confidence


def main() -> None:
    set_seed()
    demonstrate_attention_masking()

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

    vocabulary = build_vocabulary(train_texts)

    train_dataset = TextDataset(
        train_texts,
        train_labels,
        vocabulary,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TransformerTextClassifier(
        vocabulary_size=len(vocabulary),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()

    print("\nTRANSFORMER TRAINING")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_texts)}")
    print(f"Vocabulary size: {len(vocabulary)}")

    for epoch in range(1, EPOCHS + 1):
        loss, accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"loss={loss:.4f} | accuracy={accuracy:.3f}"
            )

    print("\nTEST SET ERROR ANALYSIS")
    print("=" * 65)

    true_labels: list[int] = []
    predicted_labels: list[int] = []

    for text, actual_label in zip(test_texts, test_labels):
        token_ids, effective_length, unknown_count = encode_text(
            text,
            vocabulary,
        )

        predicted_label, confidence = predict(
            model,
            torch.tensor(token_ids, dtype=torch.long),
            device,
        )

        true_labels.append(actual_label)
        predicted_labels.append(predicted_label)

        actual_name = "sports" if actual_label == 0 else "technology"
        predicted_name = "sports" if predicted_label == 0 else "technology"
        status = "CORRECT" if actual_label == predicted_label else "ERROR"

        print(f"\nText: {text}")
        print(f"Actual class: {actual_name}")
        print(f"Predicted class: {predicted_name}")
        print(f"Prediction confidence: {confidence:.3f}")
        print(f"Effective sequence length: {effective_length}")
        print(f"Unknown token count: {unknown_count}")
        print(f"Result: {status}")

    print("\nCLASSIFICATION REPORT")
    print("-" * 65)
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=[0, 1],
            target_names=["sports", "technology"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(true_labels, predicted_labels, labels=[0, 1]))

    print("\nSEQUENCE LENGTH AND TRUNCATION CHECKS")
    print("-" * 65)

    diagnostic_texts = [
        "The team won the match",
        "The software engineer deployed a new application",
        "unknownword anotherunknownword",
        " ".join(["token"] * (MAX_SEQUENCE_LENGTH + 5)),
    ]

    for text in diagnostic_texts:
        original_tokens = tokenize(text)
        token_ids, effective_length, unknown_count = encode_text(
            text,
            vocabulary,
        )
        truncated = len(original_tokens) > MAX_SEQUENCE_LENGTH
        padded_length = max(
            effective_length,
            1,
        )

        print(f"\nText: {text}")
        print(f"Original token count: {len(original_tokens)}")
        print(f"Encoded token count: {len(token_ids)}")
        print(f"Effective sequence length: {effective_length}")
        print(f"Unknown token count: {unknown_count}")
        print(f"Truncated: {truncated}")
        print(f"Padding needed in a batch: determined by batch maximum length")

    print("\nLIMITATIONS")
    print("-" * 65)
    print(
        "Attention weights show how the model distributes attention, "
        "but they are not a complete explanation of its predictions."
    )
    print(
        "The dataset has only six test examples. Accuracy, confidence, "
        "and error patterns are illustrative and statistically unstable."
    )
    print(
        "Unknown tokens, truncation, and sequence length are useful "
        "diagnostics, but they do not by themselves establish the cause "
        "of a classification error."
    )


if __name__ == "__main__":
    main()
