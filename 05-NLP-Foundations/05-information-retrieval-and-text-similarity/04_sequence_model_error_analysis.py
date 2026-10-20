"""
Analyze errors and sequence-handling behavior in a recurrent text classifier.

This experiment covers:
1. Training an LSTM text classifier on a small labeled dataset.
2. Evaluating held-out classification performance.
3. Inspecting false positives, false negatives, and prediction confidence.
4. Comparing predictions for truncated and untruncated documents.
5. Examining unknown tokens and padding behavior.
6. Measuring how token order affects recurrent model predictions.
7. Understanding the limitations of error analysis on tiny datasets.
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
    """Build a deterministic vocabulary from training documents only."""
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
    max_length: int,
) -> tuple[list[int], int]:
    """
    Encode a document and return token IDs plus its effective length.

    The effective length excludes padding and reflects truncation.
    """
    tokens = tokenize(text)[:max_length]
    unknown_id = vocabulary[UNK_TOKEN]

    token_ids = [
        vocabulary.get(token, unknown_id)
        for token in tokens
    ]

    effective_length = max(1, len(token_ids))
    token_ids.extend(
        [vocabulary[PAD_TOKEN]] * (max_length - len(token_ids))
    )

    return token_ids, effective_length


class TextDataset(Dataset):
    """Store encoded sequences, labels, and effective lengths."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocabulary: dict[str, int],
        max_length: int,
    ) -> None:
        encoded = [
            encode_text(text, vocabulary, max_length)
            for text in texts
        ]

        self.features = torch.tensor(
            [item[0] for item in encoded],
            dtype=torch.long,
        )
        self.lengths = torch.tensor(
            [item[1] for item in encoded],
            dtype=torch.long,
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.features[index],
            self.labels[index],
            self.lengths[index],
        )


def collate_batch(batch):
    """Combine examples into a batch."""
    token_ids, labels, lengths = zip(*batch)

    return (
        torch.stack(token_ids),
        torch.stack(labels),
        torch.stack(lengths),
    )


class LSTMClassifier(nn.Module):
    """Classify documents using a packed LSTM sequence encoder."""

    def __init__(
        self,
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
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

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

        _, (hidden, _) = self.lstm(packed)
        final_hidden = hidden[-1]

        return self.classifier(final_hidden), final_hidden


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Perform one training epoch and return mean loss."""
    model.train()

    total_loss = 0.0
    total_examples = 0

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

    return total_loss / total_examples


@torch.no_grad()
def predict(
    model: nn.Module,
    texts: list[str],
    vocabulary: dict[str, int],
    max_length: int,
    batch_size: int = 8,
) -> dict:
    """Predict class labels and probabilities for arbitrary documents."""
    model.eval()

    encoded = [
        encode_text(text, vocabulary, max_length)
        for text in texts
    ]

    token_ids = torch.tensor(
        [item[0] for item in encoded],
        dtype=torch.long,
    )
    lengths = torch.tensor(
        [item[1] for item in encoded],
        dtype=torch.long,
    )

    all_probabilities = []
    all_hidden_states = []

    for start in range(0, len(texts), batch_size):
        end = start + batch_size

        batch_ids = token_ids[start:end].to(DEVICE)
        batch_lengths = lengths[start:end]

        logits, hidden = model(batch_ids, batch_lengths)
        probabilities = torch.softmax(logits, dim=1)

        all_probabilities.extend(probabilities.cpu().tolist())
        all_hidden_states.extend(hidden.cpu().tolist())

    probability_array = np.asarray(all_probabilities)

    return {
        "probabilities": probability_array,
        "predictions": probability_array.argmax(axis=1).tolist(),
        "confidences": probability_array.max(axis=1).tolist(),
        "hidden_states": np.asarray(all_hidden_states),
        "encoded": [item[0] for item in encoded],
        "lengths": [item[1] for item in encoded],
    }


def known_and_unknown_tokens(
    text: str,
    vocabulary: dict[str, int],
) -> tuple[list[str], list[str]]:
    """Separate tokens found in the vocabulary from unseen tokens."""
    known = []
    unknown = []

    for token in tokenize(text):
        if token in vocabulary:
            known.append(token)
        else:
            unknown.append(token)

    return known, unknown


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
    embedding_dim = 32
    hidden_dim = 48
    batch_size = 4
    epochs = 25
    learning_rate = 0.01

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=SEED,
        stratify=labels,
    )

    # Do not build the vocabulary using the held-out test documents.
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

    model = LSTMClassifier(
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

    print("=" * 72)
    print("TRAINING")
    print("=" * 72)
    print(f"Training documents: {len(train_dataset)}")
    print(f"Test documents: {len(test_dataset)}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Device: {DEVICE}")

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} | loss={loss:.4f}")

    print("\n" + "=" * 72)
    print("HELD-OUT EVALUATION")
    print("=" * 72)

    test_results = predict(
        model,
        test_texts,
        vocabulary,
        max_length,
        batch_size=batch_size,
    )

    y_pred = test_results["predictions"]

    print(f"Accuracy: {accuracy_score(test_labels, y_pred):.3f}")
    print(
        classification_report(
            test_labels,
            y_pred,
            labels=[0, 1],
            target_names=label_names,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(test_labels, y_pred, labels=[0, 1]))

    print("\n" + "=" * 72)
    print("1. FALSE POSITIVES AND FALSE NEGATIVES")
    print("=" * 72)

    for text, actual, predicted, confidence in zip(
        test_texts,
        test_labels,
        y_pred,
        test_results["confidences"],
    ):
        if actual != predicted:
            error_type = (
                "FALSE POSITIVE for technology"
                if actual == 0 and predicted == 1
                else "FALSE NEGATIVE for technology"
            )

            print(f"\n[{error_type}]")
            print(f"Text: {text}")
            print(f"Actual: {label_names[actual]}")
            print(f"Predicted: {label_names[predicted]}")
            print(f"Confidence: {confidence:.3f}")

    print("\n" + "=" * 72)
    print("2. UNKNOWN VOCABULARY")
    print("=" * 72)

    unknown_examples = [
        "the goalkeeper analyzed quantum robotics",
        "the quantum processor optimized neural networks",
        "the striker scored an incredible goal",
    ]

    for text in unknown_examples:
        known, unknown = known_and_unknown_tokens(text, vocabulary)
        print(f"\nText: {text}")
        print(f"Known tokens: {known}")
        print(f"Unknown tokens: {unknown}")

    unknown_results = predict(
        model,
        unknown_examples,
        vocabulary,
        max_length,
    )

    for text, predicted, confidence in zip(
        unknown_examples,
        unknown_results["predictions"],
        unknown_results["confidences"],
    ):
        print(
            f"Prediction: {label_names[predicted]} "
            f"(confidence={confidence:.3f}) | {text}"
        )

    print("\n" + "=" * 72)
    print("3. TRUNCATION SENSITIVITY")
    print("=" * 72)

    long_text = (
        "the team trained for many hours before the tournament "
        "and the goalkeeper scored a decisive goal"
    )

    short_limit = 5
    full_limit = 20

    short_result = predict(
        model,
        [long_text],
        vocabulary,
        short_limit,
    )
    full_result = predict(
        model,
        [long_text],
        vocabulary,
        full_limit,
    )

    print(f"Text: {long_text}")
    print(f"Tokens: {tokenize(long_text)}")
    print(f"\nWith max_length={short_limit}:")
    print(f"Encoded length: {short_result['lengths'][0]}")
    print(f"Prediction: {label_names[short_result['predictions'][0]]}")
    print(
        "Probabilities:",
        short_result["probabilities"][0].round(3).tolist(),
    )

    print(f"\nWith max_length={full_limit}:")
    print(f"Encoded length: {full_result['lengths'][0]}")
    print(f"Prediction: {label_names[full_result['predictions'][0]]}")
    print(
        "Probabilities:",
        full_result["probabilities"][0].round(3).tolist(),
    )

    print("\n" + "=" * 72)
    print("4. PADDING AND SEQUENCE LENGTHS")
    print("=" * 72)

    length_examples = [
        "team won",
        "the software fixed the critical bug",
        "the player scored a goal in the final match",
    ]

    length_results = predict(
        model,
        length_examples,
        vocabulary,
        max_length=10,
    )

    for text, encoded, length in zip(
        length_examples,
        length_results["encoded"],
        length_results["lengths"],
    ):
        print(f"\nText: {text}")
        print(f"Encoded sequence: {encoded}")
        print(f"Effective length: {length}")
        print(f"Padding count: {len(encoded) - length}")

    print("\n" + "=" * 72)
    print("5. TOKEN ORDER SENSITIVITY")
    print("=" * 72)

    order_examples = [
        "team won the match",
        "match won the team",
        "software fixed the bug",
        "bug fixed the software",
    ]

    order_results = predict(
        model,
        order_examples,
        vocabulary,
        max_length,
    )

    for text, predicted, probabilities in zip(
        order_examples,
        order_results["predictions"],
        order_results["probabilities"],
    ):
        print(f"\nText: {text}")
        print(f"Prediction: {label_names[predicted]}")
        print(f"Probabilities: {probabilities.round(3).tolist()}")

    print("\n" + "=" * 72)
    print("6. HIDDEN REPRESENTATIONS")
    print("=" * 72)

    hidden_states = test_results["hidden_states"]

    print(f"Hidden-state matrix shape: {hidden_states.shape}")
    print(
        "Mean absolute hidden-state activation:",
        f"{np.abs(hidden_states).mean():.4f}",
    )
    print(
        "Hidden-state standard deviation:",
        f"{hidden_states.std():.4f}",
    )

    print("\n" + "=" * 72)
    print("INTERPRETATION AND LIMITATIONS")
    print("=" * 72)
    print("- Inspect individual mistakes instead of relying only on accuracy.")
    print("- Unknown words collapse to a shared UNK embedding.")
    print("- Truncation may remove information relevant to the label.")
    print("- Packed sequences prevent right-padding from updating the LSTM state.")
    print("- Recurrent models preserve order but do not automatically understand meaning.")
    print("- Tiny datasets and repeated example patterns can lead to overfitting.")
    print("- Diagnostic examples are exploratory, not statistically conclusive.")


if __name__ == "__main__":
    main()
