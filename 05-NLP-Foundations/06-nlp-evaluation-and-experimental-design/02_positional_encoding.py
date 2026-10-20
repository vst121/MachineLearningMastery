"""
Explore positional representations in Transformer models.

This experiment covers:
1. Why self-attention needs explicit positional information.
2. Implementing sinusoidal positional encoding.
3. Implementing learned positional embeddings.
4. Comparing positional vectors across sequence positions.
5. Combining token embeddings with positional representations.
6. Investigating how positional information changes token representations.
"""

import math

import torch
from torch import nn


SEED = 42
EMBEDDING_DIM = 8
MAX_SEQUENCE_LENGTH = 12
NUM_TOKENS = 4


def set_seed(seed: int = SEED) -> None:
    """Make the experiment reproducible."""
    torch.manual_seed(seed)


class SinusoidalPositionalEncoding(nn.Module):
    """Generate fixed positional encodings using sine and cosine functions."""

    def __init__(
        self,
        embedding_dim: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()

        positions = torch.arange(
            max_sequence_length,
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

        positional_encoding = torch.zeros(
            max_sequence_length,
            embedding_dim,
            dtype=torch.float32,
        )

        positional_encoding[:, 0::2] = torch.sin(angles[:, 0::2])
        positional_encoding[:, 1::2] = torch.cos(angles[:, 1::2])

        self.register_buffer(
            "positional_encoding",
            positional_encoding.unsqueeze(0),
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to token embeddings."""
        sequence_length = token_embeddings.size(1)

        if sequence_length > self.positional_encoding.size(1):
            raise ValueError(
                f"Sequence length {sequence_length} exceeds maximum "
                f"length {self.positional_encoding.size(1)}."
            )

        positions = self.positional_encoding[:, :sequence_length, :]

        return token_embeddings + positions


class LearnedPositionalEncoding(nn.Module):
    """Learn a separate positional vector for each supported position."""

    def __init__(
        self,
        embedding_dim: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()

        self.position_embeddings = nn.Embedding(
            max_sequence_length,
            embedding_dim,
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings to token embeddings."""
        sequence_length = token_embeddings.size(1)

        if sequence_length > self.position_embeddings.num_embeddings:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds maximum "
                f"length {self.position_embeddings.num_embeddings}."
            )

        positions = torch.arange(
            sequence_length,
            device=token_embeddings.device,
        )

        position_vectors = self.position_embeddings(positions).unsqueeze(0)

        return token_embeddings + position_vectors


def print_matrix(name: str, tensor: torch.Tensor) -> None:
    """Print a tensor with readable precision."""
    print(f"\n{name}")
    print(tensor.detach().cpu().numpy().round(3))


def main() -> None:
    set_seed()

    tokens = ["AI", "systems", "learn", "patterns"]

    # Shape: (batch_size=1, sequence_length=4, embedding_dim=8).
    token_embeddings = torch.tensor(
        [
            [
                [0.8, 0.1, 0.3, 0.5, 0.2, 0.7, 0.4, 0.6],
                [0.2, 0.9, 0.4, 0.1, 0.8, 0.3, 0.5, 0.7],
                [0.6, 0.2, 0.9, 0.4, 0.3, 0.8, 0.1, 0.5],
                [0.3, 0.7, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1],
            ]
        ],
        dtype=torch.float32,
    )

    sinusoidal = SinusoidalPositionalEncoding(
        embedding_dim=EMBEDDING_DIM,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    learned = LearnedPositionalEncoding(
        embedding_dim=EMBEDDING_DIM,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    # Evaluation mode makes the intended inference behavior explicit.
    sinusoidal.eval()
    learned.eval()

    with torch.no_grad():
        sinusoidal_output = sinusoidal(token_embeddings)
        learned_output = learned(token_embeddings)

        sinusoidal_vectors = sinusoidal.positional_encoding[
            :, : len(tokens), :
        ]

        learned_positions = torch.arange(len(tokens))
        learned_vectors = learned.position_embeddings(
            learned_positions
        ).unsqueeze(0)

    print("POSITIONAL ENCODING FUNDAMENTALS")
    print("=" * 65)

    print(f"Tokens: {tokens}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Maximum supported sequence length: {MAX_SEQUENCE_LENGTH}")
    print(f"Input tensor shape: {tuple(token_embeddings.shape)}")

    print_matrix("Token embeddings:", token_embeddings[0])
    print_matrix("Sinusoidal positional vectors:", sinusoidal_vectors[0])
    print_matrix("Token embeddings + sinusoidal positions:", sinusoidal_output[0])
    print_matrix("Learned positional vectors (random initialization):", learned_vectors[0])
    print_matrix("Token embeddings + learned positions:", learned_output[0])

    print("\nPOSITIONAL VECTOR DISTANCES")
    print("-" * 65)

    for name, vectors in (
        ("Sinusoidal", sinusoidal_vectors[0]),
        ("Learned", learned_vectors[0]),
    ):
        print(f"\n{name} encoding:")
        for index in range(len(tokens) - 1):
            distance = torch.linalg.vector_norm(
                vectors[index] - vectors[index + 1]
            )
            print(
                f"  Position {index} to {index + 1}: "
                f"{distance.item():.3f}"
            )

    print("\nREPRESENTATION CHANGE")
    print("-" * 65)

    sinusoidal_change = torch.linalg.vector_norm(
        sinusoidal_output - token_embeddings,
        dim=-1,
    )[0]

    learned_change = torch.linalg.vector_norm(
        learned_output - token_embeddings,
        dim=-1,
    )[0]

    for index, token in enumerate(tokens):
        print(
            f"{token:<12} "
            f"sinusoidal_change={sinusoidal_change[index].item():.3f}, "
            f"learned_change={learned_change[index].item():.3f}"
        )

    print("\nKEY OBSERVATIONS")
    print("-" * 65)
    print(
        "Sinusoidal encodings are calculated from fixed mathematical "
        "functions and do not require training."
    )
    print(
        "Learned positional embeddings are trainable parameters. Their "
        "initial values are random until optimized during training."
    )
    print(
        "Both approaches add position-dependent information to token "
        "embeddings, allowing a Transformer to distinguish positions."
    )
    print(
        "This experiment does not train a model, so the learned positional "
        "vectors should not be interpreted as meaningful representations."
    )


if __name__ == "__main__":
    main()
