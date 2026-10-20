"""
Explore the fundamentals of self-attention.

This experiment covers:
1. Representing tokens as vectors.
2. Projecting token representations into queries, keys, and values.
3. Computing scaled dot-product attention.
4. Inspecting attention scores and attention weights.
5. Comparing token representations before and after attention.
6. Understanding how attention combines contextual information.
"""

import math

import torch
from torch import nn


SEED = 42
EMBEDDING_DIM = 8
NUM_TOKENS = 4


def set_seed(seed: int = SEED) -> None:
    """Make the experiment reproducible."""
    torch.manual_seed(seed)


class SelfAttention(nn.Module):
    """Implement single-head scaled dot-product self-attention."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)

        self.scale = math.sqrt(embedding_dim)

    def forward(
        self,
        token_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return contextual representations, attention weights, and scores.

        Args:
            token_embeddings: Tensor with shape (sequence_length, embedding_dim).

        Returns:
            contextual_embeddings: Tensor with shape
                (sequence_length, embedding_dim).
            attention_weights: Tensor with shape
                (sequence_length, sequence_length).
            attention_scores: Tensor with shape
                (sequence_length, sequence_length).
        """
        queries = self.query_projection(token_embeddings)
        keys = self.key_projection(token_embeddings)
        values = self.value_projection(token_embeddings)

        attention_scores = queries @ keys.T
        attention_scores = attention_scores / self.scale

        attention_weights = torch.softmax(attention_scores, dim=-1)

        contextual_embeddings = attention_weights @ values

        return contextual_embeddings, attention_weights, attention_scores


def print_matrix(name: str, tensor: torch.Tensor) -> None:
    """Print a labeled tensor with readable precision."""
    print(f"\n{name}")
    print(tensor.detach().cpu().numpy().round(3))


def main() -> None:
    set_seed()

    # Each row represents one token in a short sentence.
    tokens = ["AI", "systems", "learn", "patterns"]

    # In a real NLP model, these vectors would normally come from an
    # embedding layer. Here, they are fixed to isolate the attention mechanism.
    token_embeddings = torch.tensor(
        [
            [0.8, 0.1, 0.3, 0.5, 0.2, 0.7, 0.4, 0.6],
            [0.2, 0.9, 0.4, 0.1, 0.8, 0.3, 0.5, 0.7],
            [0.6, 0.2, 0.9, 0.4, 0.3, 0.8, 0.1, 0.5],
            [0.3, 0.7, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1],
        ],
        dtype=torch.float32,
    )

    attention = SelfAttention(embedding_dim=EMBEDDING_DIM)
    attention.eval()

    with torch.no_grad():
        contextual_embeddings, attention_weights, attention_scores = attention(
            token_embeddings
        )

    print("SELF-ATTENTION FUNDAMENTALS")
    print("=" * 60)

    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")

    print_matrix("Input token embeddings:", token_embeddings)
    print_matrix("Scaled attention scores:", attention_scores)
    print_matrix("Attention weights:", attention_weights)
    print_matrix("Contextual token representations:", contextual_embeddings)

    print("\nATTENTION WEIGHTS BY TOKEN")
    print("-" * 60)

    print(f"{'Query token':<15}", end="")
    for token in tokens:
        print(f"{token:>12}", end="")
    print()

    for index, token in enumerate(tokens):
        print(f"{token:<15}", end="")
        for weight in attention_weights[index]:
            print(f"{weight.item():>12.3f}", end="")
        print()

    print("\nATTENTION WEIGHT CHECKS")
    print("-" * 60)

    row_sums = attention_weights.sum(dim=-1)

    print(f"Attention matrix shape: {tuple(attention_weights.shape)}")
    print(f"Each row sums to one: {torch.allclose(row_sums, torch.ones_like(row_sums))}")
    print(f"All weights non-negative: {bool(torch.all(attention_weights >= 0))}")

    print("\nCONTEXTUAL REPRESENTATION CHANGES")
    print("-" * 60)

    input_norms = torch.linalg.vector_norm(token_embeddings, dim=-1)
    output_norms = torch.linalg.vector_norm(contextual_embeddings, dim=-1)
    representation_changes = torch.linalg.vector_norm(
        contextual_embeddings - token_embeddings,
        dim=-1,
    )

    for index, token in enumerate(tokens):
        print(
            f"{token:<12} "
            f"input_norm={input_norms[index].item():.3f}, "
            f"output_norm={output_norms[index].item():.3f}, "
            f"change={representation_changes[index].item():.3f}"
        )

    print("\nINTERPRETATION")
    print("-" * 60)
    print(
        "Each row of the attention matrix describes how strongly one query "
        "token attends to every token in the sequence."
    )
    print(
        "The contextual representation for each token is a weighted "
        "combination of the projected value vectors."
    )
    print(
        "The model parameters are randomly initialized, so these weights "
        "are not meaningful linguistic explanations. Training is required "
        "for the projections to learn task-relevant relationships."
    )


if __name__ == "__main__":
    main()
