"""
Build and validate Transformer encoder and decoder components.

This experiment covers:
1. Building encoder blocks with self-attention and feed-forward networks.
2. Building decoder blocks with causal self-attention and cross-attention.
3. Applying residual connections and layer normalization.
4. Understanding causal attention masks and tensor dimensions.
5. Validating the architecture with PyTorch Transformer modules.
6. Distinguishing encoder-only, decoder-only, and encoder-decoder architectures.
"""

import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    """Implement a Transformer encoder block."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, model_dim),
        )

        self.norm_attention = nn.LayerNorm(model_dim)
        self.norm_feedforward = nn.LayerNorm(model_dim)
        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_feedforward = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output, _ = self.self_attention(
            query=inputs,
            key=inputs,
            value=inputs,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        hidden = self.norm_attention(
            inputs + self.dropout_attention(attention_output)
        )

        feedforward_output = self.feedforward(hidden)

        output = self.norm_feedforward(
            hidden + self.dropout_feedforward(feedforward_output)
        )

        return output


class TransformerDecoderBlock(nn.Module):
    """Implement a Transformer decoder block."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, model_dim),
        )

        self.norm_self_attention = nn.LayerNorm(model_dim)
        self.norm_cross_attention = nn.LayerNorm(model_dim)
        self.norm_feedforward = nn.LayerNorm(model_dim)

        self.dropout_self_attention = nn.Dropout(dropout)
        self.dropout_cross_attention = nn.Dropout(dropout)
        self.dropout_feedforward = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Decoder tokens attend only to permitted target positions.
        self_attention_output, _ = self.self_attention(
            query=target,
            key=target,
            value=target,
            attn_mask=causal_mask,
            key_padding_mask=target_padding_mask,
            need_weights=False,
        )

        hidden = self.norm_self_attention(
            target + self.dropout_self_attention(self_attention_output)
        )

        # Decoder queries attend to encoder output representations.
        cross_attention_output, _ = self.cross_attention(
            query=hidden,
            key=memory,
            value=memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )

        hidden = self.norm_cross_attention(
            hidden + self.dropout_cross_attention(cross_attention_output)
        )

        feedforward_output = self.feedforward(hidden)

        output = self.norm_feedforward(
            hidden + self.dropout_feedforward(feedforward_output)
        )

        return output


def create_causal_mask(
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a Boolean causal mask.

    True indicates a position that must not be attended to.
    """
    return torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )


def main() -> None:
    torch.manual_seed(42)

    batch_size = 2
    source_length = 5
    target_length = 4
    model_dim = 16
    num_heads = 4
    feedforward_dim = 64
    dropout = 0.0

    # Batch-first input: (batch_size, sequence_length, model_dim).
    source = torch.randn(batch_size, source_length, model_dim)
    target = torch.randn(batch_size, target_length, model_dim)

    # True entries in key padding masks identify positions to ignore.
    source_padding_mask = torch.tensor(
        [
            [False, False, False, False, False],
            [False, False, False, True, True],
        ]
    )

    target_padding_mask = torch.tensor(
        [
            [False, False, False, False],
            [False, False, False, True],
        ]
    )

    causal_mask = create_causal_mask(
        target_length,
        target.device,
    )

    print("=== Transformer Architecture ===")
    print(f"Source shape: {tuple(source.shape)}")
    print(f"Target shape: {tuple(target.shape)}")
    print(f"Model dimension: {model_dim}")
    print(f"Attention heads: {num_heads}")
    print(f"Feed-forward dimension: {feedforward_dim}")

    print("\nCausal attention mask:")
    print(causal_mask.int())

    # 1. Custom Transformer encoder block.
    encoder_block = TransformerEncoderBlock(
        model_dim=model_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
    )

    encoder_block.eval()

    with torch.no_grad():
        memory = encoder_block(
            source,
            key_padding_mask=source_padding_mask,
        )

    print("\n=== Custom Encoder Block ===")
    print(f"Encoder output shape: {tuple(memory.shape)}")

    assert memory.shape == source.shape
    assert torch.isfinite(memory).all()

    # 2. Custom Transformer decoder block.
    decoder_block = TransformerDecoderBlock(
        model_dim=model_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
    )

    decoder_block.eval()

    with torch.no_grad():
        decoded = decoder_block(
            target=target,
            memory=memory,
            causal_mask=causal_mask,
            target_padding_mask=target_padding_mask,
            memory_padding_mask=source_padding_mask,
        )

    print("\n=== Custom Decoder Block ===")
    print(f"Decoder output shape: {tuple(decoded.shape)}")

    assert decoded.shape == target.shape
    assert torch.isfinite(decoded).all()

    # 3. Validate with PyTorch's built-in Transformer encoder.
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=model_dim,
        nhead=num_heads,
        dim_feedforward=feedforward_dim,
        dropout=dropout,
        batch_first=True,
    )

    built_in_encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=2,
        enable_nested_tensor=False,
    )
    built_in_encoder.eval()

    with torch.no_grad():
        built_in_memory = built_in_encoder(
            source,
            src_key_padding_mask=source_padding_mask,
        )

    print("\n=== Built-in PyTorch Encoder ===")
    print(f"Encoder output shape: {tuple(built_in_memory.shape)}")

    assert built_in_memory.shape == source.shape
    assert torch.isfinite(built_in_memory).all()

    # 4. Validate with PyTorch's built-in Transformer decoder.
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=model_dim,
        nhead=num_heads,
        dim_feedforward=feedforward_dim,
        dropout=dropout,
        batch_first=True,
    )

    built_in_decoder = nn.TransformerDecoder(
        decoder_layer,
        num_layers=2,
    )
    built_in_decoder.eval()

    with torch.no_grad():
        built_in_decoded = built_in_decoder(
            tgt=target,
            memory=built_in_memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )

    print("\n=== Built-in PyTorch Decoder ===")
    print(f"Decoder output shape: {tuple(built_in_decoded.shape)}")

    assert built_in_decoded.shape == target.shape
    assert torch.isfinite(built_in_decoded).all()

    # 5. Verify that the causal mask blocks future positions.
    assert not causal_mask[0, 0]
    assert causal_mask[0, 1]
    assert causal_mask[0, 2]
    assert not causal_mask[2, 1]
    assert causal_mask[2, 3]

    print("\nAll Transformer architecture checks passed.")


if __name__ == "__main__":
    main()
