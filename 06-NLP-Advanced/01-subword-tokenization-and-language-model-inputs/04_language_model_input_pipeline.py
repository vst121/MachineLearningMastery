"""
Language Model Input Pipeline

Experiments:
- Build a reusable pipeline for preparing raw text for a pretrained model.
- Batch variable-length sequences with padding and truncation.
- Validate input IDs, attention masks, and tensor dimensions.
- Compare padding efficiency across batches.
- Demonstrate a forward pass through a pretrained language model.
"""

from dataclasses import dataclass

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
)


MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 64
BATCH_SIZE = 2

TEXTS = [
    "Subword tokenization converts text into model-readable units.",
    "Attention masks distinguish meaningful tokens from padding.",
    "Pretrained language models learn contextual representations.",
    "Short sequences require less computation than long sequences.",
    "A robust pipeline validates model inputs before inference.",
]


@dataclass
class PipelineConfig:
    """Configuration for language model input preparation."""

    model_name: str = MODEL_NAME
    max_length: int = MAX_LENGTH
    batch_size: int = BATCH_SIZE


class LanguageModelInputPipeline:
    """Prepare and validate batches for a pretrained language model."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def prepare_batch(self, texts: list[str]) -> BatchEncoding:
        """Tokenize text and create padded PyTorch tensors."""
        if not texts:
            raise ValueError("The input batch must not be empty.")

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every input item must be a string.")

        return self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    def validate_batch(self, batch: BatchEncoding) -> None:
        """Check the structural validity of model inputs."""
        required_keys = {"input_ids", "attention_mask"}
        missing_keys = required_keys - set(batch.keys())

        if missing_keys:
            raise ValueError(f"Missing required inputs: {missing_keys}")

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if input_ids.ndim != 2:
            raise ValueError("input_ids must be a two-dimensional tensor.")

        if attention_mask.shape != input_ids.shape:
            raise ValueError("Input IDs and attention masks must match.")

        if input_ids.shape[1] > self.config.max_length:
            raise ValueError("Sequence exceeds the configured maximum.")

        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("Attention masks must contain only 0 and 1.")

        if not torch.all(attention_mask.sum(dim=1) > 0):
            raise ValueError("Each sequence must contain a real token.")

    def iter_batches(self, texts: list[str]):
        """Yield prepared batches without processing the entire dataset at once."""
        for start in range(0, len(texts), self.config.batch_size):
            yield self.prepare_batch(
                texts[start : start + self.config.batch_size]
            )


def inspect_batch(
    tokenizer,
    batch: BatchEncoding,
    batch_number: int,
) -> None:
    """Print useful diagnostics for one batch."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    total_positions = attention_mask.numel()
    real_tokens = int(attention_mask.sum().item())
    padding_tokens = total_positions - real_tokens
    efficiency = real_tokens / total_positions if total_positions else 0.0

    print(f"\n--- Batch {batch_number} ---")
    print("Input shape:", tuple(input_ids.shape))
    print("Attention mask shape:", tuple(attention_mask.shape))
    print("Input IDs:")
    print(input_ids)
    print("Attention mask:")
    print(attention_mask)

    print("\nDecoded sequences:")
    for row in input_ids:
        print(" ", tokenizer.decode(row, skip_special_tokens=True))

    print("\nReal tokens:", real_tokens)
    print("Padding tokens:", padding_tokens)
    print(f"Padding efficiency: {efficiency:.2%}")


def run_model_forward_pass(
    config: PipelineConfig,
    batch: BatchEncoding,
) -> None:
    """Run inference and inspect contextual representation dimensions."""
    model = AutoModel.from_pretrained(config.model_name)
    model.eval()

    with torch.inference_mode():
        outputs = model(**batch)

    print("\n--- Model Forward Pass ---")
    print("Last hidden state shape:",
          tuple(outputs.last_hidden_state.shape))

    # CLS provides a convenient sequence-level representation for inspection.
    cls_representations = outputs.last_hidden_state[:, 0, :]
    print("CLS representation shape:",
          tuple(cls_representations.shape))


def main() -> None:
    config = PipelineConfig()
    pipeline = LanguageModelInputPipeline(config)

    all_batches = list(pipeline.iter_batches(TEXTS))

    for batch_number, batch in enumerate(all_batches, start=1):
        pipeline.validate_batch(batch)
        inspect_batch(pipeline.tokenizer, batch, batch_number)

    if all_batches:
        run_model_forward_pass(config, all_batches[0])


if __name__ == "__main__":
    main()
