
"""
Training, Validation, and Checkpointing

Experiments:
- Train a pretrained Transformer for sentiment classification.
- Track training and validation loss across epochs.
- Apply early stopping when validation loss stops improving.
- Save and restore the best model checkpoint.
- Compare the best checkpoint with the final training state.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODEL_NAME = "distilbert/distilbert-base-uncased"
SEED = 42
MAX_LENGTH = 128
OUTPUT_DIR = Path("outputs/training-validation-checkpointing")

SAMPLES = [
    ("I absolutely loved this movie.", 1),
    ("The story was engaging and beautifully told.", 1),
    ("The acting was excellent.", 1),
    ("A wonderful film with a satisfying ending.", 1),
    ("I would happily watch it again.", 1),
    ("The characters were believable and memorable.", 1),
    ("It was entertaining from beginning to end.", 1),
    ("The soundtrack made the movie even better.", 1),
    ("A thoughtful and enjoyable experience.", 1),
    ("The performances were impressive.", 1),
    ("This movie was boring and too long.", 0),
    ("I disliked almost everything about it.", 0),
    ("The plot was confusing and poorly written.", 0),
    ("The acting felt unnatural.", 0),
    ("I regret spending time on this film.", 0),
    ("The ending was disappointing.", 0),
    ("It was a dull and forgettable movie.", 0),
    ("The characters were flat and uninteresting.", 0),
    ("Nothing about the story made sense.", 0),
    ("I would not recommend this film.", 0),
    ("The film was surprisingly moving.", 1),
    ("A clever script and strong performances.", 1),
    ("The pacing was excellent and the story worked.", 1),
    ("I found the movie charming and funny.", 1),
    ("The film had heart and originality.", 1),
    ("The dialogue was awkward and repetitive.", 0),
    ("The movie failed to hold my attention.", 0),
    ("The story was predictable and lifeless.", 0),
    ("The film felt rushed and incomplete.", 0),
    ("A disappointing waste of a good premise.", 0),
    ("I enjoyed the emotional depth of the story.", 1),
    ("The cinematography was stunning.", 1),
    ("Every scene contributed to a compelling story.", 1),
    ("The film was warm, funny, and sincere.", 1),
    ("The cast worked together exceptionally well.", 1),
    ("The dialogue was tedious and the plot was weak.", 0),
    ("The movie was messy and frustrating.", 0),
    ("The performances lacked energy.", 0),
    ("I found the entire film unconvincing.", 0),
    ("The story dragged without a clear purpose.", 0),
]


def prepare_datasets(tokenizer):
    """Create reproducible training and validation datasets."""
    texts = [text for text, _ in SAMPLES]
    labels = [label for _, label in SAMPLES]

    train_texts, validation_texts, train_labels, validation_labels = (
        train_test_split(
            texts,
            labels,
            test_size=0.25,
            random_state=SEED,
            stratify=labels,
        )
    )

    def encode(texts, labels):
        tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        tokens["labels"] = labels
        return Dataset.from_dict(tokens)

    return (
        encode(train_texts, train_labels),
        encode(validation_texts, validation_labels),
    )


def compute_metrics(eval_prediction):
    """Calculate classification metrics from model logits."""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def print_training_history(trainer):
    """Display epoch-level training and validation metrics."""
    print("\n--- Training history ---")

    for record in trainer.state.log_history:
        if "epoch" not in record:
            continue

        epoch = record["epoch"]

        if "loss" in record:
            print(
                f"Epoch {epoch:.2f} | "
                f"Training loss: {record['loss']:.4f}"
            )

        if "eval_loss" in record:
            print(
                f"Epoch {epoch:.2f} | "
                f"Validation loss: {record['eval_loss']:.4f} | "
                f"Accuracy: {record.get('eval_accuracy', float('nan')):.3f} | "
                f"F1: {record.get('eval_f1', float('nan')):.3f}"
            )


def main():
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    train_dataset, validation_dataset = prepare_datasets(tokenizer)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=10,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.0,
            )
        ],
    )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")
    print("\n--- Training with early stopping ---")

    start_time = perf_counter()
    trainer.train()
    elapsed_seconds = perf_counter() - start_time

    # load_best_model_at_end restores the checkpoint with the best
    # monitored validation metric.
    best_checkpoint = trainer.state.best_model_checkpoint

    print_training_history(trainer)

    print("\n--- Best checkpoint ---")
    print(f"Best checkpoint path: {best_checkpoint}")
    print(f"Best validation metric: {trainer.state.best_metric}")
    print(f"Completed training epochs: {trainer.state.epoch:.2f}")
    print(f"Training duration: {elapsed_seconds:.2f} seconds")

    print("\n--- Final evaluation of restored best model ---")
    metrics = trainer.evaluate()

    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")

    final_model_dir = OUTPUT_DIR / "best-model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"\nBest model and tokenizer saved to: {final_model_dir}")


if __name__ == "__main__":
    main()
