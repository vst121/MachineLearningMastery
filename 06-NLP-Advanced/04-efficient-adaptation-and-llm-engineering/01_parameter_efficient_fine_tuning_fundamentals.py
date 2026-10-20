"""
Parameter-Efficient Fine-Tuning Fundamentals

Experiments:
- Compare full fine-tuning with a frozen-backbone baseline.
- Apply LoRA to a pretrained Transformer classification model.
- Measure trainable parameters and parameter storage requirements.
- Compare validation performance and training duration.
"""

from time import perf_counter

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODEL_NAME = "distilbert/distilbert-base-uncased"
SEED = 42
MAX_LENGTH = 128
OUTPUT_DIR = "outputs/peft-fundamentals"

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

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=SEED,
        stratify=labels,
    )

    def encode(texts, labels):
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = labels
        return Dataset.from_dict(encoded)

    return encode(train_texts, train_labels), encode(val_texts, val_labels)


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def estimate_parameter_storage(model):
    """Estimate parameter storage by dtype, excluding optimizer states."""
    total_bytes = 0
    trainable_bytes = 0

    for parameter in model.parameters():
        parameter_bytes = parameter.numel() * parameter.element_size()
        total_bytes += parameter_bytes

        if parameter.requires_grad:
            trainable_bytes += parameter_bytes

    return total_bytes, trainable_bytes


def compute_metrics(eval_prediction):
    """Calculate classification metrics from model logits."""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def create_model(strategy):
    """Create a full fine-tuning, frozen-backbone, or LoRA model."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    if strategy == "full_fine_tuning":
        return model

    if strategy == "frozen_backbone":
        for parameter in model.distilbert.parameters():
            parameter.requires_grad = False
        return model

    if strategy == "lora":
        # DistilBERT attention uses q_lin and v_lin projections.
        # The classification head remains trainable.
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
            modules_to_save=["pre_classifier", "classifier"],
        )
        return get_peft_model(model, config)

    raise ValueError(f"Unknown strategy: {strategy}")


def run_experiment(
    strategy,
    tokenizer,
    train_dataset,
    validation_dataset,
    experiment_number,
):
    """Train and evaluate one adaptation strategy."""
    set_seed(SEED)
    model = create_model(strategy)

    total_parameters, trainable_parameters = count_parameters(model)
    total_bytes, trainable_bytes = estimate_parameter_storage(model)

    model_output = f"{OUTPUT_DIR}/{experiment_number:02d}-{strategy}"

    training_args = TrainingArguments(
        output_dir=model_output,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=(
            2e-5 if strategy != "lora" else 1e-3
        ),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
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
    )

    print(f"\n--- Strategy: {strategy} ---")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(
        "Trainable percentage: "
        f"{100 * trainable_parameters / total_parameters:.2f}%"
    )
    print(f"Parameter storage: {total_bytes / 1024**2:.2f} MiB")
    print(f"Trainable parameter storage: {trainable_bytes / 1024**2:.2f} MiB")

    start_time = perf_counter()
    trainer.train()
    training_seconds = perf_counter() - start_time

    metrics = trainer.evaluate()

    result = {
        "strategy": strategy,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percent": 100 * trainable_parameters / total_parameters,
        "parameter_storage_mib": total_bytes / 1024**2,
        "trainable_storage_mib": trainable_bytes / 1024**2,
        "training_seconds": training_seconds,
        "validation_loss": metrics["eval_loss"],
        "accuracy": metrics["eval_accuracy"],
        "f1": metrics["eval_f1"],
    }

    print("\nResults:")
    for name, value in result.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    return result


def main():
    set_seed(SEED)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, validation_dataset = prepare_datasets(tokenizer)

    strategies = [
        "full_fine_tuning",
        "frozen_backbone",
        "lora",
    ]

    results = []

    for index, strategy in enumerate(strategies, start=1):
        result = run_experiment(
            strategy=strategy,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            experiment_number=index,
        )
        results.append(result)

    print("\n--- Summary ---")
    print(
        f"{'Strategy':<20} {'Trainable %':>12} "
        f"{'Accuracy':>10} {'F1':>8} {'Seconds':>10}"
    )
    print("-" * 64)

    for result in results:
        print(
            f"{result['strategy']:<20} "
            f"{result['trainable_percent']:>11.2f}% "
            f"{result['accuracy']:>10.3f} "
            f"{result['f1']:>8.3f} "
            f"{result['training_seconds']:>10.2f}"
        )


if __name__ == "__main__":
    main()
