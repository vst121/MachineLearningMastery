"""
LoRA and Adapter Comparison

Experiments:
- Compare full fine-tuning, frozen-backbone training, and LoRA.
- Investigate how LoRA rank changes trainable parameter counts.
- Compare LoRA scaling configurations.
- Measure validation metrics, training duration, and parameter storage.
"""

from pathlib import Path
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
OUTPUT_DIR = Path("outputs/lora-adapter-comparison")

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
    """Create identical stratified splits for all configurations."""
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


def compute_metrics(eval_prediction):
    """Calculate classification metrics from model logits."""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def build_model(strategy, rank=8, alpha=16):
    """Construct a model with the selected adaptation strategy."""
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
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
            modules_to_save=["pre_classifier", "classifier"],
        )
        return get_peft_model(model, config)

    raise ValueError(f"Unknown strategy: {strategy}")


def parameter_statistics(model):
    """Count trainable parameters and estimate parameter storage."""
    total_parameters = 0
    trainable_parameters = 0
    total_bytes = 0
    trainable_bytes = 0

    for parameter in model.parameters():
        count = parameter.numel()
        size_bytes = count * parameter.element_size()

        total_parameters += count
        total_bytes += size_bytes

        if parameter.requires_grad:
            trainable_parameters += count
            trainable_bytes += size_bytes

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percent": (
            100 * trainable_parameters / total_parameters
        ),
        "parameter_storage_mib": total_bytes / 1024**2,
        "trainable_storage_mib": trainable_bytes / 1024**2,
    }


def run_experiment(
    strategy,
    train_dataset,
    validation_dataset,
    experiment_index,
    rank=8,
    alpha=16,
):
    """Train and evaluate one configuration."""
    set_seed(SEED)
    model = build_model(strategy, rank=rank, alpha=alpha)
    statistics = parameter_statistics(model)

    run_name = f"{experiment_index:02d}-{strategy}-r{rank}-a{alpha}"

    learning_rate = (
        2e-5 if strategy in {"full_fine_tuning", "frozen_backbone"}
        else 1e-3
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / run_name),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=learning_rate,
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
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )

    start_time = perf_counter()
    trainer.train()
    training_seconds = perf_counter() - start_time
    metrics = trainer.evaluate()

    result = {
        "experiment": run_name,
        "strategy": strategy,
        "rank": rank if strategy == "lora" else None,
        "alpha": alpha if strategy == "lora" else None,
        "learning_rate": learning_rate,
        **statistics,
        "training_seconds": training_seconds,
        "validation_loss": metrics["eval_loss"],
        "accuracy": metrics["eval_accuracy"],
        "f1": metrics["eval_f1"],
    }

    print(f"\n--- {run_name} ---")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return result


def save_results(results):
    """Write the comparison results to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "comparison-results.csv"

    columns = list(results[0].keys())

    import csv

    with result_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {result_path}")


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, validation_dataset = prepare_datasets(tokenizer)

    # Change one variable at a time where possible.
    experiments = [
        {"strategy": "full_fine_tuning"},
        {"strategy": "frozen_backbone"},
        {"strategy": "lora", "rank": 4, "alpha": 8},
        {"strategy": "lora", "rank": 8, "alpha": 16},
        {"strategy": "lora", "rank": 16, "alpha": 32},
        {"strategy": "lora", "rank": 8, "alpha": 8},
    ]

    results = []

    for index, experiment in enumerate(experiments, start=1):
        result = run_experiment(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            experiment_index=index,
            **experiment,
        )
        results.append(result)

    print("\n--- Summary ---")
    print(
        f"{'Experiment':<28} {'Trainable %':>11} "
        f"{'Accuracy':>9} {'F1':>8} {'Seconds':>9}"
    )
    print("-" * 70)

    for result in results:
        print(
            f"{result['experiment']:<28} "
            f"{result['trainable_percent']:>10.2f}% "
            f"{result['accuracy']:>9.3f} "
            f"{result['f1']:>8.3f} "
            f"{result['training_seconds']:>9.2f}"
        )

    save_results(results)


if __name__ == "__main__":
    main()
