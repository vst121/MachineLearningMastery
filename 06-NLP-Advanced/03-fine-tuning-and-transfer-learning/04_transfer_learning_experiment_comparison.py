"""
Transfer Learning Experiment Comparison

Experiments:
- Compare a majority-class baseline, a frozen pretrained backbone,
  partial fine-tuning, and full fine-tuning.
- Use identical training and validation splits across experiments.
- Measure accuracy, precision, recall, F1, validation loss, and training time.
- Compare trainable parameter counts and save results for later analysis.
"""

import csv
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from datasets import Dataset
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
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
OUTPUT_DIR = Path("outputs/transfer-learning-comparison")

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


def prepare_data(tokenizer):
    """Create one fixed stratified split shared by all experiments."""
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
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = labels
        return Dataset.from_dict(encoded)

    return (
        encode(train_texts, train_labels),
        encode(validation_texts, validation_labels),
        train_labels,
        validation_labels,
    )


def compute_metrics(eval_prediction):
    """Calculate classification metrics from model logits."""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def configure_strategy(model, strategy):
    """Configure trainable parameters for each transfer-learning strategy."""
    if strategy == "full_fine_tuning":
        pass

    elif strategy == "frozen_backbone":
        for parameter in model.distilbert.parameters():
            parameter.requires_grad = False

    elif strategy == "last_two_layers":
        for parameter in model.distilbert.parameters():
            parameter.requires_grad = False

        for layer in model.distilbert.transformer.layer[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters


def run_model_experiment(
    strategy,
    learning_rate,
    train_dataset,
    validation_dataset,
    experiment_index,
):
    """Train and evaluate one pretrained-model configuration."""
    set_seed(SEED)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    total_parameters, trainable_parameters = configure_strategy(
        model, strategy
    )

    run_name = f"{experiment_index:02d}-{strategy}"

    training_args = TrainingArguments(
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
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )

    start_time = perf_counter()
    trainer.train()
    training_seconds = perf_counter() - start_time

    metrics = trainer.evaluate()

    return {
        "experiment": run_name,
        "strategy": strategy,
        "learning_rate": learning_rate,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percent": (
            100.0 * trainable_parameters / total_parameters
        ),
        "training_seconds": training_seconds,
        "validation_loss": metrics["eval_loss"],
        "accuracy": metrics["eval_accuracy"],
        "precision": metrics["eval_precision"],
        "recall": metrics["eval_recall"],
        "f1": metrics["eval_f1"],
    }


def evaluate_majority_baseline(train_labels, validation_labels):
    """Measure a simple baseline without using a pretrained model."""
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(
        np.zeros((len(train_labels), 1)),
        train_labels,
    )
    predictions = baseline.predict(
        np.zeros((len(validation_labels), 1))
    )

    return {
        "experiment": "majority_class_baseline",
        "strategy": "baseline",
        "learning_rate": None,
        "total_parameters": 0,
        "trainable_parameters": 0,
        "trainable_percent": 0.0,
        "training_seconds": 0.0,
        "validation_loss": None,
        "accuracy": accuracy_score(validation_labels, predictions),
        "precision": precision_score(
            validation_labels, predictions, zero_division=0
        ),
        "recall": recall_score(
            validation_labels, predictions, zero_division=0
        ),
        "f1": f1_score(
            validation_labels, predictions, zero_division=0
        ),
    }


def save_results(results):
    """Persist experiment metrics as CSV and JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "experiment-results.csv"
    json_path = OUTPUT_DIR / "experiment-results.json"

    fieldnames = list(results[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nCSV results:  {csv_path}")
    print(f"JSON results: {json_path}")


def print_comparison(results):
    """Print the key metrics in a compact comparison table."""
    print("\n--- Transfer learning comparison ---")
    print(
        f"{'Experiment':<27} {'Trainable %':>11} "
        f"{'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
        f"{'F1':>8} {'Val loss':>10} {'Seconds':>9}"
    )
    print("-" * 104)

    for result in results:
        loss = result["validation_loss"]
        loss_text = f"{loss:.4f}" if loss is not None else "N/A"

        print(
            f"{result['experiment']:<27} "
            f"{result['trainable_percent']:>10.2f}% "
            f"{result['accuracy']:>9.3f} "
            f"{result['precision']:>10.3f} "
            f"{result['recall']:>8.3f} "
            f"{result['f1']:>8.3f} "
            f"{loss_text:>10} "
            f"{result['training_seconds']:>9.2f}"
        )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    set_seed(SEED)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    (
        train_dataset,
        validation_dataset,
        train_labels,
        validation_labels,
    ) = prepare_data(tokenizer)

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")

    results = [
        evaluate_majority_baseline(train_labels, validation_labels)
    ]

    experiments = [
        ("frozen_backbone", 2e-5),
        ("last_two_layers", 2e-5),
        ("full_fine_tuning", 2e-5),
    ]

    for index, (strategy, learning_rate) in enumerate(
        experiments, start=1
    ):
        result = run_model_experiment(
            strategy=strategy,
            learning_rate=learning_rate,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            experiment_index=index,
        )
        results.append(result)

    print_comparison(results)
    save_results(results)

    best_model_result = max(
        results[1:],
        key=lambda result: result["f1"],
    )
    print(
        "\nHighest validation F1 among pretrained-model strategies: "
        f"{best_model_result['strategy']} "
        f"({best_model_result['f1']:.3f})"
    )


if __name__ == "__main__":
    main()
