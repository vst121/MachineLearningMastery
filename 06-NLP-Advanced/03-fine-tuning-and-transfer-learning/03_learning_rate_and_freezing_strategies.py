"""
Learning Rates and Freezing Strategies

Experiments:
- Compare full fine-tuning with a frozen Transformer backbone.
- Fine-tune only the final Transformer layers.
- Compare learning rates under consistent data splits.
- Measure trainable parameters, validation metrics, and training time.
"""

from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
from datasets import Dataset
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
OUTPUT_DIR = "outputs/learning-rate-freezing"

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
    """Create fixed, stratified training and validation datasets."""
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
    """Calculate accuracy and F1 from model logits."""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def configure_trainable_parameters(model, strategy):
    """Apply a parameter-freezing strategy to DistilBERT."""
    # Freeze the complete pretrained Transformer backbone first.
    for parameter in model.distilbert.parameters():
        parameter.requires_grad = False

    if strategy == "full_fine_tuning":
        for parameter in model.distilbert.parameters():
            parameter.requires_grad = True

    elif strategy == "frozen_backbone":
        # Only the task-specific classification head remains trainable.
        pass

    elif strategy == "last_two_layers":
        # Keep the lower Transformer layers frozen.
        for layer in model.distilbert.transformer.layer[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Always train the classification head.
    for parameter in model.pre_classifier.parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def run_experiment(
    strategy,
    learning_rate,
    train_dataset,
    validation_dataset,
    experiment_number,
):
    """Train and evaluate one model configuration."""
    set_seed(SEED)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    total_parameters, trainable_parameters = configure_trainable_parameters(
        model, strategy
    )

    run_name = f"{experiment_number:02d}-{strategy}-lr-{learning_rate:g}"

    args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{run_name}",
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
        "strategy": strategy,
        "learning_rate": learning_rate,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_percent": 100 * trainable_parameters / total_parameters,
        "training_seconds": training_seconds,
        "eval_loss": metrics["eval_loss"],
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


def main():
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, validation_dataset = prepare_datasets(tokenizer)

    experiments = [
        ("full_fine_tuning", 2e-5),
        ("full_fine_tuning", 5e-5),
        ("frozen_backbone", 2e-5),
        ("last_two_layers", 2e-5),
    ]

    results = []

    for index, (strategy, learning_rate) in enumerate(experiments, start=1):
        result = run_experiment(
            strategy=strategy,
            learning_rate=learning_rate,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            experiment_number=index,
        )
        results.append(result)

    print("\n--- Experiment comparison ---")
    header = (
        f"{'Strategy':<22} {'LR':>9} {'Trainable %':>12} "
        f"{'Accuracy':>9} {'F1':>9} {'Val loss':>10} {'Seconds':>9}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result['strategy']:<22} "
            f"{result['learning_rate']:>9.1e} "
            f"{result['trainable_percent']:>11.2f}% "
            f"{result['accuracy']:>9.3f} "
            f"{result['f1']:>9.3f} "
            f"{result['eval_loss']:>10.4f} "
            f"{result['training_seconds']:>9.2f}"
        )


if __name__ == "__main__":
    main()
