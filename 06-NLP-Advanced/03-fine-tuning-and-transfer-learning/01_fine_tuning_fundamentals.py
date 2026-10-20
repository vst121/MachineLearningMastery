"""
Fine-Tuning Fundamentals

Experiments:
- Load a pretrained Transformer and inspect its classification head.
- Establish baseline predictions before fine-tuning.
- Fine-tune the model on a small, reproducible sentiment dataset.
- Compare predictions, evaluation metrics, and parameters before and after training.
"""

from collections import Counter
from random import Random
from time import perf_counter

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
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

# A small illustrative dataset for demonstrating the fine-tuning workflow.
# For meaningful conclusions, use a larger, representative dataset.
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
    """Split the data and tokenize examples for model training."""
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

    def build_dataset(texts, labels):
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = labels
        return Dataset.from_dict(encoded)

    train_dataset = build_dataset(train_texts, train_labels)
    validation_dataset = build_dataset(
        validation_texts, validation_labels
    )

    print(f"Training examples:   {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")
    print(f"Label distribution:  {dict(Counter(train_labels))}")

    return train_dataset, validation_dataset, validation_texts, validation_labels


def inspect_model(model):
    """Inspect model architecture and parameter counts."""
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\n--- Model inspection ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Total parameters:     {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Trainable percentage: {100 * trainable_parameters / total_parameters:.2f}%")
    print("\nClassification head:")
    print(model.classifier)

    return total_parameters, trainable_parameters


def predict_texts(model, tokenizer, texts, batch_size=8):
    """Return predicted labels and confidence scores."""
    model.eval()
    device = next(model.parameters()).device

    predictions = []
    confidences = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
            )
            inputs = {
                name: tensor.to(device)
                for name, tensor in inputs.items()
            }

            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            scores, labels = probabilities.max(dim=-1)

            predictions.extend(labels.cpu().tolist())
            confidences.extend(scores.cpu().tolist())

    return predictions, confidences


def evaluate_predictions(labels, predictions, title):
    """Report basic classification metrics."""
    print(f"\n--- {title} ---")
    print(f"Accuracy: {accuracy_score(labels, predictions):.3f}")
    print(
        classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["negative", "positive"],
            zero_division=0,
        )
    )


def snapshot_parameters(model):
    """Copy parameters to CPU for a before-and-after comparison."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def compare_parameters(before, model):
    """Measure parameter changes after fine-tuning."""
    changed_tensors = 0
    changed_parameters = 0
    total_absolute_change = 0.0
    total_parameters = 0

    for name, parameter in model.named_parameters():
        original = before[name]
        updated = parameter.detach().cpu()

        difference = updated - original
        changed_tensors += int(torch.any(difference != 0).item())
        changed_parameters += int(torch.count_nonzero(difference).item())
        total_absolute_change += difference.abs().sum().item()
        total_parameters += parameter.numel()

    print("\n--- Parameter changes ---")
    print(f"Parameter tensors changed: {changed_tensors}")
    print(f"Individual parameters changed: {changed_parameters:,}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Mean absolute parameter change: {total_absolute_change / total_parameters:.8f}")


def main():
    set_seed(SEED)
    Random(SEED)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    train_dataset, validation_dataset, validation_texts, validation_labels = (
        prepare_datasets(tokenizer)
    )

    inspect_model(model)

    before_training = snapshot_parameters(model)

    baseline_predictions, _ = predict_texts(
        model, tokenizer, validation_texts
    )
    evaluate_predictions(
        validation_labels,
        baseline_predictions,
        "Before fine-tuning",
    )

    training_args = TrainingArguments(
        output_dir="outputs/fine-tuning-fundamentals",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = logits.argmax(axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n--- Fine-tuning ---")
    start_time = perf_counter()
    trainer.train()
    elapsed_seconds = perf_counter() - start_time

    model = trainer.model
    final_predictions, final_confidences = predict_texts(
        model, tokenizer, validation_texts
    )

    evaluate_predictions(
        validation_labels,
        final_predictions,
        "After fine-tuning",
    )
    compare_parameters(before_training, model)

    print(f"\nTraining time: {elapsed_seconds:.2f} seconds")
    print("\n--- Example predictions after fine-tuning ---")

    for text, expected, predicted, confidence in zip(
        validation_texts[:5],
        validation_labels[:5],
        final_predictions[:5],
        final_confidences[:5],
    ):
        print(f"\nText: {text}")
        print(f"Expected:   {expected}")
        print(f"Predicted:  {predicted}")
        print(f"Confidence: {confidence:.3f}")

    output_dir = "outputs/fine-tuning-fundamentals/final-model"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved fine-tuned model to: {output_dir}")


if __name__ == "__main__":
    main()
