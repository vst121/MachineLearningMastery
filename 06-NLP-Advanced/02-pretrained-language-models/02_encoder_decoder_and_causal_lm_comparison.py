"""
Encoder-Decoder and Causal Language Model Comparison

Experiments:
- Compare encoder-only, encoder-decoder, and decoder-only architectures.
- Inspect architecture configurations and parameter counts.
- Compare masked, bidirectional, and causal attention behavior.
- Observe encoder representations, sequence-to-sequence outputs,
  and autoregressive text generation.
- Relate architecture choices to common NLP tasks.
"""

from time import perf_counter

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


MODELS = {
    "encoder_only": "google-bert/bert-base-uncased",
    "encoder_decoder": "google-t5/t5-small",
    "decoder_only": "openai-community/gpt2",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def inspect_configuration(name: str, model) -> None:
    """Print model architecture metadata."""
    config = model.config
    total, trainable = count_parameters(model)

    print(f"\n--- {name} ---")
    print("Model type:", config.model_type)
    print("Architectures:", getattr(config, "architectures", None))
    print("Hidden size:", getattr(config, "hidden_size", None))
    print("Encoder layers:", getattr(config, "num_hidden_layers", None))
    print("Decoder layers:", getattr(config, "num_decoder_layers", None))
    print("Attention heads:", getattr(config, "num_attention_heads", None))
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")


def run_encoder_only() -> None:
    """Inspect bidirectional encoder representations."""
    model_name = MODELS["encoder_only"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    text = "The model learns contextual representations of language."
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model(**inputs)

    print("\n--- Encoder-Only: BERT ---")
    print("Input tokens:", tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0].tolist()
    ))
    print("Input shape:", tuple(inputs["input_ids"].shape))
    print("Last hidden state:", tuple(outputs.last_hidden_state.shape))
    print("Typical tasks: classification, token labeling, retrieval.")

    inspect_configuration("BERT", model)


def run_encoder_decoder() -> None:
    """Run a text-to-text generation example."""
    model_name = MODELS["encoder_decoder"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    text = "summarize: Pretrained language models learn patterns from large text corpora."
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    with torch.inference_mode():
        encoder_outputs = model.get_encoder()(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    print("\n--- Encoder-Decoder: T5 ---")
    print("Input shape:", tuple(inputs["input_ids"].shape))
    print("Encoder hidden states:",
          tuple(encoder_outputs.last_hidden_state.shape))
    print("Generated IDs shape:", tuple(generated_ids.shape))
    print("Generated text:", tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ))
    print("Typical tasks: summarization, translation, text transformation.")

    inspect_configuration("T5", model)


def run_decoder_only() -> None:
    """Inspect causal language modeling and autoregressive generation."""
    model_name = MODELS["decoder_only"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    # GPT-2 has no dedicated padding token by default.
    tokenizer.pad_token = tokenizer.eos_token

    text = "Pretrained language models can"
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model(**inputs)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("\n--- Decoder-Only: GPT-2 ---")
    print("Input shape:", tuple(inputs["input_ids"].shape))
    print("Logits shape:", tuple(outputs.logits.shape))
    print("Generated IDs shape:", tuple(generated_ids.shape))
    print("Generated text:", tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ))
    print("Typical tasks: text generation and next-token prediction.")

    inspect_configuration("GPT-2", model)


def compare_loading_and_inference_time() -> None:
    """Measure basic loading and forward-pass times."""
    print("\n--- Timing Comparison ---")
    print("Device:", DEVICE)

    examples = [
        (
            "Encoder-only",
            MODELS["encoder_only"],
            AutoModel,
            "Language models process text.",
        ),
        (
            "Encoder-decoder",
            MODELS["encoder_decoder"],
            AutoModelForSeq2SeqLM,
            "translate English to German: Good morning.",
        ),
        (
            "Decoder-only",
            MODELS["decoder_only"],
            AutoModelForCausalLM,
            "Language models process",
        ),
    ]

    for label, model_name, model_class, text in examples:
        start = perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name).to(DEVICE)
        model.eval()
        load_time = perf_counter() - start

        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = perf_counter()
        with torch.inference_mode():
            model(**inputs)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        inference_time = perf_counter() - start

        print(
            f"{label:18} | load={load_time:.2f}s "
            f"| forward={inference_time * 1000:.2f}ms"
        )


def main() -> None:
    print("Comparing pretrained Transformer architectures.")
    print("Device:", DEVICE)

    run_encoder_only()
    run_encoder_decoder()
    run_decoder_only()
    compare_loading_and_inference_time()


if __name__ == "__main__":
    main()
