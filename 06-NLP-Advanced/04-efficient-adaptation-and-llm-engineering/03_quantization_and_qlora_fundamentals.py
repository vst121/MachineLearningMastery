"""
Quantization and QLoRA Fundamentals

Experiments:
- Compare floating-point and quantized model storage estimates.
- Inspect 8-bit and 4-bit quantization configurations.
- Load a 4-bit pretrained model when compatible CUDA hardware is available.
- Prepare a quantized model for training and apply LoRA adapters.
- Compare trainable parameters and inspect the quantized model structure.
"""

from pathlib import Path

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_NAME = "facebook/opt-125m"
OUTPUT_DIR = Path("outputs/quantization-qlora")
PROMPT = "Efficient language model adaptation is useful because"


def print_hardware_information():
    """Report hardware and quantization prerequisites."""
    print("--- Hardware information ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB"
        )

    try:
        import bitsandbytes

        print(f"bitsandbytes version: {bitsandbytes.__version__}")
    except ImportError:
        print("bitsandbytes is not installed.")


def estimate_weight_storage(parameter_count):
    """Estimate raw weight storage at common numerical precisions."""
    bytes_per_parameter = {
        "FP32": 4,
        "FP16/BF16": 2,
        "INT8": 1,
        "INT4": 0.5,
    }

    print("\n--- Theoretical raw weight storage ---")
    print(f"Parameter count: {parameter_count:,}")

    for precision, size in bytes_per_parameter.items():
        memory_mib = parameter_count * size / 1024**2
        print(f"{precision:10}: {memory_mib:10.2f} MiB")

    print(
        "\nThese are idealized estimates. Real quantized storage also "
        "includes scales, metadata, padding, and implementation overhead."
    )


def inspect_quantization_configurations():
    """Display representative 8-bit and 4-bit configurations."""
    print("\n--- Quantization configurations ---")

    int8_config = BitsAndBytesConfig(load_in_8bit=True)

    int4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("8-bit configuration:")
    print(int8_config)

    print("\n4-bit configuration:")
    print(int4_config)

    print("\nNF4 uses a quantization representation suited to normally")
    print("distributed pretrained weights.")
    print("Double quantization quantizes the quantization constants.")


def inspect_model_parameters(model):
    """Report parameter counts and the number of trainable parameters."""
    total = 0
    trainable = 0

    for parameter in model.parameters():
        total += parameter.numel()

        if parameter.requires_grad:
            trainable += parameter.numel()

    print("\n--- Model parameter statistics ---")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable percentage: {100 * trainable / total:.4f}%")

    return total, trainable


def run_quantized_inference():
    """Load a quantized model and generate a short continuation."""
    if not torch.cuda.is_available():
        print(
            "\nSkipping quantized model loading: this experiment expects "
            "compatible CUDA hardware for bitsandbytes quantization."
        )
        return

    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        print("\nSkipping quantized loading: install bitsandbytes first.")
        return

    print("\n--- Loading a 4-bit model ---")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
        )
    except (ImportError, RuntimeError, ValueError, OSError) as error:
        print(f"Could not load the quantized model: {error}")
        print(
            "Check CUDA compatibility, bitsandbytes installation, "
            "available memory, and model access."
        )
        return

    print(f"Model: {MODEL_NAME}")
    print(f"Quantization: {model.config}")
    print(f"Model device: {model.get_input_embeddings().weight.device}")

    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("\nPrompt:")
    print(PROMPT)
    print("\nGenerated continuation:")
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


def run_qlora_preparation():
    """Prepare a 4-bit causal LM for LoRA fine-tuning."""
    if not torch.cuda.is_available():
        print(
            "\nSkipping QLoRA preparation: compatible CUDA hardware "
            "is required for this bitsandbytes configuration."
        )
        return

    print("\n--- Preparing a QLoRA model ---")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
        )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

        model = get_peft_model(model, lora_config)

    except (ImportError, RuntimeError, ValueError, OSError) as error:
        print(f"Could not prepare the QLoRA model: {error}")
        print("Check your GPU, package versions, and available memory.")
        return

    inspect_model_parameters(model)
    print("\nTrainable parameter breakdown:")
    model.print_trainable_parameters()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adapter_path = OUTPUT_DIR / "lora-adapter"
    model.save_pretrained(adapter_path)

    print(f"\nSaved adapter configuration and weights to: {adapter_path}")
    print(
        "This saves the adapter, not a fully fine-tuned copy of the "
        "entire base model."
    )


def main():
    print_hardware_information()

    # OPT-125M has approximately 125 million parameters. This is an
    # approximate reference value for storage calculations.
    estimate_weight_storage(parameter_count=125_000_000)

    inspect_quantization_configurations()
    run_quantized_inference()
    run_qlora_preparation()


if __name__ == "__main__":
    main()
