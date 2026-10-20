"""
LLM Memory, Performance, and Deployment

Experiments:
- Measure model loading time and inference latency.
- Compare latency across input lengths and batch sizes.
- Measure throughput and GPU memory usage when CUDA is available.
- Inspect generation settings that affect inference performance.
- Save a reproducible performance report for deployment analysis.
"""

import json
import platform
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "facebook/opt-125m"
OUTPUT_DIR = Path("outputs/llm-performance")
MAX_NEW_TOKENS = 32
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5

PROMPTS = [
    "Explain why software architecture matters.",
    "Describe how language models generate text.",
    "Explain the difference between training and inference.",
    "Describe the benefits of efficient model deployment.",
]


def synchronize_device():
    """Ensure asynchronous CUDA work finishes before timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_device():
    """Choose CUDA when available, otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_memory_snapshot():
    """Return CUDA memory measurements when available."""
    if not torch.cuda.is_available():
        return {
            "allocated_mib": None,
            "reserved_mib": None,
            "peak_allocated_mib": None,
        }

    mib = 1024**2
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / mib, 2),
        "reserved_mib": round(torch.cuda.memory_reserved() / mib, 2),
        "peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / mib, 2
        ),
    }


def load_model():
    """Load the tokenizer and model, and measure loading duration."""
    device = get_device()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    synchronize_device()
    load_seconds = time.perf_counter() - start

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("--- Model loading ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Loading time: {load_seconds:.3f} seconds")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Memory after loading: {get_memory_snapshot()}")

    return tokenizer, model, device, load_seconds


def generate_batch(tokenizer, model, device, prompts):
    """Generate continuations for a batch of prompts."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    return inputs, outputs


def benchmark_batch(tokenizer, model, device, prompts):
    """Measure end-to-end generation latency and throughput."""
    for _ in range(WARMUP_RUNS):
        generate_batch(tokenizer, model, device, prompts)

    synchronize_device()

    durations = []
    output_token_counts = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()

        inputs, outputs = generate_batch(tokenizer, model, device, prompts)

        synchronize_device()
        durations.append(time.perf_counter() - start)

        # Count generated tokens, excluding padded input tokens.
        generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        output_token_counts.append(max(generated_tokens, 0) * len(prompts))

    mean_latency = statistics.mean(durations)
    total_generated_tokens = statistics.mean(output_token_counts)

    return {
        "batch_size": len(prompts),
        "mean_latency_seconds": round(mean_latency, 4),
        "median_latency_seconds": round(statistics.median(durations), 4),
        "mean_generated_tokens": round(total_generated_tokens, 2),
        "throughput_tokens_per_second": round(
            total_generated_tokens / mean_latency, 2
        )
        if mean_latency > 0
        else 0,
        "memory": get_memory_snapshot(),
    }


def benchmark_batch_sizes(tokenizer, model, device):
    """Compare single-request and batched inference."""
    print("\n--- Batch-size comparison ---")
    results = []

    for batch_size in (1, 2, 4):
        prompts = PROMPTS[:batch_size]

        result = benchmark_batch(tokenizer, model, device, prompts)
        results.append(result)

        print(
            f"Batch size: {result['batch_size']} | "
            f"Latency: {result['mean_latency_seconds']:.4f}s | "
            f"Throughput: "
            f"{result['throughput_tokens_per_second']:.2f} tokens/s"
        )

    return results


def benchmark_input_lengths(tokenizer, model, device):
    """Measure how prompt length affects generation latency."""
    print("\n--- Input-length comparison ---")
    results = []

    prompt_variants = {
        "short": "Explain AI.",
        "medium": (
            "Explain how artificial intelligence systems use language "
            "models to process input and generate useful responses."
        ),
        "long": (
            "Explain how production language model systems process user "
            "requests, tokenize text, construct model inputs, run transformer "
            "layers, generate output tokens, and manage memory and latency. "
        )
        * 5,
    }

    for name, prompt in prompt_variants.items():
        encoded = tokenizer(prompt, return_tensors="pt")
        input_tokens = encoded["input_ids"].shape[1]

        result = benchmark_batch(tokenizer, model, device, [prompt])
        result["prompt_category"] = name
        result["input_tokens"] = input_tokens
        results.append(result)

        print(
            f"{name:8} | Input tokens: {input_tokens:4} | "
            f"Latency: {result['mean_latency_seconds']:.4f}s"
        )

    return results


def inspect_generation_configuration(model):
    """Report generation settings relevant to inference."""
    config = model.generation_config

    settings = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": getattr(config, "use_cache", None),
        "model_max_length": getattr(model.config, "max_position_embeddings", None),
    }

    print("\n--- Inference configuration ---")
    print(json.dumps(settings, indent=2))

    print(
        "\nKV caching can avoid recomputing prior attention keys and values "
        "during autoregressive generation, at the cost of additional memory."
    )

    return settings


def save_report(report):
    """Persist benchmark results as JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "performance_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nSaved report to: {report_path}")


def main():
    tokenizer, model, device, load_seconds = load_model()

    batch_results = benchmark_batch_sizes(tokenizer, model, device)
    length_results = benchmark_input_lengths(tokenizer, model, device)
    generation_settings = inspect_generation_configuration(model)

    report = {
        "model_name": MODEL_NAME,
        "device": str(device),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model_load_seconds": round(load_seconds, 4),
        "generation_settings": generation_settings,
        "batch_size_experiments": batch_results,
        "input_length_experiments": length_results,
    }

    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)

    save_report(report)


if __name__ == "__main__":
    main()
