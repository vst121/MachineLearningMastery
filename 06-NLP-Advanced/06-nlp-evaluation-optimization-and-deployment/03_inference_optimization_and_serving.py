
"""
NLP Inference Optimization and Serving

Experiments:
- Establish a baseline for NLP inference latency and throughput.
- Compare inference batch sizes and sequence lengths.
- Measure the effect of dynamic padding on token processing.
- Compare sequential and batched inference.
- Track approximate model memory usage when supported.
- Export benchmark results as a JSON report.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_DIR = Path("outputs/inference-optimization")
BATCH_SIZES = [1, 2, 4, 8]
SEQUENCE_LENGTHS = [32, 64, 128]


@dataclass
class BenchmarkResult:
    """Store the results of one inference benchmark."""

    batch_size: int
    sequence_length: int
    samples: int
    elapsed_seconds: float
    latency_ms_per_batch: float
    latency_ms_per_sample: float
    throughput_samples_per_second: float
    padded_tokens_processed: int
    peak_cuda_memory_mb: float | None


def get_device() -> torch.device:
    """Choose an available inference device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    model_name: str,
    device: torch.device,
) -> tuple[Any, Any]:
    """Load a pretrained classifier and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )
    model.to(device)
    model.eval()

    return tokenizer, model


def create_texts(
    count: int,
    sequence_length: int,
) -> list[str]:
    """Create repeatable text inputs of approximate lengths."""
    base_text = (
        "The product is useful and reliable, and the overall "
        "experience was positive. "
    )
    texts = []

    for index in range(count):
        words = (base_text * (index % 4 + 1)).split()
        words = (words * (sequence_length // max(len(words), 1) + 1))
        text = " ".join(words[:sequence_length])
        texts.append(text)

    return texts


def synchronize_device(device: torch.device) -> None:
    """Synchronize asynchronous GPU work before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_batch(
    tokenizer: Any,
    model: Any,
    texts: list[str],
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    warmup_batches: int = 1,
) -> BenchmarkResult:
    """Measure batched inference latency and throughput."""
    if not texts:
        raise ValueError("At least one text is required.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")

    batches = [
        texts[start : start + batch_size]
        for start in range(0, len(texts), batch_size)
    ]

    def run_batch(batch_texts: list[str]) -> int:
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=sequence_length,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            outputs = model(**encoded)

        return int(encoded["input_ids"].numel())

    # Warm up the model before collecting measurements.
    for batch in batches[:warmup_batches]:
        run_batch(batch)

    synchronize_device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    padded_tokens_processed = 0
    synchronize_device(device)
    start_time = time.perf_counter()

    for batch in batches:
        padded_tokens_processed += run_batch(batch)

    synchronize_device(device)
    elapsed = time.perf_counter() - start_time

    sample_count = len(texts)
    batch_count = len(batches)
    elapsed = max(elapsed, 1e-12)

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )

    return BenchmarkResult(
        batch_size=batch_size,
        sequence_length=sequence_length,
        samples=sample_count,
        elapsed_seconds=elapsed,
        latency_ms_per_batch=elapsed / batch_count * 1000,
        latency_ms_per_sample=elapsed / sample_count * 1000,
        throughput_samples_per_second=sample_count / elapsed,
        padded_tokens_processed=padded_tokens_processed,
        peak_cuda_memory_mb=peak_memory_mb,
    )


def benchmark_dynamic_padding(
    tokenizer: Any,
    model: Any,
    texts: list[str],
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Compare dynamic padding with fixed-length padding."""
    results: dict[str, float | int] = {}

    for padding_mode in ("dynamic", "fixed"):
        def run_once() -> int:
            total_tokens = 0

            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]

                encoded = tokenizer(
                    batch,
                    padding=True if padding_mode == "dynamic" else "max_length",
                    truncation=True,
                    max_length=sequence_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(device)
                    for key, value in encoded.items()
                }

                with torch.inference_mode():
                    model(**encoded)

                total_tokens += int(encoded["input_ids"].numel())

            return total_tokens

        # Warm-up, then time the complete run.
        run_once()
        synchronize_device(device)
        start_time = time.perf_counter()
        token_count = run_once()
        synchronize_device(device)

        elapsed = max(time.perf_counter() - start_time, 1e-12)
        results[f"{padding_mode}_seconds"] = elapsed
        results[f"{padding_mode}_padded_tokens"] = token_count

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Display benchmark results in a readable table."""
    header = (
        f"{'Batch':>5} {'Length':>6} {'Latency/sample':>15} "
        f"{'Samples/sec':>12} {'Padded tokens':>14} {'CUDA MB':>10}"
    )
    print("\nInference Benchmark Results")
    print("---------------------------")
    print(header)
    print("-" * len(header))

    for result in results:
        memory = (
            f"{result.peak_cuda_memory_mb:.1f}"
            if result.peak_cuda_memory_mb is not None
            else "N/A"
        )
        print(
            f"{result.batch_size:>5} "
            f"{result.sequence_length:>6} "
            f"{result.latency_ms_per_sample:>12.2f} ms "
            f"{result.throughput_samples_per_second:>12.2f} "
            f"{result.padded_tokens_processed:>14} "
            f"{memory:>10}"
        )


def save_report(
    results: list[BenchmarkResult],
    padding_results: dict[str, float | int],
    device: torch.device,
) -> Path:
    """Save benchmark measurements and environment metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "inference_benchmark.json"

    report = {
        "model": MODEL_NAME,
        "device": str(device),
        "python_platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "benchmarks": [asdict(result) for result in results],
        "dynamic_vs_fixed_padding": padding_results,
        "notes": [
            "Measurements depend on hardware, software, and model cache state.",
            "Warm-up and timing are performed separately.",
            "Latency per sample is elapsed time divided by sample count.",
            "CUDA peak memory is allocated memory observed during each benchmark.",
            "Dynamic padding may reduce token computation for mixed-length batches.",
        ],
    }

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    """Run inference performance experiments."""
    device = get_device()

    print("NLP Inference Optimization and Serving")
    print("======================================")
    print(f"Model:  {MODEL_NAME}")
    print(f"Device: {device}")

    tokenizer, model = load_model(MODEL_NAME, device)

    # Keep the dataset modest so the experiment is practical on a CPU.
    texts = create_texts(count=32, sequence_length=128)
    results: list[BenchmarkResult] = []

    for sequence_length in SEQUENCE_LENGTHS:
        for batch_size in BATCH_SIZES:
            result = benchmark_batch(
                tokenizer=tokenizer,
                model=model,
                texts=texts,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
            )
            results.append(result)

    print_results(results)

    padding_results = benchmark_dynamic_padding(
        tokenizer=tokenizer,
        model=model,
        texts=texts,
        batch_size=8,
        sequence_length=128,
        device=device,
    )

    print("\nDynamic vs. Fixed-Length Padding")
    print("--------------------------------")
    print(
        "Dynamic padding time: "
        f"{padding_results['dynamic_seconds']:.3f} seconds"
    )
    print(
        "Fixed padding time:   "
        f"{padding_results['fixed_seconds']:.3f} seconds"
    )
    print(
        "Dynamic padded tokens: "
        f"{padding_results['dynamic_padded_tokens']}"
    )
    print(
        "Fixed padded tokens:   "
        f"{padding_results['fixed_padded_tokens']}"
    )

    report_path = save_report(results, padding_results, device)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
