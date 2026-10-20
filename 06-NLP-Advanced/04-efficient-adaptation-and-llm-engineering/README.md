# Efficient Adaptation and LLM Engineering

## Learning Objectives

- Understand parameter-efficient fine-tuning (PEFT) and its advantages over full fine-tuning.
- Apply Low-Rank Adaptation (LoRA) and analyze the effects of rank and scaling.
- Understand 8-bit and 4-bit quantization, NF4, double quantization, and QLoRA.
- Measure model memory consumption, inference latency, throughput, and batch-size effects.
- Evaluate practical trade-offs between model quality, computational cost, and deployment performance.

## Core Concepts

### 1. Parameter-Efficient Fine-Tuning

Full fine-tuning updates all model parameters, while PEFT methods freeze most of the pretrained model and update a small subset of additional parameters.

LoRA approximates a weight update using two low-rank matrices:

$$
W' = W + \Delta W
$$

$$
\Delta W = BA
$$

For a weight matrix with dimensions $$d \times k$$ and LoRA rank $$r$$, the number of trainable parameters becomes:

$$
N_{\text{LoRA}} = r(d+k)
$$

When $$r \ll \min(d,k)$$, this can substantially reduce the number of trainable parameters.

### 2. LoRA Rank and Scaling

The rank controls the capacity of the low-rank update. A larger rank increases trainable parameters and may improve adaptation capacity, but also increases computational and memory requirements.

A common LoRA scaling formulation is:

$$
\Delta W = \frac{\alpha}{r}BA
$$

Here, $$\alpha$$ is the scaling factor. Rank and scaling should be evaluated experimentally rather than assuming that larger values always improve results.

### 3. Quantization and QLoRA

Quantization reduces the number of bits used to represent model weights.

Approximate raw weight storage is:

$$
M_{\text{weights}} \approx N_{\text{parameters}} \times \frac{b}{8}
$$

Here, $$b$$ is the number of bits per parameter. Four-bit quantization can substantially reduce base-model storage compared with FP32 or FP16, although actual memory use also includes metadata, activations, and runtime overhead.

QLoRA combines a frozen quantized base model with trainable LoRA adapters:

$$
W' \approx D(Q(W)) + BA
$$

Here, $$Q(W)$$ represents quantized weights and $$D$$ represents their dequantized values during computation.

Normal Float 4 (NF4) is designed for normally distributed pretrained weights. Double quantization further reduces the overhead associated with quantization constants.

### 4. LLM Inference Performance

Inference performance depends on model size, input length, output length, hardware, batch size, and generation settings.

Latency measures the time required to process a request:

$$
T_{\text{latency}} = T_{\text{completion}} - T_{\text{start}}
$$

Token throughput measures the rate of generated tokens:

$$
\text{Throughput} =
\frac{N_{\text{generated tokens}}}{T_{\text{generation}}}
$$

Batching can improve throughput by processing multiple requests together, but it can increase memory consumption and affect individual-request latency. KV caching reduces repeated computation during autoregressive generation at the cost of additional memory.

## Implementation and Experiments

| File                                                 | Experiment                                                                                                                                             |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `01_parameter_efficient_fine_tuning_fundamentals.py` | Compare full fine-tuning, a frozen backbone, and LoRA using trainable parameter counts, storage estimates, validation metrics, and training time.      |
| `02_lora_and_adapter_comparison.py`                  | Investigate LoRA ranks and scaling factors, comparing adaptation capacity, trainable parameters, model performance, and training duration.             |
| `03_quantization_and_qlora_fundamentals.py`          | Estimate FP32, FP16/BF16, INT8, and INT4 weight storage; inspect quantization configurations; explore 4-bit inference and QLoRA adapter preparation.   |
| `04_llm_memory_performance_and_deployment.py`        | Measure model loading time, inference latency, throughput, memory usage, batch-size effects, and input-length effects. Save benchmark results as JSON. |

The experiments use Hugging Face Transformers and PEFT. Quantized loading and QLoRA preparation require compatible CUDA hardware and a suitable `bitsandbytes` installation. The quantization experiment demonstrates QLoRA preparation and adapter saving, not a complete QLoRA training run.

## Expected Outcome

After completing this section, you should be able to:

- Explain when PEFT is preferable to full fine-tuning.
- Analyze LoRA rank, scaling, and trainable parameter counts.
- Estimate the memory benefits and limitations of quantization.
- Explain how QLoRA combines quantization with parameter-efficient adaptation.
- Benchmark basic inference workloads and interpret latency, throughput, and memory measurements.
- Make informed engineering decisions about fine-tuning strategies, model precision, hardware requirements, and deployment trade-offs.
