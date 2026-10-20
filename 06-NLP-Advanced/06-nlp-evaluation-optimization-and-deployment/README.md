# NLP Evaluation, Optimization, and Deployment

## Learning Objectives

- Evaluate NLP classifiers using complementary metrics and class-level analysis.
- Identify prediction errors and assess robustness to text perturbations.
- Benchmark inference latency, throughput, batch sizes, sequence lengths, and memory usage.
- Understand dynamic padding and its effect on inference efficiency.
- Build a lightweight NLP inference API with input validation and monitoring.
- Apply production-oriented practices for logging, health checks, and prediction tracking.

## Core Concepts

### 1. Advanced Evaluation Metrics

Accuracy measures overall correctness, while precision and recall capture different aspects of classification performance.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Macro F1 gives equal weight to each class, while weighted F1 accounts for class support.

$$
F_{1,\text{macro}} = \frac{1}{K}\sum_{k=1}^{K} F_{1,k}
$$

$$
F_{1,\text{weighted}} = \sum_{k=1}^{K} \frac{n_k}{N} F_{1,k}
$$

For single-label multiclass classification, micro F1 equals accuracy when all classes are included.

### 2. Error Analysis and Robustness

Robustness testing examines how model predictions respond to variations in input text, including typos, casing, punctuation, and shortened text.

Prediction consistency measures the proportion of predictions unchanged by a perturbation.

$$
\text{Consistency} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = \hat{y}'_i]
$$

A consistent prediction is not necessarily correct. Perturbations that change meaning, such as adding negation, require separate interpretation.

### 3. Inference Optimization

Inference performance can be assessed using latency and throughput.

$$
\text{Latency per sample} = \frac{\text{Total inference time}}{\text{Number of samples}}
$$

$$
\text{Throughput} = \frac{\text{Number of samples}}{\text{Total inference time}}
$$

Batching can improve throughput by processing multiple inputs together. Dynamic padding reduces unnecessary computation by padding each batch to its longest sequence rather than always using a fixed maximum length.

### 4. Deployment and Monitoring

A deployed NLP service should validate inputs, manage model lifecycle, handle failures, and expose operational metrics.

Important monitoring signals include:

- Request volume and error counts
- Average inference latency
- Prediction-label distributions
- Model readiness and service health

Softmax scores are not necessarily calibrated probabilities. Monitoring prediction distributions can reveal changes in incoming data, but does not independently establish model quality or data drift.

## Implementation and Experiments

| File                                          | Experiment                                                                                                                                                                                        |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_advanced_nlp_evaluation_fundamentals.py`  | Calculate accuracy, precision, recall, macro F1, weighted F1, and micro F1. Inspect confusion matrices and select a classification threshold using validation data.                               |
| `02_error_analysis_and_robustness_testing.py` | Categorize misclassifications and measure prediction consistency and accuracy under text perturbations using a TF-IDF and logistic regression baseline.                                           |
| `03_inference_optimization_and_serving.py`    | Benchmark inference across batch sizes and sequence lengths. Compare dynamic and fixed-length padding, measure throughput and latency, inspect available CUDA memory, and export results to JSON. |
| `04_nlp_deployment_and_monitoring.py`         | Implement a FastAPI classification service with input validation, startup model loading, health checks, prediction endpoints, structured logging, and process-local metrics.                      |

## Expected Outcome

After completing this folder, you should be able to evaluate NLP models beyond a single score, investigate common failure patterns, measure inference performance, and implement a basic NLP inference service with operational monitoring.

The experiments establish a foundation for production-oriented NLP engineering. The datasets and service are educational baselines; reliable production deployment requires representative evaluation data, calibrated decision thresholds where appropriate, load testing, security controls, privacy-aware logging, and durable monitoring.
