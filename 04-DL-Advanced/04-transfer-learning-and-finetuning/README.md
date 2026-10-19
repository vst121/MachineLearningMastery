# Transfer Learning and Fine-Tuning

Learn how to adapt pretrained deep learning models to new tasks efficiently. This module focuses on reusing learned representations, selectively fine-tuning model parameters, and evaluating training strategies with PyTorch.

## 🎯 Learning Objectives

- Understand transfer learning and the value of pretrained models.
- Distinguish feature extraction from fine-tuning.
- Load pretrained models and replace task-specific output layers.
- Freeze and selectively unfreeze model parameters.
- Apply discriminative learning rates and learning-rate scheduling.
- Build an end-to-end transfer learning pipeline.
- Evaluate generalization, overfitting, and computational trade-offs.

## 🧠 Core Concepts

### 1. Transfer Learning Fundamentals

Transfer learning reuses knowledge learned from one task to improve performance on another.

- Pretrained models and learned representations
- Source and target tasks
- Dataset size and domain similarity
- Benefits, limitations, and negative transfer
- Feature extraction versus fine-tuning

### 2. Pretrained Models and Feature Extraction

A pretrained model can serve as a fixed feature extractor when its learned parameters are frozen.

- Loading pretrained models and weights
- Freezing parameters with `requires_grad`
- Replacing the task-specific classification head
- Training a new classification head
- Understanding trainable and frozen parameters

### 3. Selective Fine-Tuning

Fine-tuning adapts pretrained parameters to the target dataset while balancing learning speed and preservation of existing knowledge.

- Full versus partial fine-tuning
- Freezing and unfreezing selected layers
- Gradual unfreezing
- Discriminative learning rates
- Learning-rate schedulers
- Catastrophic forgetting and overfitting

### 4. Practical Pipeline and Evaluation

Build and evaluate an end-to-end transfer learning workflow using a pretrained image classification model.

- Dataset preparation and transformations
- Training and validation loops
- Reproducibility and device selection
- Checkpointing and model selection
- Accuracy and loss curves
- Feature extraction versus fine-tuning
- Parameter counts, training time, and generalization

## 🛠️ Implementation and Experiments

| File                                   | Experiment                                                                                                                                           |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_transfer_learning_fundamentals.py` | Explore pretrained representations, parameter freezing, and the differences between feature extraction and fine-tuning.                              |
| `02_pretrained_feature_extraction.py`  | Load a pretrained model, freeze its backbone, replace the classification head, and train the new head.                                               |
| `03_selective_fine_tuning.py`          | Unfreeze selected layers and experiment with discriminative learning rates and learning-rate scheduling.                                             |
| `04_transfer_learning_pipeline.py`     | Build an end-to-end training pipeline and compare feature extraction with fine-tuning using validation metrics, parameter counts, and training time. |

## ✅ Expected Outcome

By completing this module, you will be able to:

- Select and load pretrained models for new tasks.
- Implement feature extraction and fine-tuning in PyTorch.
- Control which parameters are trainable.
- Configure learning rates for pretrained and newly initialized layers.
- Build reproducible transfer learning pipelines.
- Evaluate predictive performance, computational cost, and overfitting.
- Choose an appropriate transfer learning strategy based on data, resources, and validation results.
