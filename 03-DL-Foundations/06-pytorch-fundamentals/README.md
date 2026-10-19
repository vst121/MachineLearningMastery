# PyTorch Fundamentals

This module introduces the essential PyTorch concepts and workflows required to build, train, evaluate, and deploy neural networks. It focuses on tensor operations, shape management, device placement, model construction, training loops, inference, and model persistence.

## Learning Objectives

- Create and manipulate tensors using PyTorch.
- Understand tensor shapes, dimensions, broadcasting, and matrix operations.
- Manage data types and CPU/GPU device placement.
- Build neural networks using `torch.nn.Module`.
- Implement training loops with automatic differentiation and optimizers.
- Distinguish training, evaluation, and inference modes.
- Save, load, and restore trained model parameters.
- Apply reliable practices for developing deep learning applications.

## Core Concepts

| Topic                    | Key Concepts                                                           |
| ------------------------ | ---------------------------------------------------------------------- |
| Tensor Operations        | Creation, indexing, slicing, reshaping, and arithmetic                 |
| Tensor Shapes            | Dimensions, broadcasting, matrix multiplication, and batch operations  |
| Device Management        | CPU, CUDA, device placement, and device compatibility                  |
| Model Construction       | `nn.Module`, layers, parameters, and model composition                 |
| Training Loops           | Forward pass, loss calculation, backpropagation, and optimizer updates |
| Evaluation and Inference | `model.eval()`, `torch.no_grad()`, and `torch.inference_mode()`        |
| Model Persistence        | State dictionaries, saving, loading, and reproducibility               |

## Implementation and Experiments

1. `01_tensor_operations.py`
   - Create tensors and perform common tensor operations.
   - Explore indexing, slicing, reshaping, and transposition.
   - Practice arithmetic, matrix multiplication, and broadcasting.
   - Understand tensor data types, views, copies, and NumPy interoperability.

2. `02_tensor_shapes_and_broadcasting.py`
   - Understand tensor dimensions and shape compatibility.
   - Practice reshaping, squeezing, unsqueezing, and transposition.
   - Explore broadcasting rules and batch operations.
   - Diagnose common tensor shape mismatches.

3. `03_device_management.py`
   - Detect available computing devices.
   - Move tensors and models between CPU and GPU.
   - Understand device compatibility and common device mismatch errors.
   - Compare execution behavior across available devices.

4. `04_nn_module_and_model_composition.py`
   - Build neural networks using `nn.Module`.
   - Explore layers, trainable parameters, and model composition.
   - Inspect model architectures and parameter counts.
   - Understand parameter initialization and model state.

5. `05_training_loop_from_scratch.py`
   - Implement a complete PyTorch training loop.
   - Integrate forward propagation, loss calculation, backpropagation, and optimizer updates.
   - Track training loss and validation performance.
   - Understand gradient management and parameter updates.

6. `06_evaluation_and_inference.py`
   - Implement validation and inference workflows.
   - Understand the differences between `model.train()` and `model.eval()`.
   - Compare `torch.no_grad()` and `torch.inference_mode()`.
   - Evaluate model predictions using appropriate metrics.

7. `07_model_saving_and_loading.py`
   - Save and restore model parameters using `state_dict`.
   - Reconstruct a model and load its saved parameters.
   - Verify prediction consistency before and after loading.
   - Understand reliable model checkpointing practices.

## Expected Outcome

By completing this module, you will be able to build and manage a complete neural-network workflow in PyTorch, from tensor manipulation and model construction to training, evaluation, inference, and model persistence.
