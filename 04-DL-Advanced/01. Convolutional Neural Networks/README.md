# Convolutional Neural Networks

## Learning Objectives

- Understand how convolutional layers extract spatial features from images.
- Explain kernels, filters, feature maps, strides, padding, and receptive fields.
- Understand pooling and its role in reducing spatial dimensions.
- Build and train CNNs using PyTorch.
- Evaluate image classification models using appropriate metrics.
- Understand how CNN architecture choices affect model capacity, computational cost, and generalization.

## Core Concepts

### 1. Convolution Operations

A convolutional layer applies learnable filters across an input tensor to detect local patterns such as edges, textures, shapes, and increasingly complex visual features.

Key concepts:

- Input channels and output channels
- Kernels and filters
- Feature maps
- Stride and padding
- Spatial dimensions
- Receptive fields
- Learnable weights and biases

### 2. Output Dimensions

For an input dimension of size $$N$$, kernel size $$K$$, padding $$P$$, stride $$S$$, and dilation $$D$$, the output dimension is:

$$
O = \left\lfloor \frac{N + 2P - D(K-1) - 1}{S} + 1 \right\rfloor
$$

Understanding this calculation helps predict tensor shapes and debug CNN architectures.

### 3. Pooling and Downsampling

- Max pooling selects the largest value within each local region.
- Average pooling computes the average value within each region.
- Global average pooling reduces each feature map to a single value.

Pooling and strided convolutions can reduce computational cost while retaining useful spatial information.

### 4. CNN Architecture

A typical CNN classifier consists of:

1. Input preprocessing and normalization
2. Convolutional layers
3. Nonlinear activation functions
4. Pooling or strided downsampling
5. Deeper feature extraction blocks
6. Global pooling or flattening
7. A classification head

### 5. Training and Generalization

- Cross-entropy loss for classification
- Mini-batch training and gradient-based optimization
- Training and validation learning curves
- Early stopping and regularization
- Confusion matrices and per-class metrics
- Model checkpointing and inference

### 6. Architectural Trade-offs

- Number of channels and network depth
- Kernel size and receptive field
- Parameter count and computational complexity
- Flattening versus global average pooling
- Underfitting, overfitting, and generalization

## Implementation and Experiments

| File                                | Experiment                                                                          |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| `01_convolution_operations.py`      | Explore convolution, kernels, feature maps, stride, padding, and output dimensions. |
| `02_pooling_and_downsampling.py`    | Compare max pooling, average pooling, and global average pooling.                   |
| `03_building_a_cnn_classifier.py`   | Build and train an end-to-end CNN image classifier using PyTorch.                   |
| `04_cnn_architecture_comparison.py` | Compare CNN configurations by accuracy, parameter count, and training behavior.     |

Experiments should use reproducible seeds, clear tensor-shape checks, and visualizations where they improve understanding. Training experiments should separate training, validation, and test data and avoid using test results to select the model.

## Expected Outcome

By completing this module, you should be able to:

- Explain the mechanics of convolution and pooling.
- Calculate CNN tensor dimensions and estimate parameter counts.
- Implement and debug CNN architectures in PyTorch.
- Train, evaluate, save, and run inference with an image classifier.
- Compare architectural choices using both predictive performance and computational cost.
- Explain when CNNs are appropriate and how their feature-extraction principles differ from those of Transformers.
