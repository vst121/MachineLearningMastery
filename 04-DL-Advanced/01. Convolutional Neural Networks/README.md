# Convolution Operations

This module explores the mathematical foundations and implementation of convolution operations used in Convolutional Neural Networks (CNNs). It progresses from manual convolution calculations to multi-channel tensor operations and PyTorch verification.

## Learning Objectives

- Understand convolution and cross-correlation operations.
- Calculate output tensor dimensions using kernel size, stride, padding, and dilation.
- Understand input channels, output channels, filters, and feature maps.
- Implement convolution operations using NumPy.
- Verify manual implementations against PyTorch.
- Calculate the number of trainable parameters in a convolutional layer.

## Core Concepts

### 1. Convolution and Cross-Correlation

A convolutional operation applies a kernel to local regions of an input tensor. Each output value is calculated by multiplying corresponding input and kernel elements and summing the results.

For a single-channel two-dimensional input, the operation used by typical CNN libraries is cross-correlation:

$$
Y_{i,j} =
\sum_{m=0}^{K_h-1}
\sum_{n=0}^{K_w-1}
X_{i+m,j+n}K_{m,n}
$$

Where:

- $$X$$ is the input matrix.
- $$K$$ is the kernel or filter.
- $$Y$$ is the output feature map.
- $$K_h$$ and $$K_w$$ are the kernel height and width.
- $$i$$ and $$j$$ identify an output position.

Mathematical convolution flips the kernel along its spatial dimensions before applying the operation. CNN libraries such as PyTorch generally use cross-correlation but refer to the operation as convolution.

### 2. Input and Output Tensor Dimensions

For a two-dimensional convolution, PyTorch uses the following input shape:

$$
X \in \mathbb{R}^{N \times C_{\mathrm{in}} \times H \times W}
$$

The output tensor has the shape:

$$
Y \in \mathbb{R}^{N \times C_{\mathrm{out}} \times H_{\mathrm{out}} \times W_{\mathrm{out}}}
$$

Where:

- $$N$$ is the batch size.
- $$C_{\mathrm{in}}$$ is the number of input channels.
- $$C_{\mathrm{out}}$$ is the number of output channels.
- $$H$$ and $$W$$ are the input height and width.
- $$H_{\mathrm{out}}$$ and $$W_{\mathrm{out}}$$ are the output height and width.

For a single spatial dimension, the general output-size formula is:

$$
L_{\mathrm{out}} =
\left\lfloor
\frac{L_{\mathrm{in}}+2P-D(K-1)-1}{S}
+1
\right\rfloor
$$

Where:

- $$L_{\mathrm{in}}$$ is the input dimension.
- $$K$$ is the kernel size.
- $$S$$ is the stride.
- $$P$$ is the padding applied to each side.
- $$D$$ is the dilation.
- $$\lfloor \cdot \rfloor$$ denotes rounding down to the nearest integer.

Apply this formula independently to the height and width dimensions.

For a two-dimensional convolution:

$$
H_{\mathrm{out}} =
\left\lfloor
\frac{H+2P_h-D_h(K_h-1)-1}{S_h}
+1
\right\rfloor
$$

$$
W_{\mathrm{out}} =
\left\lfloor
\frac{W+2P_w-D_w(K_w-1)-1}{S_w}
+1
\right\rfloor
$$

### 3. Stride, Padding, and Dilation

**Stride** determines how many positions the kernel moves between consecutive operations.

**Padding** adds values around the input boundaries. Zero padding is commonly used in CNNs.

**Dilation** determines the spacing between kernel elements and increases the receptive field without increasing the number of kernel parameters.

The effective kernel size is:

$$
K_{\mathrm{eff}} = D(K-1)+1
$$

The output-size formula can equivalently be expressed as:

$$
L_{\mathrm{out}} =
\left\lfloor
\frac{L_{\mathrm{in}}+2P-K_{\mathrm{eff}}}{S}
\right\rfloor+1
$$

For a convolution with an odd-sized kernel, unit stride, unit dilation, and symmetric padding, the spatial dimensions are preserved when:

$$
P = \frac{K-1}{2}
$$

For example, a kernel of size $$3 \times 3$$ with padding of one preserves the height and width when the stride is one.

### 4. Multi-Channel Convolution

A convolutional filter spans all input channels. For each output channel, the operation combines the spatial convolution results from all input channels.

The output at position $$i,j$$ for output channel $$o$$ is:

$$
Y_{n,o,i,j} =
b_o +
\sum_{c=0}^{C_{\mathrm{in}}-1}
\sum_{m=0}^{K_h-1}
\sum_{q=0}^{K_w-1}
X_{n,c,iS_h+mD_h-P_h,\,
jS_w+qD_w-P_w}
K_{o,c,m,q}
$$

Here, $$b_o$$ is the bias for output channel $$o$$. Input positions outside the padded input are treated as zero.

The filter tensor has the shape:

$$
K \in
\mathbb{R}^{C_{\mathrm{out}} \times C_{\mathrm{in}} \times K_h \times K_w}
$$

Each output channel produces one feature map. Therefore, the number of output channels equals the number of filters.

### 5. Trainable Parameters

For a standard convolutional layer with a bias term, the total number of trainable parameters is:

$$
P_{\mathrm{total}} =
C_{\mathrm{out}}
\left(
C_{\mathrm{in}}K_hK_w+1
\right)
$$

Without bias:

$$
P_{\mathrm{total}} =
C_{\mathrm{out}}C_{\mathrm{in}}K_hK_w
$$

The number of parameters depends on the channel counts and kernel dimensions, not directly on the input image height or width.

### 6. One-Dimensional Convolution

The same principles apply to one-dimensional signals, sequences, and temporal data.

For an input of length $$L_{\mathrm{in}}$$:

$$
L_{\mathrm{out}} =
\left\lfloor
\frac{L_{\mathrm{in}}+2P-D(K-1)-1}{S}
+1
\right\rfloor
$$

This formulation is useful for understanding temporal convolutions and the dimensional behavior of sequence-processing networks.

## Implementation and Experiments

| File                                     | Focus                    | Main Experiments                                                                                |
| ---------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------- |
| `01_convolution_operations.py`           | Convolution fundamentals | Implement 2D cross-correlation using NumPy and verify a small numerical example.                |
| `02_convolution_dimensions.py`           | Tensor dimensions        | Experiment with kernel size, stride, padding, and dilation; validate output-size calculations.  |
| `03_multi_channel_convolution.py`        | Channels and filters     | Implement multi-channel convolution, inspect tensor shapes, and calculate parameter counts.     |
| `04_convolution_pytorch_verification.py` | Framework verification   | Compare NumPy calculations with `torch.nn.functional.conv2d` and test different configurations. |

## Expected Outcome

After completing these experiments, you should be able to:

- Calculate convolution output dimensions without executing a model.
- Explain how convolution transforms an input tensor into feature maps.
- Distinguish between input channels, output channels, and kernel dimensions.
- Understand the influence of stride, padding, and dilation on spatial resolution.
- Calculate the number of trainable convolutional parameters.
- Validate custom implementations against PyTorch using numerical assertions.

These foundations establish the mathematical and tensor-level understanding needed to build, train, and debug complete CNN architectures.
