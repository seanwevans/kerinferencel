# eBPF Neural Network Inference

An efficient implementation of MNIST digit recognition using eBPF (extended Berkeley Packet Filter) that runs directly in the Linux kernel.

## Overview

This project demonstrates how to run machine learning inference workloads directly in the Linux kernel using eBPF. It implements a small neural network with a single hidden layer that can recognize handwritten digits from the MNIST dataset.

Key features:
- Quantized neural network (8-bit weights, 32-bit bias)
- Single hidden layer with 32 neurons and LeakyReLU activation
- Parameters trained using PyTorch and quantized for kernel execution
- BPF maps for parameter storage and input/output exchange
- Integration with system tracepoints for triggering inference

## Project Structure

- `kerinferencel.bpf.c` - eBPF program that performs the neural network inference
- `loader.c` - User-space loader that loads the BPF program into the kernel
- `train.py` - Python script to train the model using PyTorch and export quantized parameters
- `infer.py` - Python script to load an image and trigger inference through the loaded eBPF program
- `vmlinux.h` - Minimal header for BPF development
- Parameter files:
  - `hweights8.bin` - Hidden layer weights (8-bit)
  - `hbias32.bin` - Hidden layer biases (32-bit)
  - `outweights8.bin` - Output layer weights (8-bit)
  - `outbias32.bin` - Output layer biases (32-bit)

## Prerequisites

- Linux kernel 5.10+ with eBPF support
- clang and LLVM (for BPF compilation)
- libbpf development files
- Python 3.6+ with PyTorch, torchvision, and numpy (for training)
- BPF tools (`bpftool`)

## Building

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install clang llvm libbpf-dev python3-pip bpftool
pip3 install torch torchvision numpy pillow

# Build the project
make
```

## Usage

### Training the Model

To train the model and generate the parameter files:

```bash
./train.py
```

This will:
1. Download the MNIST dataset
2. Train a small neural network using PyTorch
3. Quantize the model parameters
4. Export the parameters to binary files for eBPF usage

### Loading the BPF Program

To load the BPF inference program into the kernel:

```bash
sudo ./loader
```

The loader will:
1. Load the model parameters from binary files
2. Pin the BPF maps at `/sys/fs/bpf/mnist_input` and `/sys/fs/bpf/mnist_output`
3. Attach the BPF program to a system tracepoint
4. Execute a test inference

### Running Inference

To run inference on a custom image:

```bash
sudo python3 infer.py image.png
```

The script will:
1. Resize and preprocess the image
2. Update the input BPF map
3. Trigger the inference by executing a syscall
4. Read the results from the output BPF map
5. Print the predicted digit

## How It Works

1. The neural network architecture consists of:
   - Input layer: 784 neurons (28x28 image)
   - Hidden layer: 32 neurons with LeakyReLU activation
   - Output layer: 10 neurons (digits 0-9)

2. The eBPF program uses several BPF maps:
   - `mnist_input`: Input image data (784 uint8 values)
   - `hidden_weights`: Hidden layer weights (784×32 int8 values)
   - `hidden_bias`: Hidden layer biases (32 int32 values)
   - `output_weights`: Output layer weights (32×10 int8 values)
   - `output_bias`: Output layer biases (10 int32 values)
   - `mnist_output`: Output scores (10 int32 values)

3. When a syscall occurs, the eBPF program:
   - Reads the input image from the `mnist_input` map
   - Performs matrix multiplication with the hidden layer weights
   - Applies LeakyReLU activation
   - Performs matrix multiplication with the output layer weights
   - Writes the results to the `mnist_output` map

## Performance Optimizations

- Quantized 8-bit weights to reduce memory usage
- LeakyReLU activation for numerical stability
- Loop unrolling with `#pragma unroll` for better performance
- Stack-allocated activations to avoid additional maps

## Limitations

- Fixed input size (28x28 grayscale images)
- Simple architecture (single hidden layer)
- May require kernel headers specific to your system (adjust paths in Makefile)

## License

This project is licensed under the GPL-2.0 License - see the LICENSE file for details.

## Acknowledgments

- The eBPF and libbpf projects
- PyTorch for model training and quantization
- The MNIST dataset creators
