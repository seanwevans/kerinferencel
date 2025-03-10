// SPDX-License-Identifier: GPL-2.0
//
// mnist_inference_8bit_small.bpf.c
// Minimal eBPF program for quantized MNIST inference with LeakyReLU.
// Single hidden layer of 32 units, parameters stored as int8 (weights) and
// int32 (biases).
//
// Build with:
//   clang -O2 -g -target bpf -c mnist_inference_8bit_small.bpf.c -o
//   mnist_inference_8bit_small.bpf.o

#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <vmlinux.h>

// 1) Input: 784 bytes (uint8)
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __array(value, __u8[784]);
} mnist_input SEC(".maps");

// 2) Hidden layer weights: 784*32 int8 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __array(value, __s8[784 * 32]);
} hidden_weights SEC(".maps");

// 3) Hidden layer bias: 32 int32 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __array(value, int[32]);
} hidden_bias SEC(".maps");

// 4) Output layer weights: 32*10 int8 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __array(value, __s8[32 * 10]);
} output_weights SEC(".maps");

// 5) Output layer bias: 10 int32 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __array(value, int[10]);
} output_bias SEC(".maps");

// 6) Output array: 10 int32 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __array(value, int[10]);
} mnist_output SEC(".maps");

// Model dimensions
#define INPUT_SIZE 784
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 10

// Maximum verifier complexity - helps with large BPF programs
#define MAX_LAYERS 2

static __always_inline int leaky_relu_int32(int x) {
  return (x >= 0) ? x : (x / 100);
}

SEC("tracepoint/raw_syscalls/sys_enter")
int bpf_mnist_infer(struct trace_event_raw_sys_enter *ctx) {
  __u32 zero = 0;

  // Lookup map pointers
  __u8(*in_ptr)[INPUT_SIZE] = bpf_map_lookup_elem(&mnist_input, &zero);
  __s8(*hidW_ptr)[INPUT_SIZE * HIDDEN_SIZE] =
      bpf_map_lookup_elem(&hidden_weights, &zero);
  int(*hidB_ptr)[HIDDEN_SIZE] = bpf_map_lookup_elem(&hidden_bias, &zero);
  __s8(*outW_ptr)[HIDDEN_SIZE * OUTPUT_SIZE] =
      bpf_map_lookup_elem(&output_weights, &zero);
  int(*outB_ptr)[OUTPUT_SIZE] = bpf_map_lookup_elem(&output_bias, &zero);
  int(*out_ptr)[OUTPUT_SIZE] = bpf_map_lookup_elem(&mnist_output, &zero);

  if (!in_ptr || !hidW_ptr || !hidB_ptr || !outW_ptr || !outB_ptr || !out_ptr)
    return 0;

  // Temporary stack array for hidden activations (int32)
  int hidden_layer[HIDDEN_SIZE];

  for (int layer = 0; layer < MAX_LAYERS; layer++) {
    if (layer == 0) {
// hidden[j] = LeakyReLU( bias[j] + sum_{i=0}^{783} (hidden_weights[j*INPUT_SIZE
// + i] * input[i]) )
#pragma unroll
      for (int j = 0; j < HIDDEN_SIZE; j++) {
        int sum_j = (*hidB_ptr)[j]; // bias is int32

// break into chunks
#pragma unroll 16
        for (int i = 0; i < INPUT_SIZE; i++) {
          int weight = (*hidW_ptr)[j * INPUT_SIZE + i]; // int8, sign-extended
          int input_val = (*in_ptr)[i];                 // uint8, treat as int
          sum_j += (weight * input_val);
        }
        hidden_layer[j] = leaky_relu_int32(sum_j);
      }
    } else if (layer == 1) {
// output[o] = LeakyReLU( bias[o] + sum_{j=0}^{HIDDEN_SIZE-1}
// (output_weights[o*HIDDEN_SIZE + j] * hidden_layer[j]) )
#pragma unroll
      for (int o = 0; o < OUTPUT_SIZE; o++) {
        int sum_o = (*outB_ptr)[o];

#pragma unroll
        for (int j = 0; j < HIDDEN_SIZE; j++) {
          int weight = (*outW_ptr)[o * HIDDEN_SIZE + j]; // int8
          int hidden_val = hidden_layer[j];
          sum_o += (weight * hidden_val);
        }
        (*out_ptr)[o] = leaky_relu_int32(sum_o);
      }
    }
  }
  bpf_trace_printk("BPF_INFER: inference executed\n");
  return 0;
}

char _license[] SEC("license") = "GPL";
