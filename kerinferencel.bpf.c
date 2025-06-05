// SPDX-License-Identifier: GPL-2.0
//
// mnist_inference_8bit_small.bpf.c
// Minimal eBPF program for quantized MNIST inference with LeakyReLU.
// Single hidden layer of 32 units, parameters stored as int8 (weights) and
// int32 (biases).

#include "vmlinux.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <linux/bpf.h>

// Model dimensions
#define INPUT_SIZE 784
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 10

// Maximum verifier complexity - helps with large BPF programs
#define MAX_LAYERS 2

// Structs to hold entire arrays as single map values
struct input_val {
  __u8 input[INPUT_SIZE];
};

struct hidden_weights_val {
  __s8 weights[INPUT_SIZE * HIDDEN_SIZE];
};

struct hidden_bias_val {
  int bias[HIDDEN_SIZE];
};

struct output_weights_val {
  __s8 weights[HIDDEN_SIZE * OUTPUT_SIZE];
};

struct output_bias_val {
  int bias[OUTPUT_SIZE];
};

struct output_val {
  int output[OUTPUT_SIZE];
};

// 1) Input: 784 bytes (uint8) stored as single value
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __type(value, struct input_val);
} mnist_input SEC(".maps");

// 2) Hidden layer weights: 784*32 int8 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __type(value, struct hidden_weights_val);
} hidden_weights SEC(".maps");

// 3) Hidden layer bias: 32 int32 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __type(value, struct hidden_bias_val);
} hidden_bias SEC(".maps");

// 4) Output layer weights: 32*10 int8 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __type(value, struct output_weights_val);
} output_weights SEC(".maps");

// 5) Output layer bias: 10 int32 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __type(value, struct output_bias_val);
} output_bias SEC(".maps");

// 6) Output array: 10 int32 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, 1);
  __type(key, __u32);
  __type(value, struct output_val);
} mnist_output SEC(".maps");

// Leaky ReLU activation function
static __always_inline int leaky_relu_int32(int x) {
  if (x >= 0) {
    return x;
  } else {
    unsigned int abs_x = (unsigned int)(-x);
    return -(abs_x / 100U);
  }
}

SEC("tracepoint/raw_syscalls/sys_enter")
int bpf_mnist_infer(struct trace_event_raw_sys_enter *ctx) {
  __u32 zero = 0;

  // Lookup map pointers to the value structs
  struct input_val *in_val = bpf_map_lookup_elem(&mnist_input, &zero);
  struct hidden_weights_val *hidW_val =
      bpf_map_lookup_elem(&hidden_weights, &zero);
  struct hidden_bias_val *hidB_val = bpf_map_lookup_elem(&hidden_bias, &zero);
  struct output_weights_val *outW_val =
      bpf_map_lookup_elem(&output_weights, &zero);
  struct output_bias_val *outB_val = bpf_map_lookup_elem(&output_bias, &zero);
  struct output_val *out_val = bpf_map_lookup_elem(&mnist_output, &zero);

  if (!in_val || !hidW_val || !hidB_val || !outW_val || !outB_val || !out_val)
    return 0;

  __u8 *in_ptr = in_val->input;
  __s8 *hidW_ptr = hidW_val->weights;
  int *hidB_ptr = hidB_val->bias;
  __s8 *outW_ptr = outW_val->weights;
  int *outB_ptr = outB_val->bias;
  int *out_ptr = out_val->output;

  // Temporary stack array for hidden activations (int32)
  int hidden_layer[HIDDEN_SIZE];

  for (int layer = 0; layer < MAX_LAYERS; layer++) {
    if (layer == 0) {
#pragma unroll
      for (int j = 0; j < HIDDEN_SIZE; j++) {
        int sum_j = hidB_ptr[j]; // bias is int32

#pragma unroll 16
        for (int i = 0; i < INPUT_SIZE; i++) {
          int weight = hidW_ptr[j * INPUT_SIZE + i]; // int8, sign-extended
          int input_val = in_ptr[i];                 // uint8, treat as int
          sum_j += (weight * input_val);
        }
        hidden_layer[j] = leaky_relu_int32(sum_j);
      }
    } else if (layer == 1) {
#pragma unroll
      for (int o = 0; o < OUTPUT_SIZE; o++) {
        int sum_o = outB_ptr[o];

#pragma unroll
        for (int j = 0; j < HIDDEN_SIZE; j++) {
          int weight = outW_ptr[o * HIDDEN_SIZE + j]; // int8
          int hidden_val = hidden_layer[j];
          sum_o += (weight * hidden_val);
        }
        out_ptr[o] = leaky_relu_int32(sum_o);
      }
    }
  }

  bpf_trace_printk("BPF_INFER: inference executed\n",
                   sizeof("BPF_INFER: inference executed\n") - 1);
  return 0;
}

char _license[] SEC("license") = "GPL";
