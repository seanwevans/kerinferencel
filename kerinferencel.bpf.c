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

// 1) Input: 784 bytes (uint8)
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, INPUT_SIZE);
  __type(key, __u32);
  __type(value, __u8);
} mnist_input SEC(".maps");

// 2) Hidden layer weights: 784*32 int8 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, HIDDEN_SIZE *INPUT_SIZE);
  __type(key, __u32);
  __type(value, __s8);
} hidden_weights SEC(".maps");

// 3) Hidden layer bias: 32 int32 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, HIDDEN_SIZE);
  __type(key, __u32);
  __type(value, int);
} hidden_bias SEC(".maps");

// 4) Output layer weights: 32*10 int8 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, OUTPUT_SIZE *HIDDEN_SIZE);
  __type(key, __u32);
  __type(value, __s8);
} output_weights SEC(".maps");

// 5) Output layer bias: 10 int32 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, OUTPUT_SIZE);
  __type(key, __u32);
  __type(value, int);
} output_bias SEC(".maps");

// 6) Output array: 10 int32 values
struct {
  __uint(type, BPF_MAP_TYPE_ARRAY);
  __uint(max_entries, OUTPUT_SIZE);
  __type(key, __u32);
  __type(value, int);
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

  // Lookup map pointers
  __u8 *in_ptr = bpf_map_lookup_elem(&mnist_input, &zero);
  __s8 *hidW_ptr = bpf_map_lookup_elem(&hidden_weights, &zero);
  int *hidB_ptr = bpf_map_lookup_elem(&hidden_bias, &zero);
  __s8 *outW_ptr = bpf_map_lookup_elem(&output_weights, &zero);
  int *outB_ptr = bpf_map_lookup_elem(&output_bias, &zero);
  int *out_ptr = bpf_map_lookup_elem(&mnist_output, &zero);

  if (!in_ptr || !hidW_ptr || !hidB_ptr || !outW_ptr || !outB_ptr || !out_ptr)
    return 0;

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
