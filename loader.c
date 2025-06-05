// loader.c
// user-space loader

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <sys/resource.h>
#include <sys/stat.h>

extern const unsigned char _binary_kerinferencel_bpf_o_start[];
extern const unsigned char _binary_kerinferencel_bpf_o_end[];

#define INPUT_SIZE 784
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 10

#define HIDDEN_WEIGHTS_FILE "hweights8.bin"
#define HIDDEN_BIAS_FILE "hbias32.bin"
#define OUTPUT_WEIGHTS_FILE "outweights8.bin"
#define OUTPUT_BIAS_FILE "outbias32.bin"
#define TEST_IMAGE_FILE "sean.png" // Optional test image

#define TP_NAME "raw_syscalls"
#define TP_EVENT "sys_enter"

static int set_memlock_limit(void) {
  struct rlimit rlim = {
      .rlim_cur = RLIM_INFINITY,
      .rlim_max = RLIM_INFINITY,
  };

  int err = setrlimit(RLIMIT_MEMLOCK, &rlim);
  if (err) {
    fprintf(stderr, "Failed to set memlock limit: %s\n", strerror(errno));
  }
  return err;
}

static int read_binary_file(const char *filename, void *buffer, size_t size) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    fprintf(stderr, "Failed to open %s: %s\n", filename, strerror(errno));
    return -1;
  }

  size_t read_size = fread(buffer, 1, size, f);
  fclose(f);

  if (read_size != size) {
    fprintf(stderr, "Failed to read %s (expected %zu bytes, got %zu)\n",
            filename, size, read_size);
    return -1;
  }

  return 0;
}

static int update_map_with_data(int map_fd, uint32_t key, void *data,
                                size_t size, const char *map_name) {
  int err = bpf_map_update_elem(map_fd, &key, data, 0);
  if (err) {
    fprintf(stderr, "Failed to update %s map: %s\n", map_name, strerror(errno));
    return -1;
  }
  printf("Successfully loaded %zu bytes into %s map\n", size, map_name);
  return 0;
}

static int load_model_parameters(int map_fd_input, int map_fd_hidW,
                                 int map_fd_hidB, int map_fd_outW,
                                 int map_fd_outB) {
  int err = 0;
  uint32_t key = 0;

  int8_t *hidden_weights = malloc(INPUT_SIZE * HIDDEN_SIZE);
  int32_t *hidden_bias = malloc(HIDDEN_SIZE * sizeof(int32_t));
  int8_t *output_weights = malloc(HIDDEN_SIZE * OUTPUT_SIZE);
  int32_t *output_bias = malloc(OUTPUT_SIZE * sizeof(int32_t));
  uint8_t *input_image = malloc(INPUT_SIZE);

  if (!hidden_weights || !hidden_bias || !output_weights || !output_bias ||
      !input_image) {
    fprintf(stderr, "Failed to allocate memory for model parameters\n");
    err = -ENOMEM;
    goto cleanup;
  }

  int have_params = 1;

  // Hidden layer weights and bias
  if (read_binary_file(HIDDEN_WEIGHTS_FILE, hidden_weights,
                       INPUT_SIZE * HIDDEN_SIZE) < 0) {
    have_params = 0;
    printf("Couldn't load %s, using dummy values for hidden weights\n",
           HIDDEN_WEIGHTS_FILE);
    memset(hidden_weights, 1, INPUT_SIZE * HIDDEN_SIZE);
  }

  if (read_binary_file(HIDDEN_BIAS_FILE, hidden_bias,
                       HIDDEN_SIZE * sizeof(int32_t)) < 0) {
    have_params = 0;
    printf("Couldn't load %s, using dummy values for hidden bias\n",
           HIDDEN_BIAS_FILE);
    for (int i = 0; i < HIDDEN_SIZE; i++)
      hidden_bias[i] = 1;
  }

  // Output layer weights and bias
  if (read_binary_file(OUTPUT_WEIGHTS_FILE, output_weights,
                       HIDDEN_SIZE * OUTPUT_SIZE) < 0) {
    have_params = 0;
    printf("Couldn't load %s, using dummy values for output weights\n",
           OUTPUT_WEIGHTS_FILE);
    memset(output_weights, 1, HIDDEN_SIZE * OUTPUT_SIZE);
  }

  if (read_binary_file(OUTPUT_BIAS_FILE, output_bias,
                       OUTPUT_SIZE * sizeof(int32_t)) < 0) {
    have_params = 0;
    printf("Couldn't load %s, using dummy values for output bias\n",
           OUTPUT_BIAS_FILE);
    for (int i = 0; i < OUTPUT_SIZE; i++)
      output_bias[i] = 1;
  }

  // Try to load a test image, otherwise use dummy input
  if (read_binary_file(TEST_IMAGE_FILE, input_image, INPUT_SIZE) < 0) {
    printf("Couldn't load %s, using dummy input image\n", TEST_IMAGE_FILE);
    for (int i = 0; i < INPUT_SIZE; i++)
      input_image[i] = (i % 255); // More varied pattern
  }

  if (!have_params) {
    printf("Warning: Using dummy parameters. Models won't produce meaningful "
           "predictions.\n");
    printf("Run train.py first to generate parameter files.\n");
  }

  // Update maps
  if (update_map_with_data(map_fd_hidW, key, hidden_weights,
                           INPUT_SIZE * HIDDEN_SIZE, "hidden_weights") < 0 ||
      update_map_with_data(map_fd_hidB, key, hidden_bias,
                           HIDDEN_SIZE * sizeof(int32_t), "hidden_bias") < 0 ||
      update_map_with_data(map_fd_outW, key, output_weights,
                           HIDDEN_SIZE * OUTPUT_SIZE, "output_weights") < 0 ||
      update_map_with_data(map_fd_outB, key, output_bias,
                           OUTPUT_SIZE * sizeof(int32_t), "output_bias") < 0 ||
      update_map_with_data(map_fd_input, key, input_image, INPUT_SIZE,
                           "mnist_input") < 0) {
    err = -1;
    goto cleanup;
  }

cleanup:
  free(hidden_weights);
  free(hidden_bias);
  free(output_weights);
  free(output_bias);
  free(input_image);
  return err;
}

static void predict_digit(int *output, int output_size) {
  int max_idx = 0;
  int max_val = output[0];

  for (int i = 1; i < output_size; i++) {
    if (output[i] > max_val) {
      max_val = output[i];
      max_idx = i;
    }
  }

  printf("Predicted digit: %d (confidence value: %d)\n", max_idx, max_val);
}

int main(int argc, char **argv) {
  int err;
  struct bpf_object *obj = NULL;
  struct bpf_program *prog = NULL;
  struct bpf_link *link = NULL;
  int map_fd_input = -1, map_fd_output = -1;
  int map_fd_hidW = -1, map_fd_hidB = -1;
  int map_fd_outW = -1, map_fd_outB = -1;

  err = set_memlock_limit();
  if (err) {
    return 1;
  }

  printf("Start pointer (raw): %p\n",
         (void *)_binary_kerinferencel_bpf_o_start);
  printf("End pointer (raw): %p\n", (void *)_binary_kerinferencel_bpf_o_end);
  printf("Computed size: %zu\n", (size_t)(_binary_kerinferencel_bpf_o_end -
                                          _binary_kerinferencel_bpf_o_start));

  // Create in-memory BPF object from embedded bytecode
  if (!_binary_kerinferencel_bpf_o_start || !_binary_kerinferencel_bpf_o_end) {
    fprintf(stderr, "Error: BPF bytecode start or end is NULL\n");
    return 1;
  }

  size_t obj_size =
      (size_t)((const uint8_t *)_binary_kerinferencel_bpf_o_end -
               (const uint8_t *)_binary_kerinferencel_bpf_o_start);
  if (obj_size == 0) {
    fprintf(stderr, "Error: Computed BPF object size is 0\n");
    return 1;
  }
  struct bpf_object_open_opts open_opts = {
      .sz = sizeof(struct bpf_object_open_opts),
      .object_name = "mnist_inference_8bit_small",
  };
  printf("BPF bytecode start: %p\n", _binary_kerinferencel_bpf_o_start);
  printf("BPF bytecode end: %p\n", _binary_kerinferencel_bpf_o_end);
  printf("Computed object size: %zu bytes\n", obj_size);

  obj = bpf_object__open_mem((const void *)_binary_kerinferencel_bpf_o_start,
                             obj_size, &open_opts);
  if (!obj) {
    fprintf(stderr, "Failed to open BPF object: %s\n", strerror(errno));
    return 1;
  }

  prog = bpf_object__find_program_by_name(obj, "kprobe/do_mnist_inference");
  if (!prog) {
    fprintf(stderr, "Couldn't find BPF program.\n");
    goto cleanup;
  }

  bpf_program__set_type(prog, BPF_PROG_TYPE_TRACEPOINT);
  if (bpf_program__get_type(prog) != BPF_PROG_TYPE_TRACEPOINT) {
    fprintf(stderr, "Program type mismatch: expected TRACEPOINT\n");
    goto cleanup;
  }

  err = bpf_object__load(obj);
  if (err) {
    fprintf(stderr, "Failed to load BPF object: %s\n", strerror(errno));
    goto cleanup;
  }

  // Retrieve map FDs (map names must match those in the BPF program)
  map_fd_input = bpf_object__find_map_fd_by_name(obj, "mnist_input");
  map_fd_hidW = bpf_object__find_map_fd_by_name(obj, "hidden_weights");
  map_fd_hidB = bpf_object__find_map_fd_by_name(obj, "hidden_bias");
  map_fd_outW = bpf_object__find_map_fd_by_name(obj, "output_weights");
  map_fd_outB = bpf_object__find_map_fd_by_name(obj, "output_bias");
  map_fd_output = bpf_object__find_map_fd_by_name(obj, "mnist_output");

  if (map_fd_input < 0 || map_fd_hidW < 0 || map_fd_hidB < 0 ||
      map_fd_outW < 0 || map_fd_outB < 0 || map_fd_output < 0) {
    fprintf(stderr, "Failed to get map FDs: %s\n", strerror(errno));
    goto cleanup;
  }

  err = load_model_parameters(map_fd_input, map_fd_hidW, map_fd_hidB,
                              map_fd_outW, map_fd_outB);
  if (err) {
    fprintf(stderr, "Error loading parameters into maps.\n");
    goto cleanup;
  }

  // Attach program to tracepoint
  link = bpf_program__attach_tracepoint(prog, TP_NAME, TP_EVENT);
  err = libbpf_get_error(link);
  if (err) {
    fprintf(stderr, "Failed to attach tracepoint: %s\n", strerror(-err));
    link = NULL;
    goto cleanup;
  }
  printf("program attached to %s:%s tracepoint.\n", TP_NAME, TP_EVENT);

  printf("Triggering inference by executing a syscall...\n");
  getpid();
  usleep(100000);

  int output[OUTPUT_SIZE] = {0};
  uint32_t key = 0;

  err = bpf_map_lookup_elem(map_fd_output, &key, output);
  if (err) {
    fprintf(stderr, "Failed to read output map: %s\n", strerror(errno));
    goto cleanup;
  }

  printf("MNIST Output:\n");
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    printf(" %d", output[i]);
  }
  printf("\n");

  // Determine the predicted digit
  predict_digit(output, OUTPUT_SIZE);

cleanup:
  if (link)
    bpf_link__destroy(link);
  if (obj)
    bpf_object__close(obj);

  return err ? 1 : 0;
}
