#!/usr/bin/env python3

import binascii
import os
import subprocess
import sys
import time
import json

import numpy as np
from PIL import Image

# Adjust these paths to match where your maps are pinned.
INPUT_MAP_PATH = "/sys/fs/bpf/mnist_input"
OUTPUT_MAP_PATH = "/sys/fs/bpf/mnist_output"
TRACE_PIPE = "/sys/kernel/debug/tracing/trace_pipe"


def load_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.uint8)
    flat = arr.flatten()

    return flat


def update_map(map_path, key, data_bytes):
    # The key is a 4-byte little-endian value.
    key_hex = key.to_bytes(4, byteorder="little").hex()
    data_hex = binascii.hexlify(data_bytes).decode("ascii")
    cmd = [
        "bpftool",
        "map",
        "update",
        "pinned",
        map_path,
        "key",
        key_hex,
        "value",
        "data",
        data_hex,
    ]
    subprocess.run(cmd, check=True)
    print(f"Updated {map_path} at key {key}")


def lookup_map(map_path, key):
    """Lookup a BPF map entry using bpftool in JSON mode."""
    key_hex = key.to_bytes(4, byteorder="little").hex()
    cmd = [
        "bpftool",
        "-j",
        "map",
        "lookup",
        "pinned",
        map_path,
        "key",
        key_hex,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    return result.stdout


def trigger_inference():
    # Trigger the BPF program by executing a syscall.
    # Since the eBPF program is attached to a tracepoint,
    # any syscall (like getpid) will trigger it.
    os.getpid()


def parse_output(data):
    """Decode JSON output from bpftool into an array of int32 values."""
    obj = json.loads(data.decode("utf-8"))
    if isinstance(obj, list):
        if not obj:
            raise ValueError("No data returned from bpftool")
        obj = obj[0]
    if "value" not in obj:
        raise ValueError("Unexpected bpftool output")
    value_bytes = bytes(obj["value"])
    if len(value_bytes) < 40:
        raise ValueError("Unexpected output map size")
    arr = np.frombuffer(value_bytes, dtype=np.int32)
    return arr


def check_kernel_trace(timeout=1.0):
    """
    Read the kernel trace pipe for a short time and check for our marker.
    Returns True if the marker "BPF_INFER:" is seen.
    """
    marker = b"BPF_INFER:"
    found = False
    try:
        with open(TRACE_PIPE, "rb") as f:
            start = time.time()
            while time.time() - start < timeout:
                # Non-blocking read of a line; if no data, sleep briefly.
                line = f.readline()
                if marker in line:
                    print("Kernel trace marker detected:", line.decode().strip())
                    found = True
                    break
    except Exception as e:
        print("Failed to read trace_pipe:", e)
    return found


def main(image_path):
    flat = load_image(image_path)
    if flat.size != 784:
        print("Error: Image did not yield 784 pixels")
        sys.exit(1)

    update_map(INPUT_MAP_PATH, 0, flat.tobytes())
    trigger_inference()
    if check_kernel_trace(timeout=1.0):
        print("Verification: BPF program executed in kernel space.")
    else:
        print(
            "Warning: No kernel trace marker detected. Verify that tracing is enabled and accessible."
        )

    # Allow the BPF program to run
    time.sleep(0.1)
    out_data = lookup_map(OUTPUT_MAP_PATH, 0)
    output = parse_output(out_data)
    print("MNIST Output (raw int32 values):", output)

    predicted_digit = int(np.argmax(output))
    print("Predicted digit:", predicted_digit)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} <image_file>".format(sys.argv[0]))
        sys.exit(1)
    main(sys.argv[1])
