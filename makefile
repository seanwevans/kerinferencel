BPF_CLANG ?= clang
BPF_LLVM_STRIP ?= llvm-strip
LDFLAGS = -lbpf
CC ?= gcc
CFLAGS ?= -O2 -g -Wall

# Location of the kernel headers. Override if your headers live elsewhere.
KDIR ?= /lib/modules/$(shell uname -r)/build
KERN_HEADERS = -I$(KDIR)/arch/x86/include/generated/uapi \
               -I$(KDIR)/arch/x86/include/uapi

BPF_SRC = kerinferencel.bpf.c
BPF_OBJ = kerinferencel.bpf.o
BPF_BIN = kerinferencel.bpf.bin.o
LOADER_SRC = loader.c
LOADER_OBJ = loader

.PHONY: all clean

all: $(BPF_BIN) $(LOADER_OBJ)

# Build eBPF Object
$(BPF_OBJ): $(BPF_SRC)
	$(BPF_CLANG) $(KERN_HEADERS) -O2 -g -target bpf -c $< -o $@
	$(BPF_LLVM_STRIP) -g $@

# Embed bytecode into ELF object
$(BPF_BIN): $(BPF_OBJ)
	ld -r -b binary $< -o $@

# Pull in the BPF_BIN object
$(LOADER_OBJ): $(BPF_BIN) $(LOADER_SRC) 
	$(CC) $(CFLAGS) -o $@ $(BPF_BIN) $(LOADER_SRC) $(LDFLAGS)
    
clean:
	rm -f $(BPF_OBJ) $(BPF_BIN) $(LOADER_OBJ)

