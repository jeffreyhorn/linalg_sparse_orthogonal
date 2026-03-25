# Makefile for linalg_sparse_orthogonal
# Simple alternative to CMake for quick builds

CC      = cc
SYSROOT = $(shell /usr/bin/xcrun --show-sdk-path 2>/dev/null)
CFLAGS  = -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2
ifneq ($(SYSROOT),)
CFLAGS += -isysroot $(SYSROOT)
endif
LDFLAGS = -lm
INCLUDE = -Iinclude

# Directories
SRCDIR  = src
TESTDIR = tests
BENCHDIR = benchmarks
BUILDDIR = build

# Library sources
LIB_SRCS = $(SRCDIR)/sparse_types.c \
           $(SRCDIR)/sparse_matrix.c \
           $(SRCDIR)/sparse_lu.c \
           $(SRCDIR)/sparse_vector.c \
           $(SRCDIR)/sparse_reorder.c \
           $(SRCDIR)/sparse_cholesky.c \
           $(SRCDIR)/sparse_csr.c
LIB_OBJS = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(LIB_SRCS))
LIB      = $(BUILDDIR)/libsparse_lu_ortho.a

# Test sources
TEST_SRCS = $(TESTDIR)/test_sparse_matrix.c \
            $(TESTDIR)/test_sparse_lu.c \
            $(TESTDIR)/test_sparse_io.c \
            $(TESTDIR)/test_known_matrices.c \
            $(TESTDIR)/test_sparse_vector.c \
            $(TESTDIR)/test_edge_cases.c \
            $(TESTDIR)/test_integration.c \
            $(TESTDIR)/test_sparse_arith.c \
            $(TESTDIR)/test_suitesparse.c \
            $(TESTDIR)/test_reorder.c \
            $(TESTDIR)/test_cholesky.c \
            $(TESTDIR)/test_csr.c \
            $(TESTDIR)/test_matmul.c \
            $(TESTDIR)/test_threads.c
TEST_BINS = $(patsubst $(TESTDIR)/%.c,$(BUILDDIR)/%,$(TEST_SRCS))

# Benchmark sources
BENCH_SRCS = $(BENCHDIR)/bench_main.c \
             $(BENCHDIR)/bench_scaling.c \
             $(BENCHDIR)/bench_fillin.c
BENCH_BINS = $(patsubst $(BENCHDIR)/%.c,$(BUILDDIR)/%,$(BENCH_SRCS))

# Default target
.PHONY: all
all: $(LIB)

# Build directory
$(BUILDDIR):
	/bin/mkdir -p $(BUILDDIR)

# Library objects
$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# Static library
$(LIB): $(LIB_OBJS)
	ar rcs $@ $^

# Thread test needs -pthread
$(BUILDDIR)/test_threads: $(TESTDIR)/test_threads.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -I$(TESTDIR) $< -L$(BUILDDIR) -lsparse_lu_ortho $(LDFLAGS) -pthread -o $@

# Test executables (any .c in tests/)
$(BUILDDIR)/%: $(TESTDIR)/%.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -I$(TESTDIR) $< -L$(BUILDDIR) -lsparse_lu_ortho $(LDFLAGS) -o $@

# Benchmark executables
$(BUILDDIR)/bench_%: $(BENCHDIR)/bench_%.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDE) $< -L$(BUILDDIR) -lsparse_lu_ortho $(LDFLAGS) -o $@

# Smoke test
.PHONY: smoke
smoke: $(BUILDDIR)/smoke_test
	$(BUILDDIR)/smoke_test

# Run all tests
.PHONY: test
test: $(TEST_BINS)
	@for t in $(TEST_BINS); do \
		echo "=== Running $$(basename $$t) ==="; \
		$$t || exit 1; \
		echo; \
	done
	@echo "All tests passed."

# Run benchmarks
.PHONY: bench
bench: $(BENCH_BINS)
	@for b in $(BENCH_BINS); do \
		echo "=== Running $$(basename $$b) ==="; \
		$$b; \
		echo; \
	done

# Benchmark SuiteSparse matrices (both pivoting modes)
.PHONY: bench-suitesparse
bench-suitesparse: $(BUILDDIR)/bench_main
	@$(BUILDDIR)/bench_main --dir tests/data/suitesparse --pivot partial --repeat 3
	@$(BUILDDIR)/bench_main --dir tests/data/suitesparse --pivot complete --repeat 3

# Build and test with UBSan
.PHONY: sanitize
sanitize: CFLAGS += -fsanitize=undefined -fno-omit-frame-pointer -g -O1
sanitize: LDFLAGS += -fsanitize=undefined
sanitize: clean test

# Build and test with ASan
# NOTE: Apple Clang ASan hangs on macOS. Use GCC or LLVM clang instead:
#   make asan CC=gcc-14
#   make asan CC=/opt/homebrew/opt/llvm/bin/clang
# On Linux this works with the default compiler.
.PHONY: asan
asan: CFLAGS += -fsanitize=address -fno-omit-frame-pointer -g -O1
asan: LDFLAGS += -fsanitize=address
asan: export MallocNanoZone=0
asan: export ASAN_OPTIONS=detect_leaks=0
asan: clean test

# Build and test with both ASan and UBSan
.PHONY: sanitize-all
sanitize-all: CFLAGS += -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1
sanitize-all: LDFLAGS += -fsanitize=address,undefined
sanitize-all: export MallocNanoZone=0
sanitize-all: export ASAN_OPTIONS=detect_leaks=0
sanitize-all: clean test

# Clean
.PHONY: clean
clean:
	/bin/rm -rf $(BUILDDIR)
