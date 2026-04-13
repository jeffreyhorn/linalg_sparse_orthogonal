# Makefile for linalg_sparse_orthogonal
# Simple alternative to CMake for quick builds

CC      = cc
SYSROOT = $(shell /usr/bin/xcrun --show-sdk-path 2>/dev/null)
CFLAGS  = -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -O2
ifneq ($(SYSROOT),)
CFLAGS += -isysroot $(SYSROOT)
endif
LDFLAGS = -lm
# When SPARSE_MUTEX is enabled, all binaries need -pthread
ifdef SPARSE_MUTEX
CFLAGS  += -DSPARSE_MUTEX
LDFLAGS += -pthread
endif
# When SPARSE_OPENMP is enabled, add OpenMP flags.
# On macOS with Apple Clang, use -Xpreprocessor -fopenmp + Homebrew libomp.
# On Linux/GCC, use -fopenmp directly.
# Prefer the `make omp` target which handles this automatically.
ifdef SPARSE_OPENMP
ifeq ($(shell uname -s),Darwin)
LIBOMP_FLAG_PREFIX := $(firstword $(wildcard /usr/local/opt/libomp /opt/homebrew/opt/libomp))
ifneq ($(LIBOMP_FLAG_PREFIX),)
CFLAGS  += -DSPARSE_OPENMP -Xpreprocessor -fopenmp -I$(LIBOMP_FLAG_PREFIX)/include
LDFLAGS += -L$(LIBOMP_FLAG_PREFIX)/lib -lomp
else
$(error libomp not found. Install with 'brew install libomp' or use 'make omp')
endif
else
CFLAGS  += -DSPARSE_OPENMP -fopenmp
LDFLAGS += -fopenmp
endif
endif
INCLUDE = -I$(BUILDDIR)/include -Iinclude

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
           $(SRCDIR)/sparse_csr.c \
           $(SRCDIR)/sparse_iterative.c \
           $(SRCDIR)/sparse_ilu.c \
           $(SRCDIR)/sparse_qr.c \
           $(SRCDIR)/sparse_dense.c \
           $(SRCDIR)/sparse_bidiag.c \
           $(SRCDIR)/sparse_svd.c \
           $(SRCDIR)/sparse_lu_csr.c \
           $(SRCDIR)/sparse_ldlt.c \
           $(SRCDIR)/sparse_ic.c \
           $(SRCDIR)/sparse_etree.c \
           $(SRCDIR)/sparse_analysis.c \
           $(SRCDIR)/sparse_colamd.c
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
            $(TESTDIR)/test_threads.c \
            $(TESTDIR)/test_sprint4_integration.c \
            $(TESTDIR)/test_iterative.c \
            $(TESTDIR)/test_ilu.c \
            $(TESTDIR)/test_omp.c \
            $(TESTDIR)/test_sprint5_integration.c \
            $(TESTDIR)/test_qr.c \
            $(TESTDIR)/test_sprint6_integration.c \
            $(TESTDIR)/test_dense.c \
            $(TESTDIR)/test_bidiag.c \
            $(TESTDIR)/test_svd.c \
            $(TESTDIR)/test_sprint8_integration.c \
            $(TESTDIR)/test_fuzz.c \
            $(TESTDIR)/test_lu_csr.c \
            $(TESTDIR)/test_block_solvers.c \
            $(TESTDIR)/test_sprint10_integration.c \
            $(TESTDIR)/test_sprint11_integration.c \
            $(TESTDIR)/test_ldlt.c \
            $(TESTDIR)/test_sprint12_integration.c \
            $(TESTDIR)/test_ic.c \
            $(TESTDIR)/test_minres.c \
            $(TESTDIR)/test_sprint13_integration.c \
            $(TESTDIR)/test_etree.c \
            $(TESTDIR)/test_colamd.c
TEST_BINS = $(patsubst $(TESTDIR)/%.c,$(BUILDDIR)/%,$(TEST_SRCS))

# Benchmark sources
BENCH_SRCS = $(BENCHDIR)/bench_main.c \
             $(BENCHDIR)/bench_scaling.c \
             $(BENCHDIR)/bench_fillin.c \
             $(BENCHDIR)/bench_convergence.c \
             $(BENCHDIR)/bench_svd.c \
             $(BENCHDIR)/bench_refactor.c \
             $(BENCHDIR)/bench_colamd.c
BENCH_BINS = $(patsubst $(BENCHDIR)/%.c,$(BUILDDIR)/%,$(BENCH_SRCS))

# Example sources
EXDIR = examples
EX_SRCS = $(wildcard $(EXDIR)/*.c)
EX_BINS = $(patsubst $(EXDIR)/%.c,$(BUILDDIR)/%,$(EX_SRCS))

# Default target
.PHONY: all
all: $(LIB)

# Build directory
$(BUILDDIR):
	/bin/mkdir -p $(BUILDDIR) $(BUILDDIR)/include
	@sed -e 's|@SPARSE_VERSION_MAJOR@|$(VERSION_MAJOR)|g' \
		-e 's|@SPARSE_VERSION_MINOR@|$(VERSION_MINOR)|g' \
		-e 's|@SPARSE_VERSION_PATCH@|$(VERSION_PATCH)|g' \
		-e 's|@SPARSE_VERSION_STRING@|$(VERSION)|g' \
		include/sparse_version.h.in > $(GENERATED_VERSION)

# Library objects
$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

# Static library
$(LIB): $(LIB_OBJS)
	ar rcs $@ $^

# Thread tests need -pthread.  When SPARSE_MUTEX is enabled (-DSPARSE_MUTEX),
# ALL compilation units (library and tests) must be compiled with -DSPARSE_MUTEX
# and linked with -pthread.
$(BUILDDIR)/test_threads: $(TESTDIR)/test_threads.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -I$(TESTDIR) -I$(SRCDIR) $< -L$(BUILDDIR) -lsparse_lu_ortho $(LDFLAGS) -lm -pthread -o $@

$(BUILDDIR)/test_sprint4_integration: $(TESTDIR)/test_sprint4_integration.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -I$(TESTDIR) -I$(SRCDIR) $< -L$(BUILDDIR) -lsparse_lu_ortho $(LDFLAGS) -lm -pthread -o $@

# Test executables (any .c in tests/)
$(BUILDDIR)/%: $(TESTDIR)/%.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -I$(TESTDIR) -I$(SRCDIR) $< -L$(BUILDDIR) -lsparse_lu_ortho $(LDFLAGS) -o $@

# Benchmark executables
$(BUILDDIR)/bench_%: $(BENCHDIR)/bench_%.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -I$(SRCDIR) $< -L$(BUILDDIR) -lsparse_lu_ortho $(LDFLAGS) -o $@

# Example executables
$(BUILDDIR)/example_%: $(EXDIR)/example_%.c $(LIB) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDE) $< -L$(BUILDDIR) -lsparse_lu_ortho $(LDFLAGS) -o $@

# Build all examples
.PHONY: examples
examples: $(EX_BINS)
	@echo "All examples built."

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

# Build and test with OpenMP-enabled SpMV
# On Linux with GCC: make omp
# On macOS with Apple Clang + Homebrew libomp: make omp
#   (auto-detects /usr/local/opt/libomp or /opt/homebrew/opt/libomp)
# On macOS with GCC: make omp CC=gcc-14
.PHONY: omp
ifeq ($(shell uname -s),Darwin)
# Apple Clang needs -Xpreprocessor -fopenmp and explicit libomp paths
LIBOMP_PREFIX := $(firstword $(wildcard /usr/local/opt/libomp /opt/homebrew/opt/libomp))
ifeq ($(LIBOMP_PREFIX),)
omp:
	@echo "error: libomp (OpenMP runtime) not found on this macOS system."
	@echo "Install it with 'brew install libomp' or set LIBOMP_PREFIX to the libomp prefix."
	@false
else
omp: CFLAGS += -DSPARSE_OPENMP -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
omp: LDFLAGS += -L$(LIBOMP_PREFIX)/lib -lomp
omp: clean test
endif
else
omp: CFLAGS += -DSPARSE_OPENMP -fopenmp
omp: LDFLAGS += -fopenmp
omp: clean test
endif

# Thread Sanitizer (for thread safety tests)
.PHONY: tsan
tsan: CFLAGS += -fsanitize=thread -fno-omit-frame-pointer -g -O1
tsan: LDFLAGS += -fsanitize=thread
tsan: clean test

# ─── Code quality targets ─────────────────────────────────────────────

# Source files for formatting/linting
ALL_SRC = $(shell find $(SRCDIR) -type f \( -name '*.c' -o -name '*.h' \))
ALL_TEST_SRC = $(shell find $(TESTDIR) -type f \( -name '*.c' -o -name '*.h' \))
ALL_BENCH_SRC = $(shell find $(BENCHDIR) -type f -name '*.c')
ALL_EX_SRC = $(wildcard $(EXDIR)/*.c)
ALL_HEADERS = $(shell find include -type f -name '*.h')

# Format all source files in-place
.PHONY: format
format:
	@echo "Formatting with clang-format..."
	clang-format -i $(ALL_SRC) $(ALL_TEST_SRC) $(ALL_BENCH_SRC) $(ALL_EX_SRC) $(ALL_HEADERS)

# Check formatting without modifying files
.PHONY: format-check
format-check:
	@echo "Checking formatting with clang-format..."
	clang-format --dry-run --Werror $(ALL_SRC) $(ALL_TEST_SRC) $(ALL_BENCH_SRC) $(ALL_EX_SRC) $(ALL_HEADERS)

# Run all linters
.PHONY: lint
lint: build/include/sparse_version.h
	@echo "Compiling with strict warnings (-Werror)..."
	$(CC) $(CFLAGS) -Wstrict-prototypes -Wformat=2 -Werror \
		$(INCLUDE) -fsyntax-only $(shell find $(SRCDIR) -type f -name '*.c')
	@echo ""
	@echo "Running clang-tidy..."
	clang-tidy $(shell find $(SRCDIR) -type f -name '*.c') -- $(INCLUDE) $(CFLAGS)
	@echo ""
	@echo "Running cppcheck..."
	cppcheck --enable=warning,style,performance,portability --error-exitcode=1 \
		--suppress=missingIncludeSystem --suppress=constVariablePointer \
		--suppress=constVariable --suppress=variableScope \
		--suppress=nullPointerOutOfMemory --suppress=uninitvar \
		--suppress=constParameterPointer --suppress=unreadVariable \
		-I include $(SRCDIR) $(TESTDIR)

# Run all quality checks: format + lint + test
.PHONY: check
check: format-check lint test

# ─── API documentation ────────────────────────────────────────────────

.PHONY: docs
docs:
	@echo "Generating API documentation with Doxygen..."
	doxygen Doxyfile
	@echo "Documentation generated in docs/api/html/"

# ─── Code coverage ────────────────────────────────────────────────────

# Build with gcov instrumentation, run tests, generate coverage report.
# Requires: gcc (real GCC, not Apple Clang shim), lcov, genhtml, bc.
# On Ubuntu:  apt install gcc lcov bc
# On macOS:   brew install gcc lcov && make coverage CC=gcc-14
# Apple Clang's gcov output is incompatible with lcov.
COVDIR = coverage
COV_THRESHOLD = 95

.PHONY: coverage
coverage: CFLAGS += --coverage -fprofile-arcs -ftest-coverage -g -O0
coverage: LDFLAGS += --coverage
coverage: clean $(TEST_BINS)
	@echo "Running tests for coverage..."
	@status=0; \
	for t in $(TEST_BINS); do \
		$$t || status=1; \
	done; \
	if [ $$status -ne 0 ]; then echo "Some tests failed"; exit 1; fi
	@echo ""
	@/bin/mkdir -p $(COVDIR)
	@echo "Collecting coverage data..."
	lcov --capture --directory $(BUILDDIR) --output-file $(COVDIR)/coverage.info \
		--ignore-errors mismatch,negative
	lcov --remove $(COVDIR)/coverage.info '*/tests/*' '*/benchmarks/*' \
		--output-file $(COVDIR)/coverage-src.info --ignore-errors unused
	@echo ""
	@echo "Generating HTML report..."
	genhtml $(COVDIR)/coverage-src.info --output-directory $(COVDIR)/html
	@echo ""
	@echo "Coverage report: $(COVDIR)/html/index.html"
	lcov --summary $(COVDIR)/coverage-src.info
	@echo ""
	@echo "Checking coverage threshold ($(COV_THRESHOLD)%)..."
	@pct=$$(lcov --summary $(COVDIR)/coverage-src.info 2>&1 \
		| awk '/lines.*:/ { gsub(/%/, "", $$2); print $$2; exit }'); \
	echo "Line coverage: $${pct}%"; \
	if [ -z "$$pct" ]; then \
		echo "FAIL: Could not parse coverage percentage"; \
		exit 1; \
	elif [ $$(echo "$$pct < $(COV_THRESHOLD)" | bc -l) -eq 1 ]; then \
		echo "FAIL: Line coverage $${pct}% is below $(COV_THRESHOLD)% threshold"; \
		exit 1; \
	else \
		echo "PASS: Line coverage $${pct}% meets $(COV_THRESHOLD)% threshold"; \
	fi

# ─── Installation ─────────────────────────────────────────────────────

PREFIX      ?= /usr/local
INSTALL_LIB  = $(DESTDIR)$(PREFIX)/lib
INSTALL_INC  = $(DESTDIR)$(PREFIX)/include/sparse
INSTALL_PC   = $(INSTALL_LIB)/pkgconfig
VERSION      = $(shell cat VERSION 2>/dev/null || echo "0.0.0")
VERSION_MAJOR = $(word 1,$(subst ., ,$(VERSION)))
VERSION_MINOR = $(word 2,$(subst ., ,$(VERSION)))
VERSION_PATCH = $(word 3,$(subst ., ,$(VERSION)))
HEADERS      = $(wildcard include/*.h)

# Extra pkg-config link flags based on build options
SPARSE_PC_LIBS_EXTRA =
ifdef SPARSE_MUTEX
SPARSE_PC_LIBS_EXTRA += -pthread
endif
ifdef SPARSE_OPENMP
ifeq ($(shell uname -s),Darwin)
SPARSE_PC_LIBS_EXTRA += -L$(LIBOMP_FLAG_PREFIX)/lib -lomp
else
SPARSE_PC_LIBS_EXTRA += -fopenmp
endif
endif

# Generate sparse_version.h from VERSION file and template
GENERATED_VERSION = $(BUILDDIR)/include/sparse_version.h

.PHONY: generate-version
generate-version: $(GENERATED_VERSION)

$(GENERATED_VERSION): VERSION include/sparse_version.h.in
	@mkdir -p $(BUILDDIR)/include
	@sed -e 's|@SPARSE_VERSION_MAJOR@|$(VERSION_MAJOR)|g' \
		-e 's|@SPARSE_VERSION_MINOR@|$(VERSION_MINOR)|g' \
		-e 's|@SPARSE_VERSION_PATCH@|$(VERSION_PATCH)|g' \
		-e 's|@SPARSE_VERSION_STRING@|$(VERSION)|g' \
		include/sparse_version.h.in > $(GENERATED_VERSION)
	@echo "Generated $(GENERATED_VERSION) ($(VERSION))"

# Library depends on generated version header
$(LIB): $(GENERATED_VERSION)

.PHONY: install
install: $(LIB)
	@echo "Installing sparse $(VERSION) to $(DESTDIR)$(PREFIX) ..."
	install -d $(INSTALL_LIB)
	install -d $(INSTALL_INC)
	install -d $(INSTALL_PC)
	install -m 644 $(LIB) $(INSTALL_LIB)/
	@for h in $(HEADERS); do \
		install -m 644 $$h $(INSTALL_INC)/; \
	done
	install -m 644 $(GENERATED_VERSION) $(INSTALL_INC)/
	@sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@VERSION@|$(VERSION)|g' \
		-e 's|@SPARSE_PC_LIBS_EXTRA@|$(SPARSE_PC_LIBS_EXTRA)|g' \
		sparse.pc.in > $(INSTALL_PC)/sparse.pc
	@echo "Installed:"
	@echo "  library  → $(INSTALL_LIB)/$(notdir $(LIB))"
	@echo "  headers  → $(INSTALL_INC)/"
	@echo "  pkg-config → $(INSTALL_PC)/sparse.pc"

.PHONY: uninstall
uninstall:
	@echo "Uninstalling sparse from $(DESTDIR)$(PREFIX) ..."
	rm -f  $(INSTALL_LIB)/$(notdir $(LIB))
	rm -rf $(INSTALL_INC)
	rm -f  $(INSTALL_PC)/sparse.pc
	-rmdir $(INSTALL_PC) 2>/dev/null || true
	@echo "Done."

# Clean
.PHONY: clean
clean:
	/bin/rm -rf $(BUILDDIR) $(COVDIR)
	/bin/rm -f *.gcda *.gcno *.gcov
