#!/bin/bash
#
# CI script for linalg_sparse_orthogonal
#
# Usage: ./scripts/ci.sh [--bench] [--sanitize] [--asan]
#
# Exits with non-zero status if any step fails.
#

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

BENCH=0
SANITIZE=0
ASAN=0
for arg in "$@"; do
    case "$arg" in
        --bench)    BENCH=1 ;;
        --sanitize) SANITIZE=1 ;;
        --asan)     ASAN=1 ;;
        *)          echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "========================================"
echo "  linalg_sparse_orthogonal CI"
echo "========================================"
echo

# Step 1: Clean build
echo "--- Step 1: Clean build ---"
make clean
make all
echo "Library built successfully."
echo

# Step 2: Build and run all tests
echo "--- Step 2: Running tests ---"
make test
echo

# Step 3: Sanitizer build (optional)
if [ "$SANITIZE" -eq 1 ]; then
    echo "--- Step 3: Sanitizer build (UBSan) ---"
    make sanitize
    echo
fi

# Step 3b: ASan build (optional)
# NOTE: Apple Clang ASan hangs on macOS. This step requires GCC or LLVM clang.
# On Linux with default compiler, or macOS with: CC=gcc-14 ./scripts/ci.sh --asan
if [ "$ASAN" -eq 1 ]; then
    echo "--- Step 3b: ASan build ---"
    if [ "$(uname)" = "Darwin" ]; then
        # Check if we're using Apple Clang (which hangs with ASan)
        if ${CC:-cc} --version 2>&1 | grep -q "Apple clang"; then
            echo "WARNING: Apple Clang ASan is known to hang on macOS."
            echo "  Skipping ASan. Use GCC or LLVM clang:"
            echo "    CC=gcc-14 ./scripts/ci.sh --asan"
            echo "    CC=/opt/homebrew/opt/llvm/bin/clang ./scripts/ci.sh --asan"
        else
            make asan
        fi
    else
        make asan
    fi
    echo
fi

# Step 4: Benchmarks (optional, informational only)
if [ "$BENCH" -eq 1 ]; then
    echo "--- Step 4: Benchmarks (informational) ---"
    make bench
    echo
fi

echo "========================================"
echo "  CI PASSED"
echo "========================================"
