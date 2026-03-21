#!/bin/bash
#
# CI script for sparse_lu_orthogonal
#
# Usage: ./scripts/ci.sh [--bench] [--sanitize]
#
# Exits with non-zero status if any step fails.
#

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

BENCH=0
SANITIZE=0
for arg in "$@"; do
    case "$arg" in
        --bench)    BENCH=1 ;;
        --sanitize) SANITIZE=1 ;;
        *)          echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "========================================"
echo "  sparse_lu_orthogonal CI"
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

# Step 4: Benchmarks (optional, informational only)
if [ "$BENCH" -eq 1 ]; then
    echo "--- Step 4: Benchmarks (informational) ---"
    make bench
    echo
fi

echo "========================================"
echo "  CI PASSED"
echo "========================================"
