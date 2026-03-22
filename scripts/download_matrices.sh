#!/bin/bash
#
# Download SuiteSparse reference matrices for testing and benchmarking.
#
# Usage: ./scripts/download_matrices.sh
#
# Downloads matrices to tests/data/suitesparse/. Idempotent — skips
# matrices that already exist.
#
# Matrix selection:
#   west0067   67x67,   294 nnz  - Chemical eng., unsymmetric
#   nos4      100x100,  594 nnz  - Structural, symmetric
#   bcsstk04  132x132, 3648 nnz  - Structural stiffness, symmetric
#   steam1    240x240, 2248 nnz  - Thermal, unsymmetric
#   fs_541_1  541x541, 4282 nnz  - Chemical process, unsymmetric
#   orsirr_1 1030x1030,6858 nnz  - Oil reservoir, unsymmetric

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${PROJECT_DIR}/tests/data/suitesparse"
BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM/HB"

mkdir -p "$DEST"

MATRICES="west0067 nos4 bcsstk04 steam1 fs_541_1 orsirr_1"

TMPDIR_BASE=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BASE"' EXIT

for name in $MATRICES; do
    mtx="${DEST}/${name}.mtx"
    if [ -f "$mtx" ]; then
        echo "  skip: ${name}.mtx (already exists)"
        continue
    fi

    echo "  fetch: ${name} ..."
    curl -sL -o "${TMPDIR_BASE}/${name}.tar.gz" "${BASE_URL}/${name}.tar.gz"

    # Validate download
    if ! file "${TMPDIR_BASE}/${name}.tar.gz" | grep -q "gzip"; then
        echo "  ERROR: ${name} download failed (not a gzip file)"
        continue
    fi

    tar xzf "${TMPDIR_BASE}/${name}.tar.gz" -C "$TMPDIR_BASE"

    if [ -f "${TMPDIR_BASE}/${name}/${name}.mtx" ]; then
        cp "${TMPDIR_BASE}/${name}/${name}.mtx" "$mtx"
        echo "  done: ${name}.mtx"
    else
        echo "  ERROR: ${name}.mtx not found in archive"
    fi
done

echo
echo "SuiteSparse matrices in ${DEST}:"
for f in "${DEST}"/*.mtx; do
    [ -f "$f" ] || continue
    dims=$(grep -v "^%" "$f" | head -1 || true)
    echo "  $(basename "$f"): ${dims}"
done
