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
#   west0067      67x67,      294 nnz  - Chemical eng., unsymmetric
#   nos4         100x100,     594 nnz  - Structural, symmetric
#   bcsstk04     132x132,    3648 nnz  - Structural stiffness, SPD
#   steam1       240x240,    2248 nnz  - Thermal, unsymmetric
#   fs_541_1     541x541,    4282 nnz  - Chemical process, unsymmetric
#   orsirr_1    1030x1030,   6858 nnz  - Oil reservoir, unsymmetric
#   bcsstk14    1806x1806,  63454 nnz  - Structural stiffness, SPD
#   s3rmt3m3    5357x5357, 207123 nnz  - Cylindrical shell, SPD
#   Kuu         7102x7102, 340200 nnz  - Finite-element stiffness, SPD
#   Pres_Poisson 14822x14822, 715804 nnz - Pressure Poisson, SPD
#   bloweybq    10001x10001, 39996 nnz  - Materials optimisation, symmetric indefinite
#   tuma1       22967x22967, 87760 nnz  - Mine model, symmetric indefinite

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${PROJECT_DIR}/tests/data/suitesparse"
BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

mkdir -p "$DEST"

# Each entry: "<collection>/<name>".
MATRICES=(
    "HB/west0067"
    "HB/nos4"
    "HB/bcsstk04"
    "HB/steam1"
    "HB/fs_541_1"
    "HB/orsirr_1"
    "HB/bcsstk14"
    "Cylshell/s3rmt3m3"
    "MathWorks/Kuu"
    "ACUSIM/Pres_Poisson"
    "GHS_indef/bloweybq"
    "GHS_indef/tuma1"
)

TMPDIR_BASE=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BASE"' EXIT

for entry in "${MATRICES[@]}"; do
    collection="${entry%%/*}"
    name="${entry##*/}"
    mtx="${DEST}/${name}.mtx"
    if [ -f "$mtx" ]; then
        echo "  skip: ${name}.mtx (already exists)"
        continue
    fi

    echo "  fetch: ${collection}/${name} ..."
    curl -sL -o "${TMPDIR_BASE}/${name}.tar.gz" "${BASE_URL}/${collection}/${name}.tar.gz"

    # Validate download (gzip -t works in minimal environments without 'file')
    if ! gzip -t "${TMPDIR_BASE}/${name}.tar.gz" 2>/dev/null; then
        echo "  ERROR: ${name} download failed (not a valid gzip file)"
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
