#!/usr/bin/env bash
# wall_check.sh — Sprint 24 Day 1 performance regression gate.
#
# Usage: wall_check.sh <bench_amd_qg> <bench_reorder> <baseline_file>
#
# Runs `bench_amd_qg --only bcsstk14` and
# `bench_reorder --only Pres_Poisson --skip-factor`, extracts the
# `reorder_ms` column for the qg-AMD / AMD rows, compares each against
# the threshold in <baseline_file>.  Exits 0 if both stay within 2× the
# baseline; non-zero otherwise.
#
# The baseline file format is two lines of `KEY=VALUE_MS`:
#
#   bcsstk14_qg_amd_ms=...
#   pres_poisson_amd_ms=...
#
# Lines starting with `#` are ignored (comments).

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "wall-check: usage: $0 <bench_amd_qg> <bench_reorder> <baseline_file>" >&2
    exit 2
fi

BENCH_AMD_QG="$1"
BENCH_REORDER="$2"
BASELINE_FILE="$3"

if [ ! -x "$BENCH_AMD_QG" ]; then
    echo "wall-check: $BENCH_AMD_QG not executable" >&2
    exit 2
fi
if [ ! -x "$BENCH_REORDER" ]; then
    echo "wall-check: $BENCH_REORDER not executable" >&2
    exit 2
fi
if [ ! -r "$BASELINE_FILE" ]; then
    echo "wall-check: $BASELINE_FILE not readable" >&2
    exit 2
fi

# Parse baseline file; skip comments + blank lines.
BCSSTK14_QG_BASE=$(awk -F= '/^bcsstk14_qg_amd_ms=/ {print $2}' "$BASELINE_FILE")
PRES_POISSON_BASE=$(awk -F= '/^pres_poisson_amd_ms=/ {print $2}' "$BASELINE_FILE")

if [ -z "$BCSSTK14_QG_BASE" ] || [ -z "$PRES_POISSON_BASE" ]; then
    echo "wall-check: baseline file missing required keys (bcsstk14_qg_amd_ms, pres_poisson_amd_ms)" >&2
    exit 2
fi

# Run bench_amd_qg --only bcsstk14, extract qg row's reorder_ms (col 4).
# CSV header: matrix,n,impl,reorder_ms,peak_rss_mb,nnz_L
#
# `--skip-bitset` keeps the run to qg-only (the Sprint 22 bitset runs
# minutes on Pres_Poisson and we don't need it for the gate).  Capture
# the full output to a temp file before parsing — `awk ... exit` on a
# pipe closes stdin early and SIGPIPEs the upstream `bench_amd_qg`,
# producing exit 141 from the pipeline.
# Initialize both temp file vars before the EXIT trap installs.  Under
# `set -u`, the trap's `$TMP_REORDER` expansion would be an unbound-
# variable error if `bench_amd_qg` failed before TMP_REORDER got
# assigned below — that would mask the real failure exit code.
TMP_AMD_QG=$(mktemp -t wall_check_amd_qg.XXXXXX)
TMP_REORDER=$(mktemp -t wall_check_reorder.XXXXXX)
trap 'rm -f "$TMP_AMD_QG" "$TMP_REORDER" 2>/dev/null' EXIT

if ! "$BENCH_AMD_QG" --only bcsstk14 --skip-bitset > "$TMP_AMD_QG" 2>/dev/null; then
    echo "wall-check: bench_amd_qg --only bcsstk14 failed" >&2
    exit 2
fi

# Match by full CSV field equality rather than regex prefix:
# `^bcsstk14,.*,qg,` would also match `^bcsstk14_other,.*,qg,` if a
# future fixture name shared the prefix.  $1=="bcsstk14" && $3=="qg"
# is unambiguous regardless of row ordering / future fixture additions.
BCSSTK14_QG_ACTUAL=$(awk -F, '$1=="bcsstk14" && $3=="qg" {print $4; exit}' "$TMP_AMD_QG")

if [ -z "$BCSSTK14_QG_ACTUAL" ]; then
    echo "wall-check: could not parse bcsstk14 qg-AMD reorder_ms from bench_amd_qg output" >&2
    exit 2
fi

# Run bench_reorder --only Pres_Poisson --skip-factor, extract AMD row.
# CSV header: matrix,n,reorder,nnz_L,reorder_ms,factor_ms

if ! "$BENCH_REORDER" --only Pres_Poisson --skip-factor > "$TMP_REORDER" 2>/dev/null; then
    echo "wall-check: bench_reorder --only Pres_Poisson failed" >&2
    exit 2
fi

# Match by full CSV field equality.  `^Pres_Poisson,.*,AMD,` would also
# match the `COLAMD` row (regex `AMD` is a suffix of `COLAMD`), and the
# current code only worked because the AMD row appears before COLAMD in
# bench_reorder's emit order.  $1=="Pres_Poisson" && $3=="AMD" is
# robust to row ordering and future ordering additions.
PRES_POISSON_ACTUAL=$(awk -F, '$1=="Pres_Poisson" && $3=="AMD" {print $5; exit}' "$TMP_REORDER")

if [ -z "$PRES_POISSON_ACTUAL" ]; then
    echo "wall-check: could not parse Pres_Poisson AMD reorder_ms from bench_reorder output" >&2
    exit 2
fi

# Compare actual vs 2× baseline.  awk for floating-point comparison.
echo "wall-check: bcsstk14    qg-AMD = $BCSSTK14_QG_ACTUAL ms (baseline $BCSSTK14_QG_BASE ms)"
echo "wall-check: Pres_Poisson AMD  = $PRES_POISSON_ACTUAL ms (baseline $PRES_POISSON_BASE ms)"

FAIL=0
if awk -v a="$BCSSTK14_QG_ACTUAL" -v b="$BCSSTK14_QG_BASE" \
    'BEGIN { exit !(a > 2 * b) }'; then
    echo "wall-check: FAIL bcsstk14 qg-AMD ($BCSSTK14_QG_ACTUAL ms) > 2× baseline ($BCSSTK14_QG_BASE ms)" >&2
    FAIL=1
fi
if awk -v a="$PRES_POISSON_ACTUAL" -v b="$PRES_POISSON_BASE" \
    'BEGIN { exit !(a > 2 * b) }'; then
    echo "wall-check: FAIL Pres_Poisson AMD ($PRES_POISSON_ACTUAL ms) > 2× baseline ($PRES_POISSON_BASE ms)" >&2
    FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
echo "wall-check: PASS"
exit 0
