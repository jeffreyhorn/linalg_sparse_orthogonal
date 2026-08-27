#!/usr/bin/env bash
# check_qr_header_docs_guard.sh - QR header/docs drift guard.
#
# Keeps the Sprint 184 QR header sections, selected declarations, and
# QR-facing docs alignment in place without making ABI, package, platform,
# performance, or broad external-parity claims.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HEADER="$ROOT_DIR/include/sparse_qr.h"
README="$ROOT_DIR/README.md"
API_REFERENCE="$ROOT_DIR/docs/api_reference.md"
COOKBOOK="$ROOT_DIR/docs/cookbook.md"
SOLVER_SELECTION="$ROOT_DIR/docs/solver_selection.md"
TUTORIAL="$ROOT_DIR/docs/tutorial.md"
EXAMPLES_README="$ROOT_DIR/examples/README.md"
TMP_FILES=()

cleanup() {
    local file

    for file in "${TMP_FILES[@]:-}"; do
        rm -f "$file"
    done
}
trap cleanup EXIT

fail() {
    echo "qr-header-docs-guard: FAIL: $1" >&2
    exit 1
}

pass() {
    echo "qr-header-docs-guard: $1 ok"
}

require_fixed() {
    local needle="$1"
    local file="$2"
    local message="$3"

    if ! grep -Fq "$needle" "$file"; then
        fail "$message"
    fi
}

require_regex_absent() {
    local pattern="$1"
    local file="$2"
    local message="$3"
    local matches

    matches="$(grep -En "$pattern" "$file" 2>/dev/null || true)"
    if [ -n "$matches" ]; then
        echo "$matches" >&2
        fail "$message"
    fi
}

extract_tutorial_qr_section() {
    awk '
        /^### QR Factorization$/ { in_section = 1 }
        in_section { print }
        in_section && /^---$/ { exit }
    ' "$TUTORIAL"
}

check_header_sections() {
    require_fixed '/* Options and factor object */' "$HEADER" \
        "missing QR options/factor object section heading"
    require_fixed '/* Factorization and lifecycle */' "$HEADER" \
        "missing QR factorization/lifecycle section heading"
    require_fixed '/* Q operations */' "$HEADER" "missing QR Q operations section heading"
    require_fixed '/* Solve operations */' "$HEADER" "missing QR solve operations section heading"
    require_fixed '/* Rank, nullspace, and diagnostics */' "$HEADER" \
        "missing QR rank/nullspace/diagnostics section heading"

    pass "header sections"
}

check_header_declarations() {
    require_fixed '} sparse_qr_opts_t;' "$HEADER" "missing sparse_qr_opts_t declaration"
    require_fixed '} sparse_qr_t;' "$HEADER" "missing sparse_qr_t declaration"
    require_fixed 'sparse_qr_factor(' "$HEADER" "missing sparse_qr_factor declaration"
    require_fixed 'sparse_qr_factor_opts(' "$HEADER" "missing sparse_qr_factor_opts declaration"
    require_fixed 'sparse_qr_free(' "$HEADER" "missing sparse_qr_free declaration"
    require_fixed 'sparse_qr_apply_q(' "$HEADER" "missing sparse_qr_apply_q declaration"
    require_fixed 'sparse_qr_form_q(' "$HEADER" "missing sparse_qr_form_q declaration"
    require_fixed 'sparse_qr_solve(' "$HEADER" "missing sparse_qr_solve declaration"
    require_fixed 'sparse_qr_refine(' "$HEADER" "missing sparse_qr_refine declaration"
    require_fixed 'sparse_qr_solve_minnorm(' "$HEADER" \
        "missing sparse_qr_solve_minnorm declaration"
    require_fixed 'sparse_qr_refine_minnorm(' "$HEADER" \
        "missing sparse_qr_refine_minnorm declaration"
    require_fixed 'sparse_qr_rank(' "$HEADER" "missing sparse_qr_rank declaration"
    require_fixed 'sparse_qr_nullspace(' "$HEADER" "missing sparse_qr_nullspace declaration"
    require_fixed 'sparse_qr_diag_r(' "$HEADER" "missing sparse_qr_diag_r declaration"
    require_fixed '} sparse_qr_rank_info_t;' "$HEADER" "missing sparse_qr_rank_info_t declaration"
    require_fixed 'sparse_qr_rank_info(' "$HEADER" "missing sparse_qr_rank_info declaration"
    require_fixed 'sparse_qr_condest(' "$HEADER" "missing sparse_qr_condest declaration"

    pass "header declarations"
}

check_header_claim_boundaries() {
    local claim_pattern

    claim_pattern='raw QR basis parity|global rank-threshold policy|broad rank-deficient solve|broad minimum-norm behavior|external-library parity|SuiteSparse parity|LAPACK parity|NumPy parity|SciPy parity|Windows report freshness|package/ABI|package support|ABI support|portable performance|performance guarantee|state-of-the-art'
    require_regex_absent "$claim_pattern" "$HEADER" \
        "QR header gained unsupported parity/platform/package/performance claim wording"

    pass "header unsupported claim absence"
}

check_docs_alignment() {
    local qr_section_file

    require_fixed 'COLAMD column reordering for unsymmetric/QR workflows' "$README" \
        "README QR factor_opts bullet must name COLAMD for unsymmetric/QR workflows"
    require_fixed 'QR factorization/lifecycle, least-squares, minimum-norm, rank/nullspace, R-diagonal diagnostics, and cancellation contracts' "$API_REFERENCE" \
        "API reference QR row must include lifecycle, diagnostics, and cancellation scope"
    require_fixed 'selected QR minimum-norm and' "$COOKBOOK" \
        "cookbook QR evidence note must include selected minimum-norm comparison scope"
    require_fixed 'compatible least-squares rows from' "$COOKBOOK" \
        "cookbook QR evidence note must include selected compatible least-squares scope"
    require_fixed 'minimum-norm output, and R-diagonal diagnostics' "$SOLVER_SELECTION" \
        "solver-selection QR diagnostics row must include minimum-norm and R-diagonal diagnostics"
    require_fixed 'broad minimum-norm behavior' "$SOLVER_SELECTION" \
        "solver-selection QR evidence boundary must reject broad minimum-norm behavior"
    require_fixed 'selected QR' "$SOLVER_SELECTION" \
        "solver-selection QR evidence boundary must name selected QR comparison scope"
    require_fixed 'minimum-norm and compatible least-squares comparison rows named in' \
        "$SOLVER_SELECTION" \
        "solver-selection QR evidence boundary must name minimum-norm and compatible least-squares rows"
    require_fixed 'temporary QR factorizations built' \
        "$EXAMPLES_README" \
        "examples README minimum-norm note must mention internal QR factorizations"

    qr_section_file="$(mktemp "${TMPDIR:-/tmp}/qr_tutorial_section.XXXXXX")"
    TMP_FILES+=("$qr_section_file")
    extract_tutorial_qr_section >"$qr_section_file"
    require_fixed '### QR Factorization' "$qr_section_file" \
        "could not locate QR tutorial section for scoped checks"
    require_fixed 'sparse_err_t err = sparse_qr_factor(A, &qr);' "$qr_section_file" \
        "tutorial QR snippet must check sparse_qr_factor return status"
    require_fixed 'err = sparse_qr_solve(&qr, b, x, &residual_norm);' "$qr_section_file" \
        "tutorial QR snippet must check sparse_qr_solve return status"
    require_fixed 'x and residual_norm are caller-owned outputs' "$qr_section_file" \
        "tutorial QR snippet must name caller-owned solve outputs"
    require_fixed 'Releases factor data stored inside the caller-owned qr object' "$qr_section_file" \
        "tutorial QR snippet must describe sparse_qr_free lifecycle"

    pass "docs alignment"
}

check_header_sections
check_header_declarations
check_header_claim_boundaries
check_docs_alignment

echo "qr-header-docs-guard: passed"
