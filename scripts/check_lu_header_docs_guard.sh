#!/usr/bin/env bash
# check_lu_header_docs_guard.sh - LU header/docs drift guard.
#
# Keeps the Sprint 172 LU header section headings and tutorial refinement
# signature aligned without making ABI, package, or platform claims.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HEADER="$ROOT_DIR/include/sparse_lu.h"
TUTORIAL="$ROOT_DIR/docs/tutorial.md"
TMP_FILES=()

cleanup() {
    local file

    for file in "${TMP_FILES[@]}"; do
        rm -f "$file"
    done
}
trap cleanup EXIT

fail() {
    echo "lu-header-docs-guard: FAIL: $1" >&2
    exit 1
}

pass() {
    echo "lu-header-docs-guard: $1 ok"
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

extract_tutorial_lu_section() {
    awk '
        /^### LU Factorization$/ { in_section = 1 }
        in_section { print }
        in_section && /^### Cholesky Factorization$/ { exit }
    ' "$TUTORIAL"
}

check_header_sections() {
    require_fixed '/* Options */' "$HEADER" "missing LU Options section heading"
    require_fixed '/* Factorization */' "$HEADER" "missing LU Factorization section heading"
    require_fixed '/* Solves */' "$HEADER" "missing LU Solves section heading"
    require_fixed '/* Conditioning and transpose solves */' "$HEADER" \
        "missing LU conditioning/transpose section heading"
    require_fixed '/* Advanced solver phases */' "$HEADER" \
        "missing LU advanced solver phases section heading"
    require_fixed '/* Refinement */' "$HEADER" "missing LU Refinement section heading"

    pass "header sections"
}

check_header_declarations() {
    require_fixed 'sparse_lu_factor_opts(' "$HEADER" "missing sparse_lu_factor_opts declaration"
    require_fixed 'sparse_lu_factor(' "$HEADER" "missing sparse_lu_factor declaration"
    require_fixed 'sparse_lu_solve(' "$HEADER" "missing sparse_lu_solve declaration"
    require_fixed 'sparse_lu_solve_block(' "$HEADER" "missing sparse_lu_solve_block declaration"
    require_fixed 'sparse_lu_condest(' "$HEADER" "missing sparse_lu_condest declaration"
    require_fixed 'sparse_lu_solve_transpose(' "$HEADER" \
        "missing sparse_lu_solve_transpose declaration"
    require_fixed 'sparse_apply_row_perm(' "$HEADER" "missing sparse_apply_row_perm declaration"
    require_fixed 'sparse_apply_inv_col_perm(' "$HEADER" \
        "missing sparse_apply_inv_col_perm declaration"
    require_fixed 'sparse_forward_sub(' "$HEADER" "missing sparse_forward_sub declaration"
    require_fixed 'sparse_backward_sub(' "$HEADER" "missing sparse_backward_sub declaration"
    require_fixed 'sparse_lu_refine(' "$HEADER" "missing sparse_lu_refine declaration"

    pass "header declarations"
}

check_tutorial_refinement_signature() {
    require_fixed 'sparse_lu_refine(A, LU, b, x, 3, 1e-15);' "$TUTORIAL" \
        "tutorial LU refinement snippet must use six-argument signature"
    require_regex_absent 'sparse_lu_refine\(A, LU, b, x, 3\);' "$TUTORIAL" \
        "tutorial LU refinement snippet regressed to five-argument signature"

    pass "tutorial refinement signature"
}

check_unsupported_claim_absence() {
    local claim_pattern
    local lu_section_file

    claim_pattern='package-manager support|shared-library support|dynamic ABI|runtime-loader|broad Windows parity|Windows Makefile parity|Windows pkg-config parity|external-library parity|portable performance|performance guarantee|LU CSR parity|state-of-the-art'
    require_regex_absent "$claim_pattern" "$HEADER" \
        "LU header gained unsupported package/ABI/platform/performance claim wording"

    lu_section_file="$(mktemp "${TMPDIR:-/tmp}/lu_tutorial_section.XXXXXX")"
    TMP_FILES+=("$lu_section_file")
    extract_tutorial_lu_section >"$lu_section_file"
    require_fixed '### LU Factorization' "$lu_section_file" \
        "could not locate LU tutorial section for scoped claim scan"
    require_regex_absent "$claim_pattern" "$lu_section_file" \
        "tutorial LU section gained unsupported package/ABI/platform/performance claim wording"

    pass "unsupported claim absence"
}

check_header_sections
check_header_declarations
check_tutorial_refinement_signature
check_unsupported_claim_absence

echo "lu-header-docs-guard: passed"
