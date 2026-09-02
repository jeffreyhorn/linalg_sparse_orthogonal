#!/usr/bin/env bash
# check_qr_external_ref_helper_guard.sh - Sprint 193 QR helper guard.
#
# Keeps the selected external-reference rank/nullspace/threshold helpers tied
# to the registered `test_qr` proof-owner binary. The extracted header is an
# included helper only; it must not become a standalone test or library source.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_FILE="$ROOT_DIR/tests/test_qr.c"
HELPER="tests/test_qr_external_ref_helpers.h"
HELPER_PATH="$ROOT_DIR/$HELPER"
HELPER_NAME="$(basename "$HELPER")"
MAKEFILE="$ROOT_DIR/Makefile"
CMAKE_FILE="$ROOT_DIR/CMakeLists.txt"
LIBRARY_MANIFEST="$ROOT_DIR/build-metadata/library_sources.txt"
MAINTAINER_GUIDE="$ROOT_DIR/docs/maintainer_guide.md"

MOVED_DEFINITION_MARKERS=(
    "static int read_qr_basis_external_reference"
    "static int read_qr_threshold_external_reference"
    "static void test_qr_external_reference_readers_reject_invalid_arguments(void) {"
    "static void test_qr_external_reference_readers_reject_unsupported_fixtures(void) {"
    "static void test_qr_external_dense_reference_rank1_4x3_nullspace_projector(void) {"
    "static void test_qr_external_dense_reference_rankdef_duplicate_5x4_nullspace_projector(void) {"
    "static void test_qr_external_dense_reference_rankdef_dependent_row_4x3_nullspace_projector(void) {"
    "static void test_qr_external_dense_reference_rankdef_wide_3x5_nullspace_subspace(void) {"
    "static void test_qr_external_dense_reference_rank_threshold_diag4_family(void) {"
    "static void test_qr_external_dense_reference_rank_threshold_diag4_scaled_family(void) {"
    "static void test_qr_external_dense_reference_rank_threshold_duplicate_5x4_perturbed_family(void) {"
    "test_qr_external_dense_reference_rank_threshold_dependent_row_4x3_perturbed_family(void) {"
)

RUN_TEST_MARKERS=(
    "RUN_TEST(test_qr_external_reference_readers_reject_invalid_arguments);"
    "RUN_TEST(test_qr_external_reference_readers_reject_unsupported_fixtures);"
    "RUN_TEST(test_qr_external_dense_reference_rank1_4x3_nullspace_projector);"
    "RUN_TEST(test_qr_external_dense_reference_rankdef_duplicate_5x4_nullspace_projector);"
    "RUN_TEST(test_qr_external_dense_reference_rankdef_dependent_row_4x3_nullspace_projector);"
    "RUN_TEST(test_qr_external_dense_reference_rankdef_wide_3x5_nullspace_subspace);"
    "RUN_TEST(test_qr_external_dense_reference_rank_threshold_diag4_family);"
    "RUN_TEST(test_qr_external_dense_reference_rank_threshold_diag4_scaled_family);"
    "RUN_TEST(test_qr_external_dense_reference_rank_threshold_duplicate_5x4_perturbed_family);"
    "RUN_TEST(test_qr_external_dense_reference_rank_threshold_dependent_row_4x3_perturbed_family);"
)

fail() {
    echo "qr-external-ref-helper-guard: FAIL: $1" >&2
    exit 1
}

pass() {
    echo "qr-external-ref-helper-guard: $1 ok"
}

require_file() {
    local path="$1"
    local message="$2"

    if [ ! -f "$path" ]; then
        fail "$message"
    fi
}

require_fixed() {
    local needle="$1"
    local file="$2"
    local message="$3"

    if ! grep -Fq "$needle" "$file"; then
        fail "$message"
    fi
}

require_absent_fixed() {
    local needle="$1"
    local file="$2"
    local message="$3"
    local matches

    matches="$(grep --fixed-strings --line-number -- "$needle" "$file" 2>/dev/null || true)"
    if [ -n "$matches" ]; then
        echo "$matches" >&2
        fail "$message"
    fi
}

require_exact_fixed_count() {
    local needle="$1"
    local file="$2"
    local expected="$3"
    local message="$4"
    local count

    count="$(grep --fixed-strings --count -- "$needle" "$file" 2>/dev/null || true)"
    if [ -z "$count" ]; then
        count=0
    fi
    if [ "$count" -ne "$expected" ]; then
        fail "$message (expected $expected, found $count)"
    fi
}

check_required_files() {
    require_file "$TEST_FILE" "tests/test_qr.c is missing"
    require_file "$HELPER_PATH" "$HELPER is missing"
    require_file "$MAKEFILE" "Makefile is missing"
    require_file "$CMAKE_FILE" "CMakeLists.txt is missing"
    require_file "$LIBRARY_MANIFEST" "build-metadata/library_sources.txt is missing"
    require_file "$MAINTAINER_GUIDE" "docs/maintainer_guide.md is missing"

    pass "required files"
}

check_proof_owner_registration() {
    require_fixed '$(TESTDIR)/test_qr.c' "$MAKEFILE" \
        "Makefile no longer registers test_qr.c in TEST_SRCS"
    require_fixed 'add_sparse_test(test_qr)' "$CMAKE_FILE" \
        "CMakeLists.txt no longer registers test_qr"

    pass "proof-owner registration"
}

check_helper_boundary() {
    require_fixed "#ifndef TEST_QR_EXTERNAL_REF_HELPERS_H" "$HELPER_PATH" \
        "$HELPER is missing include guard TEST_QR_EXTERNAL_REF_HELPERS_H"
    require_fixed "#define TEST_QR_EXTERNAL_REF_HELPERS_H" "$HELPER_PATH" \
        "$HELPER is missing include guard define TEST_QR_EXTERNAL_REF_HELPERS_H"
    require_exact_fixed_count "#include \"$HELPER_NAME\"" "$TEST_FILE" 1 \
        "tests/test_qr.c must include $HELPER_NAME exactly once"

    pass "helper boundary"
}

check_selected_cluster_ownership() {
    local marker

    for marker in "${MOVED_DEFINITION_MARKERS[@]}"; do
        require_fixed "$marker" "$HELPER_PATH" \
            "$HELPER is missing moved selected-cluster definition marker '$marker'"
        require_absent_fixed "$marker" "$TEST_FILE" \
            "tests/test_qr.c still owns moved selected-cluster definition marker '$marker'"
    done

    for marker in "${RUN_TEST_MARKERS[@]}"; do
        require_exact_fixed_count "$marker" "$TEST_FILE" 1 \
            "tests/test_qr.c must retain proof-owner registration '$marker' exactly once"
    done

    require_fixed "static void test_qr_external_dense_reference_economy_projector_5x3(void) {" \
        "$TEST_FILE" \
        "tests/test_qr.c must retain the economy external-reference proof-owner body"
    require_absent_fixed \
        "static void test_qr_external_dense_reference_economy_projector_5x3(void) {" \
        "$HELPER_PATH" \
        "$HELPER must not absorb the economy external-reference proof-owner body"

    pass "selected cluster ownership"
}

check_header_only_registration() {
    local stem

    stem="${HELPER_NAME%.h}"
    require_absent_fixed "$HELPER_NAME" "$MAKEFILE" \
        "$HELPER_NAME must remain header-only and not be named in Makefile registration"
    require_absent_fixed "$HELPER_NAME" "$CMAKE_FILE" \
        "$HELPER_NAME must remain header-only and not be named in CMake registration"
    require_absent_fixed "$HELPER" "$LIBRARY_MANIFEST" \
        "$HELPER must not be listed as a library source"
    require_absent_fixed "add_sparse_test($stem)" "$CMAKE_FILE" \
        "$stem must not become a separate CMake test without a new proof-owner decision"

    pass "header-only registration"
}

check_maintainer_docs() {
    require_fixed "Sprint 193 QR external-reference helper boundary" "$MAINTAINER_GUIDE" \
        "docs/maintainer_guide.md is missing the Sprint 193 QR helper boundary marker"
    require_fixed "\`tests/test_qr_external_ref_helpers.h\` owns the selected QR" \
        "$MAINTAINER_GUIDE" \
        "docs/maintainer_guide.md is missing the QR helper owner marker"
    require_fixed "\`tests/test_qr.c\` remains the registered QR proof-owner binary" \
        "$MAINTAINER_GUIDE" \
        "docs/maintainer_guide.md is missing the QR proof-owner marker"
    require_fixed "\`make qr-external-ref-helper-guard\`" "$MAINTAINER_GUIDE" \
        "docs/maintainer_guide.md is missing the QR helper guard command"
    require_fixed "no-behavior-change review-surface reduction" "$MAINTAINER_GUIDE" \
        "docs/maintainer_guide.md is missing the QR no-behavior-change marker"

    pass "maintainer docs"
}

check_required_files
check_proof_owner_registration
check_helper_boundary
check_selected_cluster_ownership
check_header_only_registration
check_maintainer_docs

echo "qr-external-ref-helper-guard: passed"
