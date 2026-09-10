#!/usr/bin/env bash
# check_svd_helper_guard.sh - Sprint 201 SVD helper registration guard.
#
# Keeps the selected rank/pseudoinverse/low-rank helper cluster tied to the
# registered `test_svd` proof-owner binary. The helper header is intentionally
# included by `tests/test_svd.c`, not registered as a standalone test or
# library source.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_FILE="$ROOT_DIR/tests/test_svd.c"
SHARED_HELPER="tests/test_svd_helpers.h"
SHARED_HELPER_PATH="$ROOT_DIR/$SHARED_HELPER"
SHARED_HELPER_NAME="$(basename "$SHARED_HELPER")"
SELECTED_HELPER="tests/test_svd_selected_helpers.h"
SELECTED_HELPER_PATH="$ROOT_DIR/$SELECTED_HELPER"
SELECTED_HELPER_NAME="$(basename "$SELECTED_HELPER")"
MAKEFILE="$ROOT_DIR/Makefile"
CMAKE_FILE="$ROOT_DIR/CMakeLists.txt"
LIBRARY_MANIFEST="$ROOT_DIR/build-metadata/library_sources.txt"

MOVED_DEFINITION_MARKERS=(
    "static inline void tf_svd_test_rank_full(void) {"
    "static inline void tf_svd_test_rank_deficient(void) {"
    "static inline void tf_svd_test_rank_nearly_singular(void) {"
    "static inline void tf_svd_test_rank_diagonal_threshold_fixture(void) {"
    "static inline void tf_svd_test_qr_rank_dependent_row_fixture(void) {"
    "static inline void tf_svd_test_rank_null(void) {"
    "static inline void tf_svd_test_pinv_diagonal(void) {"
    "static inline void tf_svd_test_pinv_moore_penrose(void) {"
    "static inline void tf_svd_test_pinv_null(void) {"
    "static inline void tf_svd_test_pinv_rectangular(void) {"
    "static inline void tf_svd_test_pinv_underdetermined_minnorm_solution(void) {"
    "static inline void tf_svd_test_lowrank_diagonal(void) {"
    "static inline void tf_svd_test_lowrank_error_bound(void) {"
    "static inline void tf_svd_test_lowrank_errors(void) {"
)

RUN_TEST_MARKERS=(
    "RUN_TEST(test_svd_rank_full);"
    "RUN_TEST(test_svd_rank_deficient);"
    "RUN_TEST(test_svd_rank_nearly_singular);"
    "RUN_TEST(test_svd_rank_diagonal_threshold_fixture);"
    "RUN_TEST(test_svd_qr_rank_dependent_row_fixture);"
    "RUN_TEST(test_svd_rank_null);"
    "RUN_TEST(test_pinv_diagonal);"
    "RUN_TEST(test_pinv_moore_penrose);"
    "RUN_TEST(test_pinv_null);"
    "RUN_TEST(test_pinv_rectangular);"
    "RUN_TEST(test_pinv_underdetermined_minnorm_solution);"
    "RUN_TEST(test_lowrank_diagonal);"
    "RUN_TEST(test_lowrank_error_bound);"
    "RUN_TEST(test_lowrank_errors);"
)

fail() {
    echo "svd-helper-guard: FAIL: $1" >&2
    exit 1
}

pass() {
    echo "svd-helper-guard: $1 ok"
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

require_increasing_run_test_order() {
    local marker
    local line
    local previous_line=0

    for marker in "${RUN_TEST_MARKERS[@]}"; do
        line="$(grep --fixed-strings --line-number -- "$marker" "$TEST_FILE" | head -n 1 | cut -d: -f1)"
        if [ "$line" -le "$previous_line" ]; then
            fail "tests/test_svd.c selected RUN_TEST registrations changed order near '$marker'"
        fi
        previous_line="$line"
    done
}

check_required_files() {
    require_file "$TEST_FILE" "tests/test_svd.c is missing"
    require_file "$SHARED_HELPER_PATH" "$SHARED_HELPER is missing"
    require_file "$SELECTED_HELPER_PATH" "$SELECTED_HELPER is missing"
    require_file "$MAKEFILE" "Makefile is missing"
    require_file "$CMAKE_FILE" "CMakeLists.txt is missing"
    require_file "$LIBRARY_MANIFEST" "build-metadata/library_sources.txt is missing"

    pass "required files"
}

check_proof_owner_registration() {
    require_fixed '$(TESTDIR)/test_svd.c' "$MAKEFILE" \
        "Makefile no longer registers test_svd.c in TEST_SRCS"
    require_fixed 'add_sparse_test(test_svd)' "$CMAKE_FILE" \
        "CMakeLists.txt no longer registers test_svd"

    pass "proof-owner registration"
}

check_helper_boundary() {
    require_fixed "#ifndef TEST_SVD_HELPERS_H" "$SHARED_HELPER_PATH" \
        "$SHARED_HELPER is missing include guard TEST_SVD_HELPERS_H"
    require_fixed "#define TEST_SVD_HELPERS_H" "$SHARED_HELPER_PATH" \
        "$SHARED_HELPER is missing include guard define TEST_SVD_HELPERS_H"
    require_fixed "#ifndef TEST_SVD_SELECTED_HELPERS_H" "$SELECTED_HELPER_PATH" \
        "$SELECTED_HELPER is missing include guard TEST_SVD_SELECTED_HELPERS_H"
    require_fixed "#define TEST_SVD_SELECTED_HELPERS_H" "$SELECTED_HELPER_PATH" \
        "$SELECTED_HELPER is missing include guard define TEST_SVD_SELECTED_HELPERS_H"
    require_exact_fixed_count "#include \"$SHARED_HELPER_NAME\"" "$TEST_FILE" 1 \
        "tests/test_svd.c must include $SHARED_HELPER_NAME exactly once"
    require_exact_fixed_count "#include \"$SELECTED_HELPER_NAME\"" "$TEST_FILE" 1 \
        "tests/test_svd.c must include $SELECTED_HELPER_NAME exactly once"
    require_fixed "#include \"$SHARED_HELPER_NAME\"" "$SELECTED_HELPER_PATH" \
        "$SELECTED_HELPER must include $SHARED_HELPER_NAME for shared SVD fixtures"
    require_fixed '#include "sparse_qr.h"' "$SELECTED_HELPER_PATH" \
        "$SELECTED_HELPER must include sparse_qr.h for QR rank helper dependencies"
    require_fixed '#include "sparse_svd.h"' "$SELECTED_HELPER_PATH" \
        "$SELECTED_HELPER must include sparse_svd.h for SVD helper dependencies"
    require_fixed '#include "sparse_vector.h"' "$SELECTED_HELPER_PATH" \
        "$SELECTED_HELPER must include sparse_vector.h for minimum-norm helper dependencies"

    pass "helper boundary"
}

check_selected_cluster_ownership() {
    local marker

    for marker in "${MOVED_DEFINITION_MARKERS[@]}"; do
        require_fixed "$marker" "$SELECTED_HELPER_PATH" \
            "$SELECTED_HELPER is missing moved selected-cluster definition marker '$marker'"
        require_absent_fixed "$marker" "$SHARED_HELPER_PATH" \
            "$SHARED_HELPER still owns moved selected-cluster definition marker '$marker'"
        require_absent_fixed "$marker" "$TEST_FILE" \
            "tests/test_svd.c still owns moved selected-cluster definition marker '$marker'"
    done

    for marker in "${RUN_TEST_MARKERS[@]}"; do
        require_exact_fixed_count "$marker" "$TEST_FILE" 1 \
            "tests/test_svd.c must retain proof-owner registration '$marker' exactly once"
    done
    require_increasing_run_test_order

    pass "selected cluster ownership"
}

check_header_only_registration() {
    local selected_stem
    local shared_stem

    shared_stem="${SHARED_HELPER_NAME%.h}"
    selected_stem="${SELECTED_HELPER_NAME%.h}"
    require_fixed '$(TESTDIR)/test_svd_helpers.h' "$MAKEFILE" \
        "Makefile must list test_svd_helpers.h as a test_svd prerequisite"
    require_fixed '$(TESTDIR)/test_svd_selected_helpers.h' "$MAKEFILE" \
        "Makefile must list test_svd_selected_helpers.h as a test_svd prerequisite"
    require_absent_fixed "$SHARED_HELPER_NAME" "$CMAKE_FILE" \
        "$SHARED_HELPER_NAME must remain header-only and not be named in CMake registration"
    require_absent_fixed "$SELECTED_HELPER_NAME" "$CMAKE_FILE" \
        "$SELECTED_HELPER_NAME must remain header-only and not be named in CMake registration"
    require_absent_fixed "$SHARED_HELPER" "$LIBRARY_MANIFEST" \
        "$SHARED_HELPER must not be listed as a library source"
    require_absent_fixed "$SELECTED_HELPER" "$LIBRARY_MANIFEST" \
        "$SELECTED_HELPER must not be listed as a library source"
    require_absent_fixed "add_sparse_test($shared_stem)" "$CMAKE_FILE" \
        "$shared_stem must not become a separate CMake test without a new proof-owner decision"
    require_absent_fixed "add_sparse_test($selected_stem)" "$CMAKE_FILE" \
        "$selected_stem must not become a separate CMake test without a new proof-owner decision"

    pass "header-only registration"
}

check_required_files
check_proof_owner_registration
check_helper_boundary
check_selected_cluster_ownership
check_header_only_registration

echo "svd-helper-guard: passed"
