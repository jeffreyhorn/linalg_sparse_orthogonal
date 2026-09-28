#!/usr/bin/env bash
# check_ldlt_csc_helper_guard.sh - Sprint 185 LDLT CSC helper guard.
#
# Keeps the extracted family-local helper headers tied to the registered
# `test_ldlt_csc` proof-owner binary. The headers are intentionally included
# by `tests/test_ldlt_csc.c`, not registered as standalone tests or library
# sources.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_FILE="$ROOT_DIR/tests/test_ldlt_csc.c"
MAKEFILE="$ROOT_DIR/Makefile"
CMAKE_FILE="$ROOT_DIR/CMakeLists.txt"
LIBRARY_MANIFEST="$ROOT_DIR/build-metadata/library_sources.txt"
HELPERS=(
    "tests/test_ldlt_csc_fixtures.h"
    "tests/test_ldlt_csc_native_parity_helpers.h"
    "tests/test_ldlt_csc_oracle_helpers.h"
    "tests/test_ldlt_csc_supernode_helpers.h"
)

RUN_TEST_MARKERS=(
    "RUN_TEST(test_native_1x1_diagonal_matches_wrapper);"
    "RUN_TEST(test_native_1x1_tridiagonal_matches_wrapper);"
    "RUN_TEST(test_native_1x1_mixed_indefinite_matches_wrapper);"
    "RUN_TEST(test_native_1x1_with_swap_matches_wrapper);"
    "RUN_TEST(test_native_1x1_tridiag_large_matches_wrapper);"
    "RUN_TEST(test_native_detects_near_zero_1x1_pivot);"
    "RUN_TEST(test_native_1x1_identity_matches_wrapper);"
    "RUN_TEST(test_native_2x2_forced_matches_wrapper);"
    "RUN_TEST(test_native_2x2_nonadjacent_partner_matches_wrapper);"
    "RUN_TEST(test_native_mixed_pivots_matches_wrapper);"
    "RUN_TEST(test_native_mixed_pivots_larger_matches_wrapper);"
    "RUN_TEST(test_native_2x2_solve_matches_linked_list);"
    "RUN_TEST(test_native_2x2_inertia_matches_wrapper);"
)

MOVED_DEFINITION_MARKERS=(
    "static void test_native_1x1_diagonal_matches_wrapper(void) {"
    "static void test_native_1x1_tridiagonal_matches_wrapper(void) {"
    "static void test_native_1x1_mixed_indefinite_matches_wrapper(void) {"
    "static void test_native_1x1_with_swap_matches_wrapper(void) {"
    "static void test_native_1x1_tridiag_large_matches_wrapper(void) {"
    "static void test_native_detects_near_zero_1x1_pivot(void) {"
    "static void test_native_1x1_identity_matches_wrapper(void) {"
    "static void test_native_2x2_forced_matches_wrapper(void) {"
    "static void test_native_2x2_nonadjacent_partner_matches_wrapper(void) {"
    "static void test_native_mixed_pivots_matches_wrapper(void) {"
    "static void test_native_mixed_pivots_larger_matches_wrapper(void) {"
    "static void test_native_2x2_solve_matches_linked_list(void) {"
    "static void test_native_2x2_inertia_matches_wrapper(void) {"
)

NATIVE_HELPER="tests/test_ldlt_csc_native_parity_helpers.h"
NATIVE_HELPER_PATH="$ROOT_DIR/$NATIVE_HELPER"

fail() {
    echo "ldlt-csc-helper-guard: FAIL: $1" >&2
    exit 1
}

pass() {
    echo "ldlt-csc-helper-guard: $1 ok"
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

active_fixed_count_in_file() {
    local marker="$1"
    local file="$2"

    awk -v marker="$marker" '
        function strip_comments(line,    start, end, out) {
            out = ""
            while (length(line) > 0) {
                if (in_block_comment) {
                    end = index(line, "*/")
                    if (end == 0)
                        return out
                    line = substr(line, end + 2)
                    in_block_comment = 0
                    continue
                }
                start = index(line, "/*")
                if (start == 0) {
                    sub(/[[:space:]]*\/\/.*/, "", line)
                    return out line
                }
                out = out substr(line, 1, start - 1)
                line = substr(line, start + 2)
                in_block_comment = 1
            }
            return out
        }
        {
            active = strip_comments($0)
            if (index(active, marker) > 0)
                count++
        }
        END { print count + 0 }
    ' "$file"
}

active_include_count() {
    local include_name="$1"

    awk -v include_name="$include_name" '
        function strip_comments(line,    start, end, out) {
            out = ""
            while (length(line) > 0) {
                if (in_block_comment) {
                    end = index(line, "*/")
                    if (end == 0)
                        return out
                    line = substr(line, end + 2)
                    in_block_comment = 0
                    continue
                }
                start = index(line, "/*")
                if (start == 0) {
                    sub(/[[:space:]]*\/\/.*/, "", line)
                    return out line
                }
                out = out substr(line, 1, start - 1)
                line = substr(line, start + 2)
                in_block_comment = 1
            }
            return out
        }
        {
            active = strip_comments($0)
            if (active ~ /^[[:space:]]*#[[:space:]]*include[[:space:]]*"/ &&
                index(active, "\"" include_name "\"") > 0)
                count++
        }
        END { print count + 0 }
    ' "$TEST_FILE"
}

active_run_test_count() {
    local marker="$1"

    awk -v marker="$marker" '
        function strip_comments(line,    start, end, out) {
            out = ""
            while (length(line) > 0) {
                if (in_block_comment) {
                    end = index(line, "*/")
                    if (end == 0)
                        return out
                    line = substr(line, end + 2)
                    in_block_comment = 0
                    continue
                }
                start = index(line, "/*")
                if (start == 0) {
                    sub(/[[:space:]]*\/\/.*/, "", line)
                    return out line
                }
                out = out substr(line, 1, start - 1)
                line = substr(line, start + 2)
                in_block_comment = 1
            }
            return out
        }
        {
            active = strip_comments($0)
            if (active ~ /^[[:space:]]*RUN_TEST[[:space:]]*\(/ && index(active, marker) > 0)
                count++
        }
        END { print count + 0 }
    ' "$TEST_FILE"
}

active_run_test_line() {
    local marker="$1"

    awk -v marker="$marker" '
        function strip_comments(line,    start, end, out) {
            out = ""
            while (length(line) > 0) {
                if (in_block_comment) {
                    end = index(line, "*/")
                    if (end == 0)
                        return out
                    line = substr(line, end + 2)
                    in_block_comment = 0
                    continue
                }
                start = index(line, "/*")
                if (start == 0) {
                    sub(/[[:space:]]*\/\/.*/, "", line)
                    return out line
                }
                out = out substr(line, 1, start - 1)
                line = substr(line, start + 2)
                in_block_comment = 1
            }
            return out
        }
        {
            active = strip_comments($0)
            if (active ~ /^[[:space:]]*RUN_TEST[[:space:]]*\(/ && index(active, marker) > 0) {
                print NR
                found = 1
                exit
            }
        }
        END {
            if (!found)
                print 0
        }
    ' "$TEST_FILE"
}

require_active_run_test_registration() {
    local marker="$1"
    local count

    count="$(active_run_test_count "$marker")"
    if [ "$count" -ne 1 ]; then
        fail "tests/test_ldlt_csc.c must retain proof-owner registration '$marker' exactly once as an active RUN_TEST line (found $count)"
    fi
}

require_increasing_run_test_order() {
    local marker
    local line
    local previous_line=0

    for marker in "${RUN_TEST_MARKERS[@]}"; do
        line="$(active_run_test_line "$marker")"
        if [ "$line" -le "$previous_line" ]; then
            fail "tests/test_ldlt_csc.c selected RUN_TEST registrations changed order near '$marker'"
        fi
        previous_line="$line"
    done
}

require_test_ldlt_csc_rule_prerequisite() {
    local needle="$1"
    local message="$2"

    if ! awk -v needle="$needle" '
        BEGIN { done = 0 }
        /^\$\(BUILDDIR\)\/test_ldlt_csc:/ {
            done = 1
            exit(index($0, needle) > 0 ? 0 : 1)
        }
        END {
            if (!done)
                exit(1)
        }
    ' "$MAKEFILE"; then
        fail "$message"
    fi
}

check_proof_owner_registration() {
    require_file "$TEST_FILE" "tests/test_ldlt_csc.c is missing"
    require_fixed '$(TESTDIR)/test_ldlt_csc.c' "$MAKEFILE" \
        "Makefile no longer registers test_ldlt_csc.c in TEST_SRCS"
    require_fixed 'add_sparse_test(test_ldlt_csc)' "$CMAKE_FILE" \
        "CMakeLists.txt no longer registers test_ldlt_csc"

    pass "proof-owner registration"
}

check_helper_headers() {
    local helper
    local helper_path
    local include_name
    local guard
    local count

    for helper in "${HELPERS[@]}"; do
        helper_path="$ROOT_DIR/$helper"
        include_name="$(basename "$helper")"
        guard="$(printf '%s' "$include_name" | tr '[:lower:].' '[:upper:]_')"

        require_file "$helper_path" "$helper is missing"
        require_fixed "#ifndef $guard" "$helper_path" "$helper is missing include guard $guard"
        require_fixed "#define $guard" "$helper_path" "$helper is missing include guard define $guard"
        count="$(active_include_count "$include_name")"
        if [ "$count" -ne 1 ]; then
            fail "tests/test_ldlt_csc.c must include $include_name exactly once as an active include (found $count)"
        fi
        require_test_ldlt_csc_rule_prerequisite "\$(TESTDIR)/$include_name" \
            "Makefile test_ldlt_csc prerequisite rule must list $include_name"
    done

    pass "helper headers"
}

check_header_only_registration() {
    local helper
    local include_name
    local stem

    require_file "$LIBRARY_MANIFEST" "build-metadata/library_sources.txt is missing"

    for helper in "${HELPERS[@]}"; do
        include_name="$(basename "$helper")"
        stem="${include_name%.h}"

        require_absent_fixed "$include_name" "$CMAKE_FILE" \
            "$include_name must remain header-only and not be named in CMake registration"
        require_absent_fixed "$helper" "$LIBRARY_MANIFEST" \
            "$helper must not be listed as a library source"
        require_absent_fixed "add_sparse_test($stem)" "$CMAKE_FILE" \
            "$stem must not become a separate CMake test without a new proof-owner decision"
    done

    pass "header-only registration"
}

check_selected_run_test_registrations() {
    local marker

    for marker in "${RUN_TEST_MARKERS[@]}"; do
        require_active_run_test_registration "$marker"
    done
    require_increasing_run_test_order

    pass "selected RUN_TEST registrations"
}

check_moved_definition_ownership() {
    local marker
    local helper
    local helper_path
    local count

    for marker in "${MOVED_DEFINITION_MARKERS[@]}"; do
        count="$(active_fixed_count_in_file "$marker" "$NATIVE_HELPER_PATH")"
        if [ "$count" -ne 1 ]; then
            fail "$NATIVE_HELPER must own moved selected-cluster definition '$marker' exactly once as active code (found $count)"
        fi

        count="$(active_fixed_count_in_file "$marker" "$TEST_FILE")"
        if [ "$count" -ne 0 ]; then
            fail "tests/test_ldlt_csc.c must not retain moved selected-cluster definition '$marker'"
        fi

        for helper in "${HELPERS[@]}"; do
            if [ "$helper" = "$NATIVE_HELPER" ]; then
                continue
            fi
            helper_path="$ROOT_DIR/$helper"
            count="$(active_fixed_count_in_file "$marker" "$helper_path")"
            if [ "$count" -ne 0 ]; then
                fail "$helper must not own moved selected-cluster definition '$marker'"
            fi
        done
    done

    pass "moved definition ownership"
}

check_proof_owner_registration
check_helper_headers
check_header_only_registration
check_selected_run_test_registrations
check_moved_definition_ownership

echo "ldlt-csc-helper-guard: passed"
