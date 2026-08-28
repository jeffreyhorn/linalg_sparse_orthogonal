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
    "tests/test_ldlt_csc_oracle_helpers.h"
    "tests/test_ldlt_csc_supernode_helpers.h"
)

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

    matches="$(grep -Fn "$needle" "$file" 2>/dev/null || true)"
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

    count="$(grep -Fc "$needle" "$file" 2>/dev/null || true)"
    if [ "$count" -ne "$expected" ]; then
        fail "$message (expected $expected, found $count)"
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

    for helper in "${HELPERS[@]}"; do
        helper_path="$ROOT_DIR/$helper"
        include_name="$(basename "$helper")"
        guard="$(printf '%s' "$include_name" | tr '[:lower:].' '[:upper:]_')"

        require_file "$helper_path" "$helper is missing"
        require_fixed "#ifndef $guard" "$helper_path" "$helper is missing include guard $guard"
        require_fixed "#define $guard" "$helper_path" "$helper is missing include guard define $guard"
        require_exact_fixed_count "#include \"$include_name\"" "$TEST_FILE" 1 \
            "tests/test_ldlt_csc.c must include $include_name exactly once"
    done

    pass "helper headers"
}

check_header_only_registration() {
    local helper
    local include_name
    local stem

    for helper in "${HELPERS[@]}"; do
        include_name="$(basename "$helper")"
        stem="${include_name%.h}"

        require_absent_fixed "$include_name" "$MAKEFILE" \
            "$include_name must remain header-only and not be named in Makefile registration"
        require_absent_fixed "$include_name" "$CMAKE_FILE" \
            "$include_name must remain header-only and not be named in CMake registration"
        require_absent_fixed "$helper" "$LIBRARY_MANIFEST" \
            "$helper must not be listed as a library source"
        require_absent_fixed "add_sparse_test($stem)" "$CMAKE_FILE" \
            "$stem must not become a separate CMake test without a new proof-owner decision"
    done

    pass "header-only registration"
}

check_proof_owner_registration
check_helper_headers
check_header_only_registration

echo "ldlt-csc-helper-guard: passed"
