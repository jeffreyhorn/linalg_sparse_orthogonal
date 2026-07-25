#!/usr/bin/env bash
# static_package_deferral_check.sh - local static-first package contract guard.
#
# This script proves that shared-library packaging and dynamic ABI support stay
# explicit deferrals under the maintained static archive package contract.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/sparse_static_deferral.XXXXXX")"
trap 'rm -rf "$TMPDIR"' EXIT

fail() {
    echo "static-package-deferral-check: FAIL: $1" >&2
    exit 1
}

pass() {
    echo "static-package-deferral-check: $1 ok"
}

require_grep() {
    local pattern="$1"
    local file="$2"
    local message="$3"

    if ! grep -Eq "$pattern" "$file"; then
        fail "$message"
    fi
}

require_absent_grep() {
    local pattern="$1"
    local path="$2"
    local message="$3"
    local matches

    matches="$(grep -REn "$pattern" "$path" 2>/dev/null || true)"
    if [ -n "$matches" ]; then
        echo "$matches" >&2
        fail "$message"
    fi
}

check_build_shared_rejected() {
    local build_dir="$TMPDIR/build-shared-request"
    local stdout_file="$TMPDIR/cmake-shared-request.stdout"
    local stderr_file="$TMPDIR/cmake-shared-request.stderr"
    local rc
    local output

    set +e
    cmake -S "$ROOT_DIR" -B "$build_dir" -DBUILD_SHARED_LIBS=ON >"$stdout_file" 2>"$stderr_file"
    rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        fail "BUILD_SHARED_LIBS=ON unexpectedly configured; shared-library support is still deferred"
    fi

    output="$(cat "$stdout_file" "$stderr_file")"
    printf '%s\n' "$output" | grep -q "BUILD_SHARED_LIBS=ON was requested" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the rejected-input token"
    printf '%s\n' "$output" | grep -q "static archive package surface" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the static package contract"
    printf '%s\n' "$output" | grep -q "Shared-library packaging" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the shared-library deferral"
    printf '%s\n' "$output" | grep -q "dynamic ABI support are deferred" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the dynamic ABI deferral"

    pass "BUILD_SHARED_LIBS rejection"
}

check_static_target() {
    require_grep \
        'add_library[[:space:]]*\([[:space:]]*sparse_lu_ortho[[:space:]]+STATIC' \
        "$ROOT_DIR/CMakeLists.txt" \
        "sparse_lu_ortho is no longer declared as an explicit STATIC target"

    pass "static target declaration"
}

check_no_export_or_abi_metadata() {
    require_absent_grep \
        '(^|[^[:alnum:]_])(SPARSE_API|SPARSE_EXPORT|SPARSE_IMPORT)([^[:alnum:]_]|$)' \
        "$ROOT_DIR/include" \
        "public export/import macro appeared without a shared ABI decision"

    require_absent_grep \
        '(^|[^[:alnum:]_])(SOVERSION|WINDOWS_EXPORT_ALL_SYMBOLS|C_VISIBILITY_PRESET|VISIBILITY_INLINES_HIDDEN)([^[:alnum:]_]|$)|install[_-]?name|soname' \
        "$ROOT_DIR/CMakeLists.txt" \
        "shared-library ABI metadata appeared without a support decision"

    pass "no shared export/ABI metadata found"
}

check_no_package_selector() {
    require_absent_grep \
        'Libs\.private|Sparse::.*shared|shared[_-]?library|BUILD_SHARED_LIBS|COMPONENTS?.*(static|shared)|SPARSE_(ABI|SHARED|STATIC)' \
        "$ROOT_DIR/cmake/SparseConfig.cmake.in" \
        "CMake package selector appeared without a support decision"

    require_absent_grep \
        'Libs\.private|shared[_-]?library|BUILD_SHARED_LIBS|SPARSE_(ABI|SHARED|STATIC)' \
        "$ROOT_DIR/sparse.pc.in" \
        "pkg-config shared/static selector appeared without a support decision"

    pass "package metadata has no static/shared selector"
}

check_support_wording() {
    require_grep \
        'Shared-library packaging is intentionally deferred' \
        "$ROOT_DIR/README.md" \
        "README no longer keeps shared-library packaging deferred"
    require_grep \
        '`?BUILD_SHARED_LIBS=ON`? is intentionally rejected' \
        "$ROOT_DIR/INSTALL.md" \
        "INSTALL no longer documents BUILD_SHARED_LIBS rejection"
    require_grep \
        'dynamic ABI compatibility remain explicit non-claims' \
        "$ROOT_DIR/docs/maintainer_guide.md" \
        "maintainer guide no longer keeps dynamic ABI compatibility as a non-claim"
    require_grep \
        'package-manager support' \
        "$ROOT_DIR/docs/maintainer_guide.md" \
        "maintainer guide no longer records package-manager support boundaries"

    pass "support wording remains deferred"
}

check_build_shared_rejected
check_static_target
check_no_export_or_abi_metadata
check_no_package_selector
check_support_wording

echo "static-package-deferral-check: passed"
