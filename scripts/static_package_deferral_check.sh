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
    local normalized_output

    set +e
    cmake -S "$ROOT_DIR" -B "$build_dir" -DBUILD_SHARED_LIBS=ON >"$stdout_file" 2>"$stderr_file"
    rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        fail "BUILD_SHARED_LIBS=ON unexpectedly configured; shared-library support is still deferred"
    fi

    output="$(cat "$stdout_file" "$stderr_file")"
    normalized_output="$(printf '%s\n' "$output" | tr '\n' ' ')"
    printf '%s\n' "$output" | grep -q "BUILD_SHARED_LIBS=ON was requested" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the rejected-input token"
    printf '%s\n' "$output" | grep -q "static archive package surface" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the static package contract"
    printf '%s\n' "$output" | grep -q "Shared-library packaging" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the shared-library deferral"
    printf '%s\n' "$output" | grep -q "dynamic ABI support are deferred" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the dynamic ABI deferral"
    printf '%s\n' "$normalized_output" | grep -q "export/import" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the export/import blocker"
    printf '%s\n' "$normalized_output" | grep -Eq "symbol[[:space:]]+visibility" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the symbol visibility blocker"
    printf '%s\n' "$normalized_output" | grep -Eq "dynamic[[:space:]]+ABI[[:space:]]+policy" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the dynamic ABI policy blocker"
    printf '%s\n' "$normalized_output" | grep -q "SONAME" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the Linux SONAME blocker"
    printf '%s\n' "$normalized_output" | grep -q "install-name/RPATH" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the macOS install-name/RPATH blocker"
    printf '%s\n' "$normalized_output" | grep -q "DLL/import-library" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the Windows DLL/import-library blocker"
    printf '%s\n' "$normalized_output" | grep -Eq "installed[[:space:]]+shared[[:space:]]+consumer[[:space:]]+proof" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the installed shared consumer proof blocker"
    printf '%s\n' "$normalized_output" | grep -q "runtime-loader validation" ||
        fail "BUILD_SHARED_LIBS deferral wording lost the runtime-loader blocker"

    pass "BUILD_SHARED_LIBS rejection"
}

check_static_target() {
    require_grep \
        'add_library[[:space:]]*\([[:space:]]*sparse_lu_ortho[[:space:]]+STATIC' \
        "$ROOT_DIR/CMakeLists.txt" \
        "sparse_lu_ortho is no longer declared as an explicit STATIC target"

    require_absent_grep \
        'add_library[[:space:]]*\([[:space:]]*sparse_lu_ortho[[:space:]]+(SHARED|MODULE)' \
        "$ROOT_DIR/CMakeLists.txt" \
        "sparse_lu_ortho gained a non-static library target without a shared ABI decision"

    pass "static target declaration"
}

check_static_install_metadata() {
    require_grep \
        'ARCHIVE DESTINATION \$\{CMAKE_INSTALL_LIBDIR\}' \
        "$ROOT_DIR/CMakeLists.txt" \
        "CMake install metadata no longer installs the static archive"

    require_absent_grep \
        'RUNTIME DESTINATION|LIBRARY DESTINATION' \
        "$ROOT_DIR/CMakeLists.txt" \
        "CMake install metadata gained runtime/shared-library destinations without a support decision"

    require_grep \
        '^Description: Static archive package metadata for sparse linear algebra$' \
        "$ROOT_DIR/sparse.pc.in" \
        "pkg-config description no longer states the static archive package contract"

    pass "static install metadata"
}

check_no_export_or_abi_metadata() {
    require_absent_grep \
        '(^|[^[:alnum:]_])(SPARSE_API|SPARSE_EXPORT|SPARSE_IMPORT|SPARSE_SHARED|SPARSE_STATIC|SPARSE_ABI)([^[:alnum:]_]|$)' \
        "$ROOT_DIR/include" \
        "public export/import or static/shared ABI macro appeared without a shared ABI decision"

    require_absent_grep \
        '(^|[^[:alnum:]_])(SOVERSION|WINDOWS_EXPORT_ALL_SYMBOLS|C_VISIBILITY_PRESET|VISIBILITY_INLINES_HIDDEN|INSTALL_NAME_DIR|MACOSX_RPATH)([^[:alnum:]_]|$)' \
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

check_windows_package_nonclaim_wording() {
    require_grep \
        'Windows remains CMake-first' \
        "$ROOT_DIR/README.md" \
        "README no longer states that Windows package support remains CMake-first"
    require_grep \
        'Windows still does not claim Makefile' \
        "$ROOT_DIR/README.md" \
        "README no longer keeps Windows Makefile parity as a non-claim"
    require_grep \
        '`pkg-config` execution parity' \
        "$ROOT_DIR/README.md" \
        "README no longer keeps Windows pkg-config execution parity as a non-claim"

    require_grep \
        'Windows carries reviewed CMake install/downstream validation' \
        "$ROOT_DIR/INSTALL.md" \
        "INSTALL no longer describes Windows CMake install/downstream validation"
    require_grep \
        'Windows reviewed CMake install/downstream validation does not claim Windows' \
        "$ROOT_DIR/INSTALL.md" \
        "INSTALL no longer scopes Windows CMake install/downstream validation as a non-claim boundary"
    require_grep \
        'Makefile parity, Windows `pkg-config` execution parity' \
        "$ROOT_DIR/INSTALL.md" \
        "INSTALL no longer keeps Windows Makefile/pkg-config execution parity as non-claims"

    require_grep \
        'Windows CI carries reviewed CMake install/downstream validation' \
        "$ROOT_DIR/docs/maintainer_guide.md" \
        "maintainer guide no longer describes Windows CMake install/downstream validation"
    require_grep \
        'Windows still does not claim Makefile parity, `pkg-config` execution parity' \
        "$ROOT_DIR/docs/maintainer_guide.md" \
        "maintainer guide no longer keeps Windows Makefile/pkg-config execution parity as non-claims"

    require_grep \
        '`sparse\.pc` metadata' \
        "$ROOT_DIR/.github/workflows/windows-ci.yml" \
        "Windows workflow no longer identifies sparse.pc as metadata"
    require_grep \
        'Windows does not claim Makefile parity, pkg-config execution parity' \
        "$ROOT_DIR/.github/workflows/windows-ci.yml" \
        "Windows workflow no longer keeps Makefile/pkg-config execution parity as non-claims"
    require_grep \
        'CMake install/downstream scoped: sparse\.pc is metadata-only inspection; no reviewed Makefile parity and no pkg-config execution parity' \
        "$ROOT_DIR/.github/workflows/windows-ci.yml" \
        "Windows workflow no longer separates CMake install proof from Makefile/pkg-config execution parity"

    pass "Windows package non-claim wording"
}

check_windows_workflow_no_unselected_package_execution() {
    require_absent_grep \
        '^[[:space:]]*(-[[:space:]]*)?run:[[:space:]]*(&[[:space:]]*)?pkg-config(\.exe)?([[:space:]]|$)' \
        "$ROOT_DIR/.github/workflows/windows-ci.yml" \
        "Windows workflow started executing pkg-config without a selected provider and downstream proof"
    require_absent_grep \
        '^[[:space:]]*(&[[:space:]]*)?pkg-config(\.exe)?([[:space:]]|$)' \
        "$ROOT_DIR/.github/workflows/windows-ci.yml" \
        "Windows workflow started executing pkg-config inside a script block without a selected provider and downstream proof"
    require_absent_grep \
        '^[[:space:]]*(-[[:space:]]*)?run:[[:space:]]*make[[:space:]]+(install|uninstall)([[:space:]]|$)' \
        "$ROOT_DIR/.github/workflows/windows-ci.yml" \
        "Windows workflow started executing make install/uninstall without a reviewed Windows Makefile parity decision"
    require_absent_grep \
        '^[[:space:]]*make[[:space:]]+(install|uninstall)([[:space:]]|$)' \
        "$ROOT_DIR/.github/workflows/windows-ci.yml" \
        "Windows workflow started executing make install/uninstall inside a script block without a reviewed Windows Makefile parity decision"

    pass "Windows workflow has no unselected package execution"
}

check_build_shared_rejected
check_static_target
check_static_install_metadata
check_no_export_or_abi_metadata
check_no_package_selector
check_support_wording
check_windows_package_nonclaim_wording
check_windows_workflow_no_unselected_package_execution

echo "static-package-deferral-check: passed"
