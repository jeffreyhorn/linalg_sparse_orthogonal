#!/usr/bin/env bash
# test_install.sh — verify make install / uninstall / pkg-config integration
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/sparse.XXXXXX")"
trap 'rm -rf "$TMPDIR"' EXIT

EXPECTED_VERSION="$(tr -d '[:space:]' < "$ROOT_DIR/VERSION")"
PREFIX="$TMPDIR/usr"
PC_FILE="$PREFIX/lib/pkgconfig/sparse.pc"
PASS=0
FAIL=0

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }
same_dir() {
    [ -n "$1" ] && [ -n "$2" ] && [ -d "$1" ] && [ -d "$2" ] && [ "$1" -ef "$2" ]
}

echo "=== Install Validation Tests ==="
echo "  root:   $ROOT_DIR"
echo "  prefix: $PREFIX"
echo ""

# ── 1. make install to temp prefix ──────────────────────────────────
echo "--- Installing ---"
INSTALL_LOG="$TMPDIR/make.log"
if ! make -C "$ROOT_DIR" clean >"$INSTALL_LOG" 2>&1; then
    echo "make clean failed; output:"
    cat "$INSTALL_LOG"
    exit 1
fi
if ! make -C "$ROOT_DIR" install PREFIX="$PREFIX" >>"$INSTALL_LOG" 2>&1; then
    echo "make install failed; output:"
    tail -30 "$INSTALL_LOG"
    exit 1
fi

# ── 2. Verify installed files ───────────────────────────────────────
echo "--- Checking installed files ---"

if [ -f "$PREFIX/lib/libsparse_lu_ortho.a" ]; then
    pass "static library installed"
else
    fail "static library not found at $PREFIX/lib/libsparse_lu_ortho.a"
fi

SHARED_ARTIFACTS=$(find "$PREFIX/lib" "$PREFIX/bin" \( -name '*.so' -o -name '*.so.*' -o -name '*.dylib' -o -name '*.dll' \) 2>/dev/null || true)
if [ -z "$SHARED_ARTIFACTS" ]; then
    pass "no shared-library artifacts installed"
else
    fail "unexpected shared-library artifacts installed: $SHARED_ARTIFACTS"
fi

HEADER_COUNT=$(find "$PREFIX/include/sparse" -name '*.h' 2>/dev/null | wc -l | tr -d ' ')
# Count source headers plus the generated sparse_version.h
EXPECTED_HEADERS=$(( $(ls "$ROOT_DIR/include/"*.h 2>/dev/null | wc -l | tr -d ' ') + 1 ))
if [ "$HEADER_COUNT" -eq "$EXPECTED_HEADERS" ]; then
    pass "all $EXPECTED_HEADERS headers installed"
else
    fail "expected $EXPECTED_HEADERS headers, found $HEADER_COUNT"
fi

if [ -f "$PC_FILE" ]; then
    pass "pkg-config file installed"
else
    fail "sparse.pc not found at $PC_FILE"
fi

# ── 3. Verify pkg-config output ────────────────────────────────────
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"

if pkg-config --print-errors --exists sparse 2>"$TMPDIR/pkg_exists.log"; then
    pass "pkg-config can resolve sparse"
else
    fail "pkg-config cannot resolve sparse"
    if [ -s "$TMPDIR/pkg_exists.log" ]; then
        cat "$TMPDIR/pkg_exists.log"
    fi
fi

if pkg-config --print-errors --exists "sparse = $EXPECTED_VERSION" 2>"$TMPDIR/pkg_version_exists.log"; then
    pass "pkg-config exact version constraint works"
else
    fail "pkg-config exact version constraint failed for $EXPECTED_VERSION"
    if [ -s "$TMPDIR/pkg_version_exists.log" ]; then
        cat "$TMPDIR/pkg_version_exists.log"
    fi
fi

PC_PREFIX="$(pkg-config --variable=prefix sparse 2>/dev/null || true)"
if same_dir "$PC_PREFIX" "$PREFIX"; then
    pass "pkg-config prefix points at install prefix"
else
    fail "pkg-config prefix expected $PREFIX, got '$PC_PREFIX'"
fi

PC_LIBDIR="$(pkg-config --variable=libdir sparse 2>/dev/null || true)"
if same_dir "$PC_LIBDIR" "$PREFIX/lib"; then
    pass "pkg-config libdir points at installed libdir"
else
    fail "pkg-config libdir expected $PREFIX/lib, got '$PC_LIBDIR'"
fi

PC_INCLUDEDIR="$(pkg-config --variable=includedir sparse 2>/dev/null || true)"
if same_dir "$PC_INCLUDEDIR" "$PREFIX/include"; then
    pass "pkg-config includedir points at installed includedir"
else
    fail "pkg-config includedir expected $PREFIX/include, got '$PC_INCLUDEDIR'"
fi

PC_CFLAGS="$(pkg-config --cflags sparse 2>/dev/null || true)"
set -- $PC_CFLAGS
PC_CFLAGS_INCLUDE=""
if [ "$#" -gt 0 ]; then
    PC_CFLAGS_INCLUDE="${1#-I}"
fi
if [ "$#" -eq 1 ] && \
    [ "${1#-I}" != "$1" ] && \
    same_dir "$PC_CFLAGS_INCLUDE" "$PREFIX/include"; then
    pass "pkg-config --cflags returns installed include path"
else
    fail "pkg-config --cflags expected installed include path, got '$PC_CFLAGS'"
fi

PC_LIBS="$(pkg-config --libs sparse 2>/dev/null || true)"
set -- $PC_LIBS
PC_LIBDIR_FLAG=""
if [ "$#" -gt 0 ]; then
    PC_LIBDIR_FLAG="${1#-L}"
fi
if [ "$#" -eq 3 ] && \
    [ "${1#-L}" != "$1" ] && \
    same_dir "$PC_LIBDIR_FLAG" "$PREFIX/lib" && \
    [ "$2" = "-lsparse_lu_ortho" ] && \
    [ "$3" = "-lm" ]; then
    pass "pkg-config --libs returns installed static archive link flags"
else
    fail "pkg-config --libs expected installed static archive link flags, got '$PC_LIBS'"
fi

PC_STATIC_LIBS="$(pkg-config --libs --static sparse 2>/dev/null || true)"
if [ "$PC_STATIC_LIBS" = "$PC_LIBS" ]; then
    pass "pkg-config --static libs match current self-contained link flags"
else
    fail "pkg-config --static libs expected '$PC_LIBS', got '$PC_STATIC_LIBS'"
fi

if ! grep -Eq '^Libs\.private:' "$PC_FILE"; then
    pass "pkg-config file has no private dependency stanza"
else
    fail "pkg-config file unexpectedly declares Libs.private"
fi

if grep -Fxq 'Description: Static archive package metadata for sparse linear algebra' "$PC_FILE"; then
    pass "pkg-config file describes static archive package metadata"
else
    fail "pkg-config file does not describe the static archive package contract"
fi

if ! grep -Eiq 'shared|soname|dylib|dll|abi|homebrew|apt|dnf|pacman|vcpkg|conan' "$PC_FILE"; then
    pass "pkg-config file has no unsupported packaging or ABI claims"
else
    fail "pkg-config file contains unsupported packaging or ABI wording"
fi

PC_VERSION="$(pkg-config --modversion sparse 2>/dev/null || true)"
if [ "$PC_VERSION" = "$EXPECTED_VERSION" ]; then
    pass "pkg-config --modversion returns $PC_VERSION"
else
    fail "pkg-config --modversion expected $EXPECTED_VERSION, got '$PC_VERSION'"
fi

# ── 4. Compile and link downstream consumers against installed library ─────
echo "--- Compiling downstream consumers against installed library ---"
cat > "$TMPDIR/test_link.c" << 'CEOF'
#include <sparse/sparse_types.h>
#include <sparse/sparse_matrix.h>
#include <stdio.h>

int main(void) {
    printf("sparse version: %s\n", SPARSE_VERSION_STRING);
    printf("version int:    %d\n", SPARSE_VERSION);
    SparseMatrix *A = sparse_create(3, 3);
    if (!A) return 1;
    sparse_insert(A, 0, 0, 1.0);
    printf("nnz: %d\n", (int)sparse_nnz(A));
    sparse_free(A);
    printf("OK\n");
    return 0;
}
CEOF

CC="${CC:-cc}"
COMPILE_LOG="$TMPDIR/compile.log"
EXAMPLE_COMPILE_LOG="$TMPDIR/example_compile.log"
PKG_CONFIG_LOG="$TMPDIR/pkg_config.log"

if ! command -v pkg-config >/dev/null 2>&1; then
    fail "pkg-config not found; cannot validate downstream consumer checks"
elif CFLAGS_PC="$(pkg-config --cflags sparse 2>"$PKG_CONFIG_LOG")" && \
     LIBS_PC="$(pkg-config --libs sparse 2>>"$PKG_CONFIG_LOG")"; then
    if $CC -std=c11 -Wall $CFLAGS_PC "$TMPDIR/test_link.c" $LIBS_PC -o "$TMPDIR/test_link" 2>"$COMPILE_LOG"; then
        pass "basic pkg-config consumer compiles and links"
    else
        fail "basic pkg-config consumer failed to compile/link"
        if [ -s "$COMPILE_LOG" ]; then
            echo "Compiler/linker output:"
            cat "$COMPILE_LOG"
        fi
    fi

    if [ -x "$TMPDIR/test_link" ]; then
        OUTPUT="$("$TMPDIR/test_link" 2>&1)"
        if echo "$OUTPUT" | grep -q "sparse version:" && \
            echo "$OUTPUT" | grep -q "version int:" && \
            echo "$OUTPUT" | grep -q "nnz: 1" && \
            echo "$OUTPUT" | grep -q "OK"; then
            pass "basic pkg-config consumer runs correctly"
        else
            fail "basic pkg-config consumer output unexpected: $OUTPUT"
        fi
    fi

    if $CC -std=c11 -Wall $CFLAGS_PC \
        "$ROOT_DIR/examples/cmake_example/main.c" \
        $LIBS_PC \
        -o "$TMPDIR/example_pkgconfig" 2>"$EXAMPLE_COMPILE_LOG"; then
        pass "maintained example source compiles with pkg-config"
    else
        fail "maintained example source failed to compile/link with pkg-config"
        if [ -s "$EXAMPLE_COMPILE_LOG" ]; then
            echo "Compiler/linker output:"
            cat "$EXAMPLE_COMPILE_LOG"
        fi
    fi

    if [ -x "$TMPDIR/example_pkgconfig" ]; then
        OUTPUT="$("$TMPDIR/example_pkgconfig" 2>&1)"
        if echo "$OUTPUT" | grep -q "Sparse library version" && \
            echo "$OUTPUT" | grep -q "Solution:" && \
            echo "$OUTPUT" | grep -q "OK"; then
            pass "maintained example source runs with pkg-config install"
        else
            fail "maintained example source output unexpected: $OUTPUT"
        fi
    fi
else
    fail "pkg-config could not resolve sparse compiler/linker flags"
    if [ -s "$PKG_CONFIG_LOG" ]; then
        echo "pkg-config output:"
        cat "$PKG_CONFIG_LOG"
    fi
fi

# ── 5. make uninstall ───────────────────────────────────────────────
echo "--- Uninstalling ---"
make -C "$ROOT_DIR" uninstall PREFIX="$PREFIX" >/dev/null 2>&1

if [ ! -f "$PREFIX/lib/libsparse_lu_ortho.a" ]; then
    pass "library removed after uninstall"
else
    fail "library still present after uninstall"
fi

if [ ! -d "$PREFIX/include/sparse" ]; then
    pass "headers removed after uninstall"
else
    fail "header directory still present after uninstall"
fi

if [ ! -f "$PREFIX/lib/pkgconfig/sparse.pc" ]; then
    pass "pkg-config file removed after uninstall"
else
    fail "sparse.pc still present after uninstall"
fi

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "--- Summary ---"
echo "Passed: $PASS"
echo "Failed: $FAIL"

if [ "$FAIL" -ne 0 ]; then
    echo "INSTALL TESTS FAILED"
    exit 1
fi

echo "ALL INSTALL TESTS PASSED"
