#!/usr/bin/env bash
# test_install.sh — verify make install / uninstall / pkg-config integration
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/sparse.XXXXXX")"
trap 'rm -rf "$TMPDIR"' EXIT

PREFIX="$TMPDIR/usr"
PASS=0
FAIL=0

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }

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

HEADER_COUNT=$(find "$PREFIX/include/sparse" -name '*.h' 2>/dev/null | wc -l | tr -d ' ')
# Count source headers plus the generated sparse_version.h
EXPECTED_HEADERS=$(( $(ls "$ROOT_DIR/include/"*.h 2>/dev/null | wc -l | tr -d ' ') + 1 ))
if [ "$HEADER_COUNT" -eq "$EXPECTED_HEADERS" ]; then
    pass "all $EXPECTED_HEADERS headers installed"
else
    fail "expected $EXPECTED_HEADERS headers, found $HEADER_COUNT"
fi

if [ -f "$PREFIX/lib/pkgconfig/sparse.pc" ]; then
    pass "pkg-config file installed"
else
    fail "sparse.pc not found"
fi

# ── 3. Verify pkg-config output ────────────────────────────────────
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"

PC_CFLAGS="$(pkg-config --cflags sparse 2>/dev/null || true)"
if echo "$PC_CFLAGS" | grep -q "\-I.*include"; then
    pass "pkg-config --cflags returns include path"
else
    fail "pkg-config --cflags unexpected: $PC_CFLAGS"
fi

PC_LIBS="$(pkg-config --libs sparse 2>/dev/null || true)"
if echo "$PC_LIBS" | grep -q "\-lsparse_lu_ortho"; then
    pass "pkg-config --libs returns library flag"
else
    fail "pkg-config --libs unexpected: $PC_LIBS"
fi

PC_VERSION="$(pkg-config --modversion sparse 2>/dev/null || true)"
if [ -n "$PC_VERSION" ]; then
    pass "pkg-config --modversion returns $PC_VERSION"
else
    fail "pkg-config --modversion empty"
fi

# ── 4. Compile and link a test program against installed library ────
echo "--- Compiling example against installed library ---"
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
PKG_CONFIG_LOG="$TMPDIR/pkg_config.log"

if ! command -v pkg-config >/dev/null 2>&1; then
    fail "pkg-config not found; skipping compile/link test"
elif CFLAGS_PC="$(pkg-config --cflags sparse 2>"$PKG_CONFIG_LOG")" && \
     LIBS_PC="$(pkg-config --libs sparse 2>>"$PKG_CONFIG_LOG")"; then
    if $CC -std=c11 -Wall $CFLAGS_PC "$TMPDIR/test_link.c" $LIBS_PC -o "$TMPDIR/test_link" 2>"$COMPILE_LOG"; then
        pass "example compiles and links"
    else
        fail "example failed to compile/link"
        if [ -s "$COMPILE_LOG" ]; then
            echo "Compiler/linker output:"
            cat "$COMPILE_LOG"
        fi
    fi

    if [ -x "$TMPDIR/test_link" ]; then
        OUTPUT="$("$TMPDIR/test_link" 2>&1)"
        if echo "$OUTPUT" | grep -q "OK"; then
            pass "example runs correctly"
        else
            fail "example output unexpected: $OUTPUT"
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
