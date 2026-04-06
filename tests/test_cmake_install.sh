#!/usr/bin/env bash
# test_cmake_install.sh — verify CMake install, find_package, and example project
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/sparse.XXXXXX")"
trap 'rm -rf "$TMPDIR"' EXIT

PREFIX="$TMPDIR/usr"
BUILD="$TMPDIR/build"
EXAMPLE_BUILD="$TMPDIR/example_build"
LOG="$TMPDIR/cmake.log"
PASS=0
FAIL=0

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1: $2"; FAIL=$((FAIL + 1)); }

echo "=== CMake Install Validation Tests ==="
echo "  root:   $ROOT_DIR"
echo "  prefix: $PREFIX"
echo ""

# ── 1. CMake build and install ──────────────────────────────────────
echo "--- CMake configure + build + install ---"
mkdir -p "$BUILD"
if cmake -S "$ROOT_DIR" -B "$BUILD" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_C_STANDARD=11 \
    >"$LOG" 2>&1; then
    pass "cmake configure"
else
    fail "cmake configure" "see $LOG"
    cat "$LOG"
fi

if cmake --build "$BUILD" >>"$LOG" 2>&1; then
    pass "cmake build"
else
    fail "cmake build" "see $LOG"
    tail -30 "$LOG"
fi

if cmake --install "$BUILD" >>"$LOG" 2>&1; then
    pass "cmake install"
else
    fail "cmake install" "see $LOG"
    tail -20 "$LOG"
fi

# ── 2. Verify installed files ───────────────────────────────────────
echo "--- Checking installed files ---"

if [ -f "$PREFIX/lib/libsparse_lu_ortho.a" ]; then
    pass "static library installed"
else
    fail "static library" "not found"
fi

HEADER_COUNT=$(find "$PREFIX/include/sparse" -name '*.h' 2>/dev/null | wc -l | tr -d ' ')
if [ "$HEADER_COUNT" -ge 14 ]; then
    pass "headers installed ($HEADER_COUNT files)"
else
    fail "headers" "expected >= 14, found $HEADER_COUNT"
fi

if [ -f "$PREFIX/lib/cmake/Sparse/SparseConfig.cmake" ]; then
    pass "SparseConfig.cmake installed"
else
    fail "SparseConfig.cmake" "not found"
fi

if [ -f "$PREFIX/lib/cmake/Sparse/SparseConfigVersion.cmake" ]; then
    pass "SparseConfigVersion.cmake installed"
else
    fail "SparseConfigVersion.cmake" "not found"
fi

if [ -f "$PREFIX/lib/cmake/Sparse/SparseTargets.cmake" ]; then
    pass "SparseTargets.cmake installed"
else
    fail "SparseTargets.cmake" "not found"
fi

if [ -f "$PREFIX/lib/pkgconfig/sparse.pc" ]; then
    pass "sparse.pc installed"
else
    fail "sparse.pc" "not found"
fi

# ── 3. Build cmake_example against installed library ────────────────
echo "--- Building cmake_example with find_package(Sparse) ---"
mkdir -p "$EXAMPLE_BUILD"

if cmake -S "$ROOT_DIR/examples/cmake_example" -B "$EXAMPLE_BUILD" \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    >/dev/null 2>&1; then
    pass "cmake_example configure (find_package works)"
else
    fail "cmake_example configure" "find_package(Sparse) failed"
fi

if cmake --build "$EXAMPLE_BUILD" >/dev/null 2>&1; then
    pass "cmake_example build"
else
    fail "cmake_example build" "compilation/link errors"
fi

if [ -x "$EXAMPLE_BUILD/example" ]; then
    OUTPUT="$("$EXAMPLE_BUILD/example" 2>&1)"
    if echo "$OUTPUT" | grep -q "OK"; then
        pass "cmake_example runs correctly"
    else
        fail "cmake_example run" "unexpected output: $OUTPUT"
    fi
else
    fail "cmake_example executable" "not found"
fi

# ── 4. Version check ───────────────────────────────────────────────
echo "--- Version checks ---"

# pkg-config version
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"
PC_VERSION="$(pkg-config --modversion sparse 2>/dev/null || true)"
if [ "$PC_VERSION" = "1.0.0" ]; then
    pass "pkg-config version = $PC_VERSION"
else
    fail "pkg-config version" "expected 1.0.0, got '$PC_VERSION'"
fi

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "--- Summary ---"
echo "Passed: $PASS"
echo "Failed: $FAIL"

if [ "$FAIL" -ne 0 ]; then
    echo "CMAKE INSTALL TESTS FAILED"
    exit 1
fi

echo "ALL CMAKE INSTALL TESTS PASSED"
