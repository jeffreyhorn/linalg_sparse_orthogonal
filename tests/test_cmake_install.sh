#!/usr/bin/env bash
# test_cmake_install.sh — verify CMake install, find_package, and example project
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/sparse.XXXXXX")"
trap 'rm -rf "$TMPDIR"' EXIT

EXPECTED_VERSION="$(tr -d '[:space:]' < "$ROOT_DIR/VERSION")"
PREFIX="$TMPDIR/usr"
BUILD="$TMPDIR/build"
EXAMPLE_BUILD="$TMPDIR/example_build"
VERSION_EXACT_BUILD="$TMPDIR/version_exact_build"
VERSION_MISMATCH_BUILD="$TMPDIR/version_mismatch_build"
VERSION_EXACT_SRC="$TMPDIR/version_exact_src"
VERSION_MISMATCH_SRC="$TMPDIR/version_mismatch_src"
LOG="$TMPDIR/cmake.log"
CMAKE_PACKAGE_DIR="$PREFIX/lib/cmake/Sparse"
SPARSE_TARGETS="$CMAKE_PACKAGE_DIR/SparseTargets.cmake"
SPARSE_TARGETS_NOCONFIG="$CMAKE_PACKAGE_DIR/SparseTargets-noconfig.cmake"
PASS=0
FAIL=0
SKIP=0

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1: $2"; FAIL=$((FAIL + 1)); }
skip() { echo "  [SKIP] $1: $2"; SKIP=$((SKIP + 1)); }

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

SHARED_ARTIFACTS=$(find "$PREFIX/lib" "$PREFIX/bin" \( -name '*.so' -o -name '*.so.*' -o -name '*.dylib' -o -name '*.dll' \) 2>/dev/null || true)
if [ -z "$SHARED_ARTIFACTS" ]; then
    pass "no shared-library artifacts installed"
else
    fail "shared-library artifacts" "$SHARED_ARTIFACTS"
fi

HEADER_COUNT=$(find "$PREFIX/include/sparse" -name '*.h' 2>/dev/null | wc -l | tr -d ' ')
EXPECTED_HEADERS=$(( $(find "$ROOT_DIR/include" -maxdepth 1 -name '*.h' | wc -l | tr -d ' ') + 1 ))
if [ "$HEADER_COUNT" -eq "$EXPECTED_HEADERS" ]; then
    pass "headers installed ($HEADER_COUNT files)"
else
    fail "headers" "expected $EXPECTED_HEADERS, found $HEADER_COUNT"
fi

if [ -f "$CMAKE_PACKAGE_DIR/SparseConfig.cmake" ]; then
    pass "SparseConfig.cmake installed"
else
    fail "SparseConfig.cmake" "not found"
fi

if [ -f "$CMAKE_PACKAGE_DIR/SparseConfigVersion.cmake" ]; then
    pass "SparseConfigVersion.cmake installed"
else
    fail "SparseConfigVersion.cmake" "not found"
fi

if [ -f "$SPARSE_TARGETS" ]; then
    pass "SparseTargets.cmake installed"
else
    fail "SparseTargets.cmake" "not found"
fi

if [ -f "$PREFIX/lib/pkgconfig/sparse.pc" ]; then
    pass "sparse.pc installed"
else
    fail "sparse.pc" "not found"
fi

echo "--- Checking installed CMake package metadata ---"

if [ -f "$SPARSE_TARGETS" ] && \
    grep -Fq "add_library(Sparse::sparse_lu_ortho STATIC IMPORTED)" "$SPARSE_TARGETS"; then
    pass "CMake imported target is static"
else
    fail "CMake imported target type" "expected STATIC IMPORTED target"
fi

if [ -f "$SPARSE_TARGETS" ] && \
    grep -Fq 'INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"' "$SPARSE_TARGETS"; then
    pass "CMake imported target uses install include prefix"
else
    fail "CMake imported target include dirs" "expected _IMPORT_PREFIX include path"
fi

if [ -f "$SPARSE_TARGETS_NOCONFIG" ] && \
    grep -Fq 'IMPORTED_LOCATION_NOCONFIG' "$SPARSE_TARGETS_NOCONFIG" && \
    grep -Fq '${_IMPORT_PREFIX}/lib/libsparse_lu_ortho.a' "$SPARSE_TARGETS_NOCONFIG"; then
    pass "CMake imported archive uses install prefix"
else
    fail "CMake imported archive location" "expected _IMPORT_PREFIX static archive path"
fi

PACKAGE_SOURCE_LEAKS=$(grep -R -n -F "$ROOT_DIR" "$CMAKE_PACKAGE_DIR" 2>/dev/null || true)
if [ -z "$PACKAGE_SOURCE_LEAKS" ]; then
    pass "CMake package has no source-tree paths"
else
    fail "CMake package source-tree paths" "$PACKAGE_SOURCE_LEAKS"
fi

PACKAGE_BUILD_LEAKS=$(grep -R -n -F "$BUILD" "$CMAKE_PACKAGE_DIR" 2>/dev/null || true)
if [ -z "$PACKAGE_BUILD_LEAKS" ]; then
    pass "CMake package has no build-tree paths"
else
    fail "CMake package build-tree paths" "$PACKAGE_BUILD_LEAKS"
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

# ── 4. Version compatibility contract ──────────────────────────────
echo "--- Version compatibility checks ---"

mkdir -p "$VERSION_EXACT_BUILD" "$VERSION_MISMATCH_BUILD"
mkdir -p "$VERSION_EXACT_SRC" "$VERSION_MISMATCH_SRC"

cat > "$VERSION_EXACT_SRC/CMakeLists.txt" << EOF
cmake_minimum_required(VERSION 3.14)
project(sparse_version_exact C)
find_package(Sparse ${EXPECTED_VERSION} EXACT REQUIRED)
add_executable(version_exact "$ROOT_DIR/examples/cmake_example/main.c")
target_link_libraries(version_exact PRIVATE Sparse::sparse_lu_ortho)
EOF

echo "--- exact-version configure ---" >>"$LOG"
if cmake -S "$VERSION_EXACT_SRC" -B "$VERSION_EXACT_BUILD" \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    >>"$LOG" 2>&1; then
    pass "find_package exact installed version works"
else
    fail "find_package exact installed version" "see $LOG"
    tail -20 "$LOG"
fi

IFS=. read -r version_major version_minor version_patch << EOF
$EXPECTED_VERSION
EOF
if [ "$version_minor" -gt 0 ]; then
    MISMATCH_VERSION="${version_major}.$((version_minor - 1)).0"
elif [ "$version_patch" -gt 0 ]; then
    MISMATCH_VERSION="${version_major}.${version_minor}.$((version_patch - 1))"
else
    skip "find_package mismatched version" \
        "no lower same-major version exists for $EXPECTED_VERSION"
    MISMATCH_VERSION=""
fi

if [ -n "$MISMATCH_VERSION" ]; then
cat > "$VERSION_MISMATCH_SRC/CMakeLists.txt" << EOF
cmake_minimum_required(VERSION 3.14)
project(sparse_version_mismatch C)
find_package(Sparse ${MISMATCH_VERSION} REQUIRED)
EOF

echo "--- mismatched-version configure ---" >>"$LOG"
if cmake -S "$VERSION_MISMATCH_SRC" -B "$VERSION_MISMATCH_BUILD" \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    >>"$LOG" 2>&1; then
    fail "find_package mismatched version" "configure unexpectedly succeeded; see $LOG"
    tail -20 "$LOG"
else
    pass "find_package mismatched version is rejected"
fi
fi

# ── 5. Version check ───────────────────────────────────────────────
echo "--- Version checks ---"

# pkg-config version
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"
PC_VERSION="$(pkg-config --modversion sparse 2>/dev/null || true)"
if [ "$PC_VERSION" = "$EXPECTED_VERSION" ]; then
    pass "pkg-config version = $PC_VERSION"
else
    fail "pkg-config version" "expected $EXPECTED_VERSION, got '$PC_VERSION'"
fi

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "--- Summary ---"
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo "Skipped: $SKIP"

if [ "$FAIL" -ne 0 ]; then
    echo "CMAKE INSTALL TESTS FAILED"
    exit 1
fi

echo "ALL CMAKE INSTALL TESTS PASSED"
