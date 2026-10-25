#!/usr/bin/env bash
# Verify Make index-width selection without touching the default build tree.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sparse-make-width.XXXXXX")"
trap 'rm -rf "$TMP_ROOT"' EXIT

PASS=0
FAIL=0

pass() { printf '  [PASS] %s\n' "$1"; PASS=$((PASS + 1)); }
fail() { printf '  [FAIL] %s\n' "$1"; FAIL=$((FAIL + 1)); }

expect_make_rejection() {
    local label="$1" expected="$2" log="$TMP_ROOT/$1.log"
    shift 2
    if make -C "$ROOT_DIR" -n BUILDDIR="$TMP_ROOT/negative-$label" "$@" all >"$log" 2>&1; then
        fail "$label unexpectedly accepted"
    elif grep -Fq "$expected" "$log"; then
        pass "$label rejected before build"
    else
        fail "$label missing expected diagnostic"
        tail -15 "$log"
    fi
}

echo '=== Make Index-Width Selection ==='

default_dir="$TMP_ROOT/default"
default_log="$TMP_ROOT/default.log"
if make -C "$ROOT_DIR" -j4 BUILDDIR="$default_dir" \
    "$default_dir/test_idx_width" >"$default_log" 2>&1 && \
    grep -Fq -- '-DSPARSE_IDX_BITS=32' "$default_log" && \
    ! grep -Fq -- '-DSPARSE_IDX_BITS=64' "$default_log" && \
    "$default_dir/test_idx_width" >"$TMP_ROOT/default-identity.log" 2>&1 && \
    grep -Fq 'typedef=32 header=32 archive=32' "$TMP_ROOT/default-identity.log"; then
    pass 'omitted selector defaults to 32 bits'
else
    fail 'omitted selector does not select only 32 bits'
    tail -15 "$default_log"
fi

for width in 32 64; do
    build_dir="$TMP_ROOT/width-$width"
    build_log="$TMP_ROOT/build-$width.log"
    test_log="$TMP_ROOT/test-$width.log"
    if make -C "$ROOT_DIR" -j4 BUILDDIR="$build_dir" SPARSE_IDX_BITS="$width" \
        "$build_dir/test_sparse_matrix" "$build_dir/test_idx_width" "$build_dir/bench_matmul" \
        "$build_dir/example_basic_solve" >"$build_log" 2>&1; then
        pass "$width-bit library, test, benchmark, and example build"
    else
        fail "$width-bit build"
        tail -25 "$build_log"
        continue
    fi

    if [ "$(sed -n '1p' "$build_dir/.sparse-idx-bits")" = "$width" ]; then
        pass "$width-bit build stamp"
    else
        fail "$width-bit build stamp"
    fi

    if (cd "$ROOT_DIR" && "$build_dir/test_sparse_matrix") >"$test_log" 2>&1; then
        pass "$width-bit typedef/archive width test"
    else
        fail "$width-bit typedef/archive width test"
        tail -25 "$test_log"
    fi

    if "$build_dir/test_idx_width" >"$TMP_ROOT/identity-$width.log" 2>&1 && \
        grep -Fq "typedef=$width header=$width archive=$width" \
            "$TMP_ROOT/identity-$width.log"; then
        pass "$width-bit generated-header/typedef/archive identity"
    else
        fail "$width-bit generated-header/typedef/archive identity"
        tail -15 "$TMP_ROOT/identity-$width.log"
    fi

    if grep -Fq -- "-DSPARSE_IDX_BITS=$width" "$build_log"; then
        pass "$width-bit compiler flag propagation"
    else
        fail "$width-bit compiler flag propagation"
    fi

    if make -C "$ROOT_DIR" BUILDDIR="$build_dir" SPARSE_IDX_BITS="$width" \
        "$build_dir/test_sparse_matrix" >"$TMP_ROOT/reuse-$width.log" 2>&1; then
        pass "$width-bit same-width tree reuse"
    else
        fail "$width-bit same-width tree reuse"
        tail -15 "$TMP_ROOT/reuse-$width.log"
    fi
done

mixed_binary="$TMP_ROOT/mixed-width"
mixed_log="$TMP_ROOT/mixed-width.log"
if cc -std=c11 -DSPARSE_IDX_BITS=64 -I"$TMP_ROOT/width-64/include" \
    -I"$ROOT_DIR/include" "$ROOT_DIR/tests/test_idx_width.c" \
    "$TMP_ROOT/width-32/libsparse_lu_ortho.a" -lm -o "$mixed_binary" \
    >"$TMP_ROOT/mixed-build.log" 2>&1 && \
    ! "$mixed_binary" >"$mixed_log" 2>&1 && \
    grep -Fq 'typedef=64 header=64 archive=32' "$mixed_log"; then
    pass 'mixed-width identity rejected before index-bearing API use'
else
    fail 'mixed-width identity was not rejected as expected'
    tail -15 "$TMP_ROOT/mixed-build.log" "$mixed_log" 2>/dev/null || true
fi

for invalid_width in 0 16 128 text empty; do
    value="$invalid_width"
    if [ "$value" = empty ]; then value=""; fi
    expect_make_rejection "invalid-$invalid_width" 'SPARSE_IDX_BITS must be 32 or 64' \
        "SPARSE_IDX_BITS=$value"
done

expect_make_rejection conflict 'do not define it in CFLAGS' \
    SPARSE_IDX_BITS=32 'CFLAGS=-std=c11 -DSPARSE_IDX_BITS=64'
expect_make_rejection split-conflict 'do not define it in CFLAGS' \
    SPARSE_IDX_BITS=32 'CFLAGS=-std=c11 -D SPARSE_IDX_BITS=64'
expect_make_rejection compiler-conflict 'do not define it in CC' \
    SPARSE_IDX_BITS=32 'CC=cc -DSPARSE_IDX_BITS=64'

lint_log="$TMP_ROOT/lint-dry-run.log"
if make -C "$ROOT_DIR" -n BUILDDIR="$TMP_ROOT/width-64" SPARSE_IDX_BITS=64 \
    lint >"$lint_log" 2>&1 && \
    grep -F 'clang-tidy ' "$lint_log" | grep -Fq -- '-DSPARSE_IDX_BITS=64' && \
    grep -Fq -- "-I $TMP_ROOT/width-64/include -I include" "$lint_log" && \
    grep -Fq -- '-Werror' "$lint_log"; then
    pass '64-bit lint compile, clang-tidy, and cppcheck flags'
else
    fail '64-bit lint compile, clang-tidy, or cppcheck flags'
    tail -15 "$lint_log"
fi

build_dir="$TMP_ROOT/width-32"
if [ -f "$build_dir/libsparse_lu_ortho.a" ]; then
    archive_before="$(cksum "$build_dir/libsparse_lu_ortho.a")"
    stale_log="$TMP_ROOT/stale.log"
    if make -C "$ROOT_DIR" BUILDDIR="$build_dir" SPARSE_IDX_BITS=64 \
        "$build_dir/test_sparse_matrix" >"$stale_log" 2>&1; then
        fail 'opposite-width build-tree reuse unexpectedly accepted'
    elif grep -Fq 'has SPARSE_IDX_BITS=32, requested 64' "$stale_log" && \
        [ "$(cksum "$build_dir/libsparse_lu_ortho.a")" = "$archive_before" ]; then
        pass 'opposite-width build-tree reuse rejected before archive changes'
    else
        fail 'opposite-width reuse missing diagnostic or changed archive'
        tail -15 "$stale_log"
    fi
fi

for unstamped_kind in object header; do
    unstamped_dir="$TMP_ROOT/unstamped-$unstamped_kind"
    mkdir -p "$unstamped_dir/include"
    if [ "$unstamped_kind" = object ]; then
        touch "$unstamped_dir/sparse_types.o"
    else
        touch "$unstamped_dir/include/sparse_version.h"
    fi
    unstamped_log="$TMP_ROOT/unstamped-$unstamped_kind.log"
    if make -C "$ROOT_DIR" BUILDDIR="$unstamped_dir" SPARSE_IDX_BITS=32 \
        "$unstamped_dir/libsparse_lu_ortho.a" >"$unstamped_log" 2>&1; then
        fail "unstamped old $unstamped_kind unexpectedly accepted"
    elif grep -Fq 'has unstamped build artifacts' "$unstamped_log"; then
        pass "unstamped old $unstamped_kind rejected before build"
    else
        fail "unstamped old $unstamped_kind missing diagnostic"
        tail -15 "$unstamped_log"
    fi
done

printf '\nPassed: %d\nFailed: %d\n' "$PASS" "$FAIL"
if [ "$FAIL" -ne 0 ]; then exit 1; fi
