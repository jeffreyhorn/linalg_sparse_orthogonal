#!/usr/bin/env bash
# Verify width-aware imported targets from relocated CMake installations.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(tr -d '[:space:]' < "$ROOT_DIR/VERSION")"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sparse-cmake-width.XXXXXX")"
TMP_ROOT="$(cd "$TMP_ROOT" && pwd -P)"
trap 'rm -rf "$TMP_ROOT"' EXIT

PASS=0
FAIL=0
pass() { printf '  [PASS] %s\n' "$1"; PASS=$((PASS + 1)); }
fail() { printf '  [FAIL] %s\n' "$1"; FAIL=$((FAIL + 1)); }

for width in 32 64; do
    other_width=32
    if [ "$width" -eq 32 ]; then other_width=64; fi

    build="$TMP_ROOT/library-$width"
    prefix="$TMP_ROOT/install-$width"
    relocated="$TMP_ROOT/relocated-$width"
    consumer="$TMP_ROOT/consumer-$width"
    conflict="$TMP_ROOT/conflict-$width"
    log="$TMP_ROOT/build-$width.log"

    if cmake -S "$ROOT_DIR" -B "$build" -DSPARSE_IDX_BITS="$width" \
        -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib \
        >"$log" 2>&1 && \
        cmake --build "$build" --target sparse_lu_ortho --parallel 4 >>"$log" 2>&1 && \
        cmake --install "$build" >>"$log" 2>&1 && \
        python3 "$ROOT_DIR/scripts/check_static_install_contract.py" "$prefix" \
            --mode cmake --platform unix --version "$VERSION" >>"$log" 2>&1; then
        pass "$width-bit CMake library and clean-prefix install"
    else
        fail "$width-bit CMake library or install"
        tail -25 "$log"
        continue
    fi

    mv "$prefix" "$relocated"
    target_file="$relocated/lib/cmake/Sparse/SparseTargets.cmake"
    header="$relocated/include/sparse/sparse_version.h"
    if grep -Fq "INTERFACE_COMPILE_DEFINITIONS \"SPARSE_PACKAGE_IDX_BITS=$width\"" \
        "$target_file" && \
        grep -Fxq "#define SPARSE_BUILT_IDX_BITS $width" "$header" && \
        ! grep -Fq "$ROOT_DIR" "$target_file" && \
        ! grep -Fq "$build" "$target_file"; then
        pass "$width-bit exported width and relocatable target metadata"
    else
        fail "$width-bit exported target/header metadata"
    fi

    if cmake -S "$ROOT_DIR/tests/cmake_width_consumer" -B "$consumer" \
        -DSparse_DIR:PATH="$relocated/lib/cmake/Sparse" \
        -DSPARSE_TEST_VERSION="$VERSION" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >"$TMP_ROOT/consumer-$width.log" 2>&1 && \
        cmake --build "$consumer" >>"$TMP_ROOT/consumer-$width.log" 2>&1 && \
        "$consumer/width_consumer" >"$TMP_ROOT/run-$width.log" 2>&1 && \
        grep -Fq "SPARSE_PACKAGE_IDX_BITS=$width" "$consumer/compile_commands.json" && \
        grep -Fxq "Sparse index width $width" "$TMP_ROOT/run-$width.log" && \
        grep -Fq "Sparse library version $VERSION" "$TMP_ROOT/run-$width.log" && \
        grep -Fxq 'OK' "$TMP_ROOT/run-$width.log" && \
        grep -Fq "Sparse_DIR:PATH=$relocated/lib/cmake/Sparse" "$consumer/CMakeCache.txt"; then
        pass "$width-bit relocated consumer uses target width and solves"
    else
        fail "$width-bit relocated consumer"
        tail -25 "$TMP_ROOT/consumer-$width.log" "$TMP_ROOT/run-$width.log" 2>/dev/null || true
    fi

    conflict_log="$TMP_ROOT/conflict-$width.log"
    if cmake -S "$ROOT_DIR/examples/cmake_example" -B "$conflict" \
        -DSparse_DIR:PATH="$relocated/lib/cmake/Sparse" \
        "-DCMAKE_C_FLAGS=-DSPARSE_IDX_BITS=$other_width" >"$conflict_log" 2>&1 && \
        cmake --build "$conflict" >>"$conflict_log" 2>&1; then
        fail "$width-bit conflicting consumer unexpectedly built"
    elif grep -Fq 'SPARSE_IDX_BITS must match SPARSE_BUILT_IDX_BITS' "$conflict_log"; then
        pass "$width-bit conflicting consumer rejected at compile"
    else
        fail "$width-bit conflicting consumer missing width diagnostic"
        tail -25 "$conflict_log"
    fi

    version_log="$TMP_ROOT/version-mismatch-$width.log"
    if cmake -S "$ROOT_DIR/tests/cmake_width_consumer" \
        -B "$TMP_ROOT/version-mismatch-$width" \
        -DSparse_DIR:PATH="$relocated/lib/cmake/Sparse" \
        -DSPARSE_TEST_VERSION=2.3.0 >"$version_log" 2>&1; then
        fail "$width-bit wrong exact package version unexpectedly configured"
    elif grep -Fq 'Could not find a configuration file for package "Sparse"' \
        "$version_log" && grep -Fq '2.3.0' "$version_log" && \
        grep -Fq "version: $VERSION" "$version_log"; then
        pass "$width-bit exact package version mismatch rejected at configure"
    else
        fail "$width-bit exact version rejection missing diagnostic"
        tail -25 "$version_log"
    fi
done

for width in 32 64; do
    other_width=32
    if [ "$width" -eq 32 ]; then other_width=64; fi
    prefix="$TMP_ROOT/relocated-$width"
    other_prefix="$TMP_ROOT/relocated-$other_width"
    if [ ! -f "$prefix/lib/libsparse_lu_ortho.a" ] || \
       [ ! -f "$other_prefix/lib/libsparse_lu_ortho.a" ]; then
        fail "$width-bit mixed-prefix prerequisites"
        continue
    fi

    target_conflict="$TMP_ROOT/target-conflict-$width"
    target_log="$TMP_ROOT/target-conflict-$width.log"
    if cmake -S "$ROOT_DIR/tests/cmake_width_consumer" -B "$target_conflict" \
        -DSparse_DIR:PATH="$prefix/lib/cmake/Sparse" \
        -DSPARSE_TEST_VERSION="$VERSION" \
        -DSPARSE_TEST_CONSUMER_BITS="$other_width" >"$target_log" 2>&1 && \
        cmake --build "$target_conflict" >>"$target_log" 2>&1; then
        fail "$width-bit target-level width conflict unexpectedly built"
    elif grep -Fq 'SPARSE_IDX_BITS must match SPARSE_BUILT_IDX_BITS' "$target_log"; then
        pass "$width-bit target-level width conflict rejected at compile"
    else
        fail "$width-bit target-level conflict missing width diagnostic"
        tail -25 "$target_log"
    fi

    wrong_header="$TMP_ROOT/wrong-header-$width"
    header_log="$TMP_ROOT/wrong-header-$width.log"
    if cmake -S "$ROOT_DIR/tests/cmake_width_consumer" -B "$wrong_header" \
        -DSparse_DIR:PATH="$prefix/lib/cmake/Sparse" \
        -DSPARSE_TEST_VERSION="$VERSION" \
        "-DCMAKE_C_FLAGS=-I$other_prefix/include" >"$header_log" 2>&1 && \
        cmake --build "$wrong_header" >>"$header_log" 2>&1; then
        fail "$width-bit foreign header path unexpectedly built"
    elif grep -Fq 'SPARSE_PACKAGE_IDX_BITS must match SPARSE_BUILT_IDX_BITS' \
        "$header_log"; then
        pass "$width-bit foreign header rejected at compile"
    else
        fail "$width-bit foreign header missing width diagnostic"
        tail -25 "$header_log"
    fi

    wrong_archive="$TMP_ROOT/wrong-archive-$width"
    archive_log="$TMP_ROOT/wrong-archive-$width.log"
    archive_run="$TMP_ROOT/wrong-archive-run-$width.log"
    other_archive="$other_prefix/lib/libsparse_lu_ortho.a"
    if cmake -S "$ROOT_DIR/tests/cmake_width_consumer" -B "$wrong_archive" \
        -DSparse_DIR:PATH="$prefix/lib/cmake/Sparse" \
        -DSPARSE_TEST_VERSION="$VERSION" \
        -DSPARSE_TEST_ARCHIVE_OVERRIDE:FILEPATH="$other_archive" \
        >"$archive_log" 2>&1 && \
        cmake --build "$wrong_archive" --verbose >>"$archive_log" 2>&1 && \
        grep -Fq "$other_archive" "$archive_log"; then
        if "$wrong_archive/width_consumer" >"$archive_run" 2>&1; then
            fail "$width-bit foreign archive unexpectedly ran solver"
        elif grep -Fxq "Sparse index width mismatch: consumer=$width header=$width archive=$other_width" \
            "$archive_run" && ! grep -Eq 'Sparse library version|Solution:|^OK$' \
            "$archive_run"; then
            pass "$width-bit foreign archive rejected before solver use"
        else
            fail "$width-bit foreign archive missing pre-use diagnostic"
            tail -25 "$archive_run"
        fi
    else
        fail "$width-bit foreign archive did not link from intended path"
        tail -25 "$archive_log"
    fi

    stale="$TMP_ROOT/stale-cache-$width"
    stale_log="$TMP_ROOT/stale-cache-$width.log"
    if cmake -S "$ROOT_DIR/tests/cmake_width_consumer" -B "$stale" \
        -DCMAKE_PREFIX_PATH="$prefix" -DSPARSE_TEST_VERSION="$VERSION" \
        >"$stale_log" 2>&1 && \
        cmake --build "$stale" >>"$stale_log" 2>&1 && \
        "$stale/width_consumer" >"$TMP_ROOT/stale-first-$width.log" 2>&1 && \
        grep -Fxq "Sparse index width $width" "$TMP_ROOT/stale-first-$width.log" && \
        grep -Fq "Sparse_DIR:PATH=$prefix/lib/cmake/Sparse" "$stale/CMakeCache.txt"; then
        if cmake -S "$ROOT_DIR/tests/cmake_width_consumer" -B "$stale" \
            -DCMAKE_PREFIX_PATH="$other_prefix" \
            -DSPARSE_TEST_CONSUMER_BITS="$other_width" >>"$stale_log" 2>&1 && \
            cmake --build "$stale" --clean-first >>"$stale_log" 2>&1; then
            if grep -Fq "Sparse_DIR:PATH=$other_prefix/lib/cmake/Sparse" \
                "$stale/CMakeCache.txt" && \
                "$stale/width_consumer" >"$TMP_ROOT/stale-rebound-$width.log" 2>&1 && \
                grep -Fxq "Sparse index width $other_width" \
                    "$TMP_ROOT/stale-rebound-$width.log" && \
                grep -Fxq 'OK' "$TMP_ROOT/stale-rebound-$width.log"; then
                pass "$width-bit cache rebound to opposite package safely"
            else
                fail "$width-bit reconfigured consumer built without verified package rebind"
                tail -25 "$stale_log" "$TMP_ROOT/stale-rebound-$width.log" 2>/dev/null || true
                grep -F 'Sparse_DIR:' "$stale/CMakeCache.txt" || true
            fi
        elif grep -Fq "Sparse_DIR:PATH=$prefix/lib/cmake/Sparse" \
            "$stale/CMakeCache.txt" && \
            grep -Fq 'SPARSE_IDX_BITS must match SPARSE_BUILT_IDX_BITS' "$stale_log"; then
            pass "$width-bit stale cache rejected at compile before use"
        else
            fail "$width-bit stale cache missing expected diagnostic"
            tail -25 "$stale_log"
        fi
    else
        fail "$width-bit initial cache consumer did not use its own package"
        tail -25 "$stale_log" "$TMP_ROOT/stale-first-$width.log" 2>/dev/null || true
    fi
done

printf '\nPassed: %d\nFailed: %d\n' "$PASS" "$FAIL"
if [ "$FAIL" -ne 0 ]; then exit 1; fi
