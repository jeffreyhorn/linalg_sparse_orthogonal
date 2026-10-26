#!/usr/bin/env bash
# Verify the maintained consumers query archive width before index-bearing use.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sparse-width-check.XXXXXX")"
TMP_ROOT="$(cd "$TMP_ROOT" && pwd -P)"
trap 'rm -rf "$TMP_ROOT"' EXIT

VERSION="$(tr -d '[:space:]' < "$ROOT_DIR/VERSION")"
PASS=0
FAIL=0
pass() { printf '  [PASS] %s\n' "$1"; PASS=$((PASS + 1)); }
fail() { printf '  [FAIL] %s\n' "$1"; FAIL=$((FAIL + 1)); }

for width in 32 64; do
    build="$TMP_ROOT/library-$width"
    prefix="$TMP_ROOT/install-$width"
    consumer="$TMP_ROOT/cmake-consumer-$width"
    log="$TMP_ROOT/build-$width.log"
    archive="$prefix/lib/libsparse_lu_ortho.a"
    header="$prefix/include/sparse/sparse_version.h"

    if cmake -S "$ROOT_DIR" -B "$build" -DSPARSE_IDX_BITS="$width" \
        -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib >"$log" 2>&1 && \
        cmake --build "$build" --target sparse_lu_ortho --parallel 4 >>"$log" 2>&1 && \
        cmake --install "$build" >>"$log" 2>&1 && \
        python3 "$ROOT_DIR/scripts/check_static_install_contract.py" "$prefix" \
            --mode cmake --platform unix --version "$VERSION" >>"$log" 2>&1; then
        pass "$width-bit clean-prefix install and archive identity"
        printf '    header: %s\n    archive: %s\n' "$header" "$archive"
    else
        fail "$width-bit install or validator"
        tail -25 "$log"
        continue
    fi

    if cmake -S "$ROOT_DIR/examples/cmake_example" -B "$consumer" \
        -DSparse_DIR:PATH="$prefix/lib/cmake/Sparse" >"$log" 2>&1 && \
        cmake --build "$consumer" >>"$log" 2>&1 && \
        "$consumer/example" >"$TMP_ROOT/cmake-run-$width.log" 2>&1 && \
        grep -Fxq "Sparse index width $width" "$TMP_ROOT/cmake-run-$width.log" && \
        grep -Fxq 'OK' "$TMP_ROOT/cmake-run-$width.log" && \
        grep -Fq "Sparse_DIR:PATH=$prefix/lib/cmake/Sparse" "$consumer/CMakeCache.txt"; then
        pass "$width-bit CMake consumer checks width then solves"
    else
        fail "$width-bit CMake consumer"
        tail -25 "$log" "$TMP_ROOT/cmake-run-$width.log" 2>/dev/null || true
    fi

    export PKG_CONFIG_LIBDIR="$prefix/lib/pkgconfig"
    export PKG_CONFIG_PATH=""
    cflags="$(pkg-config --cflags sparse)"
    libs="$(pkg-config --libs sparse)"
    read -r -a cflag_tokens <<< "$cflags"
    pkg_consumer="$TMP_ROOT/pkg-consumer-$width"
    if [ "${#cflag_tokens[@]}" -eq 2 ] && \
        [ "${cflag_tokens[0]}" = "-I$prefix/include" ] && \
        [ "${cflag_tokens[1]}" = "-DSPARSE_PACKAGE_IDX_BITS=$width" ] && \
        cc -std=c11 -Wall -Wextra -Werror $cflags \
            "$ROOT_DIR/examples/cmake_example/main.c" $libs -o "$pkg_consumer" \
            >"$log" 2>&1 && \
        "$pkg_consumer" >"$TMP_ROOT/pkg-run-$width.log" 2>&1 && \
        grep -Fxq "Sparse index width $width" "$TMP_ROOT/pkg-run-$width.log" && \
        grep -Fxq 'OK' "$TMP_ROOT/pkg-run-$width.log"; then
        pass "$width-bit pkg-config consumer checks width then solves"
    else
        fail "$width-bit pkg-config consumer"
        tail -25 "$log" "$TMP_ROOT/pkg-run-$width.log" 2>/dev/null || true
    fi
done

for width in 32 64; do
    other_width=32
    if [ "$width" -eq 32 ]; then other_width=64; fi
    prefix="$TMP_ROOT/install-$width"
    archive="$prefix/lib/libsparse_lu_ortho.a"
    other_archive="$TMP_ROOT/install-$other_width/lib/libsparse_lu_ortho.a"
    if [ ! -f "$archive" ] || [ ! -f "$other_archive" ]; then
        fail "$width-bit archive swap prerequisites"
        continue
    fi

    mv "$archive" "$TMP_ROOT/original-$width.a"
    cp "$other_archive" "$archive"
    validator_log="$TMP_ROOT/swapped-validator-$width.log"
    if python3 "$ROOT_DIR/scripts/check_static_install_contract.py" "$prefix" \
        --mode cmake --platform unix --version "$VERSION" >"$validator_log" 2>&1; then
        fail "$width-bit swapped archive unexpectedly passed validator"
    elif grep -Fq 'installed index widths disagree' "$validator_log"; then
        pass "$width-bit swapped archive rejected by package validator"
    else
        fail "$width-bit swapped archive validator diagnostic"
        tail -25 "$validator_log"
    fi

    consumer="$TMP_ROOT/swapped-cmake-$width"
    build_log="$TMP_ROOT/swapped-cmake-build-$width.log"
    run_log="$TMP_ROOT/swapped-cmake-run-$width.log"
    if cmake -S "$ROOT_DIR/examples/cmake_example" -B "$consumer" \
        -DSparse_DIR:PATH="$prefix/lib/cmake/Sparse" >"$build_log" 2>&1 && \
        cmake --build "$consumer" >>"$build_log" 2>&1; then
        if "$consumer/example" >"$run_log" 2>&1; then
            fail "$width-bit CMake consumer accepted swapped archive"
        elif grep -Fxq "Sparse index width mismatch: consumer=$width header=$width archive=$other_width" \
            "$run_log" && ! grep -Eq 'Sparse library version|Solution:|^OK$' "$run_log"; then
            pass "$width-bit CMake consumer rejects swap before solver use"
        else
            fail "$width-bit CMake consumer missing pre-use diagnostic"
            tail -25 "$run_log"
        fi
    else
        fail "$width-bit swapped CMake consumer did not compile and link"
        tail -25 "$build_log"
    fi

    export PKG_CONFIG_LIBDIR="$prefix/lib/pkgconfig"
    export PKG_CONFIG_PATH=""
    cflags="$(pkg-config --cflags sparse)"
    libs="$(pkg-config --libs sparse)"
    pkg_consumer="$TMP_ROOT/swapped-pkg-$width"
    build_log="$TMP_ROOT/swapped-pkg-build-$width.log"
    run_log="$TMP_ROOT/swapped-pkg-run-$width.log"
    if cc -std=c11 -Wall -Wextra -Werror $cflags \
        "$ROOT_DIR/examples/cmake_example/main.c" $libs -o "$pkg_consumer" \
        >"$build_log" 2>&1; then
        if "$pkg_consumer" >"$run_log" 2>&1; then
            fail "$width-bit pkg-config consumer accepted swapped archive"
        elif grep -Fxq "Sparse index width mismatch: consumer=$width header=$width archive=$other_width" \
            "$run_log" && ! grep -Eq 'Sparse library version|Solution:|^OK$' "$run_log"; then
            pass "$width-bit pkg-config consumer rejects swap before solver use"
        else
            fail "$width-bit pkg-config consumer missing pre-use diagnostic"
            tail -25 "$run_log"
        fi
    else
        fail "$width-bit swapped pkg-config consumer did not compile and link"
        tail -25 "$build_log"
    fi

    mv "$TMP_ROOT/original-$width.a" "$archive"
done

printf '\nPassed: %d\nFailed: %d\n' "$PASS" "$FAIL"
if [ "$FAIL" -ne 0 ]; then exit 1; fi
