#!/usr/bin/env bash
# Exercise native pkg-config installs without using the repository build tree.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sparse-pc-width.XXXXXX")"
TMP_ROOT="$(cd "$TMP_ROOT" && pwd -P)"
trap 'rm -rf "$TMP_ROOT"' EXIT

VERSION="$(tr -d '[:space:]' < "$ROOT_DIR/VERSION")"
PASS=0
FAIL=0
pass() { printf '  [PASS] %s\n' "$1"; PASS=$((PASS + 1)); }
fail() { printf '  [FAIL] %s\n' "$1"; FAIL=$((FAIL + 1)); }

for route in make cmake; do
    for width in 32 64; do
        build="$TMP_ROOT/$route-build-$width"
        prefix="$TMP_ROOT/$route-prefix-$width"
        log="$TMP_ROOT/$route-$width.log"
        if [ "$route" = make ]; then
            if ! make -C "$ROOT_DIR" -j4 BUILDDIR="$build" SPARSE_IDX_BITS="$width" \
                PREFIX="$prefix" install >"$log" 2>&1; then
                fail "$route $width-bit install"
                tail -25 "$log"
                continue
            fi
        elif ! cmake -S "$ROOT_DIR" -B "$build" -DSPARSE_IDX_BITS="$width" \
            -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib >"$log" 2>&1 || \
            ! cmake --build "$build" --target sparse_lu_ortho --parallel 4 >>"$log" 2>&1 || \
            ! cmake --install "$build" >>"$log" 2>&1; then
            fail "$route $width-bit install"
            tail -25 "$log"
            continue
        fi
        pass "$route $width-bit clean-prefix install"

        if python3 "$ROOT_DIR/scripts/check_static_install_contract.py" "$prefix" \
            --mode "$route" --platform unix --version "$VERSION" >>"$log" 2>&1; then
            pass "$route $width-bit header, metadata, and archive identity"
        else
            fail "$route $width-bit static contract"
            tail -25 "$log"
            continue
        fi

        export PKG_CONFIG_LIBDIR="$prefix/lib/pkgconfig"
        export PKG_CONFIG_PATH=""
        cflags="$(pkg-config --cflags sparse)"
        libs="$(pkg-config --libs sparse)"
        pc_prefix="$(pkg-config --variable=prefix sparse)"
        read -r -a cflag_tokens <<< "$cflags"
        read -r -a lib_tokens <<< "$libs"
        if [ "${#cflag_tokens[@]}" -eq 2 ] && \
            [ "${cflag_tokens[0]}" = "-I$prefix/include" ] && \
            [ "${cflag_tokens[1]}" = "-DSPARSE_PACKAGE_IDX_BITS=$width" ] && \
            [ "${#lib_tokens[@]}" -eq 3 ] && \
            [ "${lib_tokens[0]}" = "-L$prefix/lib" ] && \
            [ "${lib_tokens[1]}" = "-lsparse_lu_ortho" ] && \
            [ "${lib_tokens[2]}" = "-lm" ] && \
            [ "$pc_prefix" = "$prefix" ] && \
            [ "$(pkg-config --modversion sparse)" = "$VERSION" ] && \
            ! grep -Fq "$build" "$prefix/lib/pkgconfig/sparse.pc"; then
            pass "$route $width-bit pkg-config flags and static link contract"
        else
            fail "$route $width-bit pkg-config flags: $cflags / $libs"
            continue
        fi

        consumer="$TMP_ROOT/$route-consumer-$width"
        archive="$prefix/lib/libsparse_lu_ortho.a"
        if cc -std=c11 -Wall -Wextra -Werror $cflags \
            "$ROOT_DIR/examples/cmake_example/main.c" $libs -o "$consumer" \
            >>"$log" 2>&1 && "$consumer" >"$TMP_ROOT/$route-run-$width.log" 2>&1 && \
            grep -Fxq "Sparse index width $width" "$TMP_ROOT/$route-run-$width.log" && \
            grep -Fq "Sparse library version $VERSION" "$TMP_ROOT/$route-run-$width.log" && \
            grep -Fxq 'OK' "$TMP_ROOT/$route-run-$width.log"; then
            pass "$route $width-bit pkg-config flags compile, link, and solve"
        else
            fail "$route $width-bit pkg-config consumer"
            tail -25 "$log" "$TMP_ROOT/$route-run-$width.log" 2>/dev/null || true
        fi

        isolated_consumer="$TMP_ROOT/$route-isolated-consumer-$width"
        isolated_run="$TMP_ROOT/$route-isolated-run-$width.log"
        if cc -std=c11 -Wall -Wextra -Werror $cflags \
            "$ROOT_DIR/examples/cmake_example/main.c" "$archive" -lm \
            -o "$isolated_consumer" >>"$log" 2>&1 && \
            "$isolated_consumer" >"$isolated_run" 2>&1 && \
            grep -Fxq "Sparse index width $width" "$isolated_run" && \
            grep -Fq "Sparse library version $VERSION" "$isolated_run" && \
            grep -Fxq 'OK' "$isolated_run"; then
            pass "$route $width-bit isolated absolute-archive consumer solves"
        else
            fail "$route $width-bit isolated absolute-archive consumer"
            tail -25 "$log" "$isolated_run" 2>/dev/null || true
        fi
    done
done

for route in make cmake; do
    other_route=make
    if [ "$route" = make ]; then other_route=cmake; fi
    for width in 32 64; do
        other_width=32
        if [ "$width" -eq 32 ]; then other_width=64; fi
        prefix="$TMP_ROOT/$route-prefix-$width"
        other_prefix="$TMP_ROOT/$other_route-prefix-$other_width"
        log="$TMP_ROOT/$route-mixed-$width.log"
        archive="$prefix/lib/libsparse_lu_ortho.a"
        other_archive="$other_prefix/lib/libsparse_lu_ortho.a"
        if [ ! -f "$archive" ] || [ ! -f "$other_archive" ]; then
            fail "$route $width-bit mixed-prefix prerequisites"
            continue
        fi

        export PKG_CONFIG_LIBDIR="$prefix/lib/pkgconfig"
        export PKG_CONFIG_PATH="$other_prefix/lib/pkgconfig"
        selected_prefix="$(pkg-config --variable=prefix sparse)"
        if [ "$selected_prefix" = "$other_prefix" ] && \
            [ "$selected_prefix" != "$prefix" ]; then
            pass "$route $width-bit foreign PKG_CONFIG_PATH rejected by prefix check"
        else
            fail "$route $width-bit foreign PKG_CONFIG_PATH selected $selected_prefix"
        fi

        export PKG_CONFIG_PATH=""
        cflags="$(pkg-config --cflags sparse)"
        if [ "$(pkg-config --variable=prefix sparse)" != "$prefix" ]; then
            fail "$route $width-bit own pkg-config prefix unavailable"
            continue
        fi

        wrong_header="$TMP_ROOT/$route-wrong-header-$width"
        if cc -std=c11 -Wall -Wextra -Werror -I"$other_prefix/include" $cflags \
            "$ROOT_DIR/examples/cmake_example/main.c" "$archive" -lm \
            -o "$wrong_header" >"$log" 2>&1; then
            fail "$route $width-bit foreign header unexpectedly compiled"
        elif grep -Fq 'SPARSE_PACKAGE_IDX_BITS must match SPARSE_BUILT_IDX_BITS' "$log" && \
            [ ! -e "$wrong_header" ]; then
            pass "$route $width-bit foreign header rejected at compile"
        else
            fail "$route $width-bit foreign header missing width diagnostic"
            tail -25 "$log"
        fi

        wrong_archive="$TMP_ROOT/$route-wrong-archive-$width"
        archive_run="$TMP_ROOT/$route-wrong-archive-run-$width.log"
        if cc -std=c11 -Wall -Wextra -Werror $cflags \
            "$ROOT_DIR/examples/cmake_example/main.c" "$other_archive" -lm \
            -o "$wrong_archive" >"$log" 2>&1; then
            if "$wrong_archive" >"$archive_run" 2>&1; then
                fail "$route $width-bit foreign archive unexpectedly ran solver"
            elif grep -Fxq "Sparse index width mismatch: consumer=$width header=$width archive=$other_width" \
                "$archive_run" && ! grep -Eq 'Sparse library version|Solution:|^OK$' \
                "$archive_run"; then
                pass "$route $width-bit foreign archive rejected before solver use"
            else
                fail "$route $width-bit foreign archive missing pre-use diagnostic"
                tail -25 "$archive_run"
            fi
        else
            fail "$route $width-bit foreign archive did not link"
            tail -25 "$log"
        fi

        wrong_lookup="$TMP_ROOT/$route-wrong-lookup-$width"
        lookup_run="$TMP_ROOT/$route-wrong-lookup-run-$width.log"
        if cc -std=c11 -Wall -Wextra -Werror $cflags \
            "$ROOT_DIR/examples/cmake_example/main.c" \
            -L"$other_prefix/lib" -L"$prefix/lib" -lsparse_lu_ortho -lm \
            -o "$wrong_lookup" >"$log" 2>&1; then
            if "$wrong_lookup" >"$lookup_run" 2>&1; then
                fail "$route $width-bit foreign library search path unexpectedly ran solver"
            elif grep -Fxq "Sparse index width mismatch: consumer=$width header=$width archive=$other_width" \
                "$lookup_run" && ! grep -Eq 'Sparse library version|Solution:|^OK$' \
                "$lookup_run"; then
                pass "$route $width-bit foreign library search path rejected before solver use"
            else
                fail "$route $width-bit foreign library search path missing pre-use diagnostic"
                tail -25 "$lookup_run"
            fi
        else
            fail "$route $width-bit foreign library search path did not link"
            tail -25 "$log"
        fi

        missing_archive="$prefix/lib/not-installed.a"
        missing_consumer="$TMP_ROOT/$route-missing-archive-$width"
        if cc -std=c11 -Wall -Wextra -Werror $cflags \
            "$ROOT_DIR/examples/cmake_example/main.c" "$missing_archive" -lm \
            -o "$missing_consumer" >"$log" 2>&1; then
            fail "$route $width-bit missing archive unexpectedly linked"
        elif grep -Fq "$missing_archive" "$log" && [ ! -e "$missing_consumer" ]; then
            pass "$route $width-bit absolute missing archive rejected at link"
        else
            fail "$route $width-bit missing archive failed without path diagnostic"
            tail -25 "$log"
        fi
    done
done

printf '\nPassed: %d\nFailed: %d\n' "$PASS" "$FAIL"
if [ "$FAIL" -ne 0 ]; then exit 1; fi
