#!/usr/bin/env bash
# Build and run the selected index-width functional lane in a fresh 64-bit tree.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sparse-functional-64.XXXXXX")"
trap 'rm -rf "$TMP_ROOT"' EXIT
build="$TMP_ROOT/build"
log="$TMP_ROOT/build.log"
ctest_log="$TMP_ROOT/ctest.log"

if ! cmake -S "$ROOT_DIR" -B "$build" -DSPARSE_IDX_BITS=64 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >"$log" 2>&1 || \
    ! cmake --build "$build" --target test_idx_width_functional --parallel 4 \
        >>"$log" 2>&1; then
    tail -40 "$log"
    exit 1
fi

if ! grep -Fxq 'SPARSE_IDX_BITS:STRING=64' "$build/CMakeCache.txt" || \
    ! grep -Fxq '#define SPARSE_BUILT_IDX_BITS 64' \
        "$build/include/sparse_version.h"; then
    echo '64-bit CMake cache or generated header identity missing' >&2
    exit 1
fi

if ! python3 - "$build/compile_commands.json" <<'PY'
import json
import shlex
import sys
from pathlib import Path

commands = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for name in ("sparse_types.c", "test_idx_width_functional.c"):
    matches = [entry for entry in commands if Path(entry["file"]).name == name]
    if len(matches) != 1:
        raise SystemExit(f"expected one compile command for {name}, found {len(matches)}")
    argv = matches[0].get("arguments") or shlex.split(matches[0]["command"])
    if "-DSPARSE_IDX_BITS=64" not in argv:
        raise SystemExit(f"64-bit compile definition missing for {name}")
PY
then
    exit 1
fi

if ! ctest --test-dir "$build" -N -R '^test_idx_width_functional$' \
    >"$ctest_log" 2>&1 || \
    ! grep -Fq 'test_idx_width_functional' "$ctest_log"; then
    cat "$ctest_log"
    exit 1
fi
if ! ctest --test-dir "$build" -V -R '^test_idx_width_functional$' \
    >"$ctest_log" 2>&1; then
    cat "$ctest_log"
    exit 1
fi

if ! grep -Fq 'index width: typedef=64 header=64 archive=64' "$ctest_log"; then
    echo 'selected lane did not run against a 64-bit archive' >&2
    cat "$ctest_log"
    exit 1
fi
for case in test_width_identity test_matrix_insert_and_lu \
    test_matrix_market_roundtrip test_csr_csc_roundtrip \
    test_index_text_boundary; do
    if ! grep -Fq "[PASS] $case" "$ctest_log"; then
        echo "missing selected case: $case" >&2
        cat "$ctest_log"
        exit 1
    fi
done

echo '64-bit CMake cache, generated header, library and test compile flags: PASS'
grep -E 'index width:|\[PASS\] test_|100% tests passed' "$ctest_log"
