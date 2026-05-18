#!/usr/bin/env bash
#
# deadcode_workflow.sh — Sprint 33 Day 5 raw dead-code workflow.
#
# Usage:
#   bash scripts/deadcode_workflow.sh <cmake_build_dir> <artifacts_dir>
#
# Generates or refreshes the raw dead-code evidence artifacts:
#   - compile_commands.json path is validated
#   - cppcheck raw output
#   - xunused raw output
#   - compile-db coverage notes for the bench/example gap

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <cmake_build_dir> <artifacts_dir>" >&2
    exit 2
fi

CMAKE_BUILD_DIR="$1"
ARTIFACTS_DIR="$2"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

COMPILE_COMMANDS="${CMAKE_BUILD_DIR}/compile_commands.json"
CPPCHECK_OUT="${ARTIFACTS_DIR}/cppcheck.txt"
XUNUSED_OUT="${ARTIFACTS_DIR}/xunused.txt"
COVERAGE_NOTES="${ARTIFACTS_DIR}/coverage-notes.txt"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "deadcode_workflow: required command '$1' not found in PATH" >&2
        exit 1
    fi
}

need_cmd cmake
need_cmd cppcheck
need_cmd xunused
if [ "$(uname -s)" = "Darwin" ]; then
    need_cmd xcrun
fi

mkdir -p "$ARTIFACTS_DIR"

if [ ! -f "$COMPILE_COMMANDS" ]; then
    echo "deadcode_workflow: missing compile database: $COMPILE_COMMANDS" >&2
    echo "run 'make deadcode-compile-db' first" >&2
    exit 1
fi

write_coverage_notes() {
    local tmp_bench_all tmp_bench_seen tmp_ex_all tmp_ex_seen
    tmp_bench_all="$(mktemp)"
    tmp_bench_seen="$(mktemp)"
    tmp_ex_all="$(mktemp)"
    tmp_ex_seen="$(mktemp)"
    trap 'rm -f "$tmp_bench_all" "$tmp_bench_seen" "$tmp_ex_all" "$tmp_ex_seen"' RETURN

    find benchmarks -maxdepth 1 -name '*.c' -print \
        | sed 's#^benchmarks/##; s#\.c$##' \
        | sort >"$tmp_bench_all"
    grep '"file":' "$COMPILE_COMMANDS" \
        | sed -n 's#.*"/.*/benchmarks/\([^"]*\)\.c".*#\1#p' \
        | sort -u >"$tmp_bench_seen"

    find examples -maxdepth 1 -name '*.c' -print \
        | sed 's#^examples/##; s#\.c$##' \
        | sort >"$tmp_ex_all"
    grep '"file":' "$COMPILE_COMMANDS" \
        | sed -n 's#.*"/.*/examples/\([^"]*\)\.c".*#\1#p' \
        | sort -u >"$tmp_ex_seen"

    {
        echo "Sprint 33 Day 5 dead-code coverage notes"
        echo "compile_commands_json $COMPILE_COMMANDS"
        echo "src $(grep '"file":' "$COMPILE_COMMANDS" | grep -c '/src/')"
        echo "tests $(grep '"file":' "$COMPILE_COMMANDS" | grep -c '/tests/')"
        echo "benchmarks $(grep '"file":' "$COMPILE_COMMANDS" | grep -c '/benchmarks/')"
        echo "examples $(grep '"file":' "$COMPILE_COMMANDS" | grep -c '/examples/')"
        echo
        echo "missing_benchmarks"
        if comm -23 "$tmp_bench_all" "$tmp_bench_seen" | sed 's/^/- /'; then
            :
        fi
        echo
        echo "missing_examples"
        if comm -23 "$tmp_ex_all" "$tmp_ex_seen" | sed 's/^/- /'; then
            :
        fi
    } >"$COVERAGE_NOTES"
}

run_cppcheck() {
    echo ">>> cppcheck"
    cppcheck \
        --enable=all \
        --quiet \
        --std=c11 \
        --suppress=missingIncludeSystem \
        -Iinclude \
        -Ibuild/include \
        -Isrc \
        src/ >"$CPPCHECK_OUT" 2>&1
}

run_xunused() {
    local -a args
    args=(xunused)

    if [ "$(uname -s)" = "Darwin" ]; then
        local sdkroot clang_for_resource resource_dir
        sdkroot="$(xcrun --sdk macosx --show-sdk-path)"
        if command -v brew >/dev/null 2>&1 && [ -x "$(brew --prefix llvm@18 2>/dev/null)/bin/clang" ]; then
            clang_for_resource="$(brew --prefix llvm@18)/bin/clang"
        else
            clang_for_resource="$(command -v clang)"
        fi
        resource_dir="$("$clang_for_resource" -print-resource-dir)"
        args+=("--extra-arg-before=-isysroot" "--extra-arg-before=${sdkroot}")
        args+=("--extra-arg-before=-resource-dir=${resource_dir}")
    fi

    args+=("$COMPILE_COMMANDS")

    echo ">>> xunused"
    "${args[@]}" >"$XUNUSED_OUT" 2>&1
}

write_coverage_notes
run_cppcheck
run_xunused

echo "deadcode_workflow: complete"
echo "  compile database: $COMPILE_COMMANDS"
echo "  cppcheck: $CPPCHECK_OUT"
echo "  xunused: $XUNUSED_OUT"
echo "  coverage notes: $COVERAGE_NOTES"
