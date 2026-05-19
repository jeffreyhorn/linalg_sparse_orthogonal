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
need_cmd python3
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
    python3 - "$REPO_ROOT" "$COMPILE_COMMANDS" "$COVERAGE_NOTES" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
compile_commands_path = Path(sys.argv[2])
coverage_notes_path = Path(sys.argv[3])

entries = json.loads(compile_commands_path.read_text())

all_benchmarks = sorted(path.stem for path in (repo_root / "benchmarks").glob("*.c"))
all_examples = sorted(path.stem for path in (repo_root / "examples").glob("*.c"))

counts = {"src": 0, "tests": 0, "benchmarks": 0, "examples": 0}
seen_benchmarks = set()
seen_examples = set()

for entry in entries:
    file_path = Path(entry["file"])
    if not file_path.is_absolute():
        file_path = Path(entry["directory"]) / file_path
    resolved_path = file_path.resolve()
    try:
        relative_parts = resolved_path.relative_to(repo_root).parts
    except ValueError:
        continue
    if not relative_parts:
        continue
    top_level = relative_parts[0]
    stem = file_path.stem
    if top_level == "src":
        counts["src"] += 1
    if top_level == "tests":
        counts["tests"] += 1
    if top_level == "benchmarks":
        counts["benchmarks"] += 1
        seen_benchmarks.add(stem)
    if top_level == "examples":
        counts["examples"] += 1
        seen_examples.add(stem)

missing_benchmarks = sorted(set(all_benchmarks) - seen_benchmarks)
missing_examples = sorted(set(all_examples) - seen_examples)

lines = [
    "Sprint 33 Day 5 dead-code coverage notes",
    f"compile_commands_json {compile_commands_path}",
    f"src {counts['src']}",
    f"tests {counts['tests']}",
    f"benchmarks {counts['benchmarks']}",
    f"examples {counts['examples']}",
    "",
    "missing_benchmarks",
]
lines.extend(f"- {symbol}" for symbol in missing_benchmarks)
lines.extend(["", "missing_examples"])
lines.extend(f"- {symbol}" for symbol in missing_examples)
coverage_notes_path.write_text("\n".join(lines) + "\n")
PY
}

run_cppcheck() {
    local -a cppcheck_args
    cppcheck_args=(
        cppcheck
        --enable=all
        --quiet
        --std=c11
        --suppress=missingIncludeSystem
        -Iinclude
        -I"${CMAKE_BUILD_DIR}/include"
        -Isrc
    )
    if [ -d "build/include" ] && [ "build/include" != "${CMAKE_BUILD_DIR}/include" ]; then
        cppcheck_args+=(-Ibuild/include)
    fi

    echo ">>> cppcheck"
    "${cppcheck_args[@]}" src/ >"$CPPCHECK_OUT" 2>&1
}

run_xunused() {
    local -a args
    local rc
    args=(xunused)

    if [ "$(uname -s)" = "Darwin" ]; then
        local sdkroot clang_for_resource resource_dir llvm18_prefix
        sdkroot="$(xcrun --sdk macosx --show-sdk-path)"
        llvm18_prefix=""
        if command -v brew >/dev/null 2>&1; then
            llvm18_prefix="$(brew --prefix llvm@18 2>/dev/null || true)"
        fi
        if [ -n "$llvm18_prefix" ] && [ -x "${llvm18_prefix}/bin/clang" ]; then
            clang_for_resource="${llvm18_prefix}/bin/clang"
        else
            clang_for_resource="$(command -v clang)"
        fi
        resource_dir="$("$clang_for_resource" -print-resource-dir)"
        args+=("--extra-arg-before=-isysroot" "--extra-arg-before=${sdkroot}")
        args+=("--extra-arg-before=-resource-dir=${resource_dir}")
    fi

    args+=("$COMPILE_COMMANDS")

    echo ">>> xunused"
    if "${args[@]}" >"$XUNUSED_OUT" 2>&1; then
        return 0
    else
        rc=$?
    fi

    # xunused can return nonzero even after it has processed the compile
    # database and emitted usable raw findings.  Preserve that output for
    # report generation instead of failing the whole dead-code job on the
    # tool's partial-success exit code.
    if grep -q "Processing file " "$XUNUSED_OUT"; then
        echo "deadcode_workflow: xunused exited with status ${rc}; preserving raw output for report generation" >&2
        return 0
    fi

    echo "deadcode_workflow: xunused failed with status ${rc} before producing a usable scan trace" >&2
    return "$rc"
}

write_coverage_notes
run_cppcheck
run_xunused

echo "deadcode_workflow: complete"
echo "  compile database: $COMPILE_COMMANDS"
echo "  cppcheck: $CPPCHECK_OUT"
echo "  xunused: $XUNUSED_OUT"
echo "  coverage notes: $COVERAGE_NOTES"
