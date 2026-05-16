#!/usr/bin/env bash
#
# epic3_warning_workflow.sh — Sprint 30 Day 7 reproducible warning
# capture + validation workflow for Epic 3 compile-hygiene work.
#
# Usage:
#   bash scripts/epic3_warning_workflow.sh <label> [artifacts_dir]
#
# Example:
#   bash scripts/epic3_warning_workflow.sh day7-workflow \
#     docs/planning/EPIC_3/SPRINT_30/artifacts
#
# Runs a clean CMake configure/build/ctest cycle plus a clean Makefile
# library build, captures stdout/stderr into stable artifact files, and
# derives warning-count summaries from the CMake build stderr.

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 <label> [artifacts_dir]" >&2
    exit 2
fi

LABEL="$1"
ARTIFACTS_DIR="${2:-docs/planning/EPIC_3/SPRINT_30/artifacts}"

case "$LABEL" in
    ""|*[!A-Za-z0-9._-]*)
        echo "epic3_warning_workflow: label must match [A-Za-z0-9._-]+" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CMAKE_BUILD_DIR="build/${LABEL}-cmake"
MAKE_BUILD_DIR="build/${LABEL}-make"
CMAKE_BUILD_JOBS="${WARNING_WORKFLOW_JOBS:-1}"

case "$CMAKE_BUILD_JOBS" in
    ''|*[!0-9]*)
        echo "epic3_warning_workflow: WARNING_WORKFLOW_JOBS must be a positive integer" >&2
        exit 2
        ;;
    0)
        echo "epic3_warning_workflow: WARNING_WORKFLOW_JOBS must be greater than zero" >&2
        exit 2
        ;;
esac

mkdir -p "$ARTIFACTS_DIR"
rm -rf "$CMAKE_BUILD_DIR" "$MAKE_BUILD_DIR"

run_step() {
    local step_name="$1"
    local stdout_file="$2"
    local stderr_file="$3"
    shift 3

    echo ">>> ${step_name}"
    "$@" >"$stdout_file" 2>"$stderr_file"
}

warning_count() {
    local path="$1"
    awk '/warning:/{count++} END{print count + 0}' "$path"
}

count_by_area() {
    local path="$1"
    awk '
        /warning:/ {
            file = $1
            area = "other"
            if (file ~ /\/src\//) {
                area = "src"
            } else if (file ~ /\/tests\//) {
                area = "tests"
            } else if (file ~ /\/benchmarks\//) {
                area = "benchmarks"
            } else if (file ~ /\/examples\//) {
                area = "examples"
            }
            counts[area]++
        }
        END {
            for (area in counts) {
                printf "%s\t%d\n", area, counts[area]
            }
        }
    ' "$path" | sort
}

count_by_class() {
    local path="$1"
    awk '
        /warning:/ {
            if (match($0, /\[-W[^]]+\]/)) {
                cls = substr($0, RSTART + 1, RLENGTH - 2)
            } else {
                cls = "unknown"
            }
            counts[cls]++
        }
        END {
            for (cls in counts) {
                printf "%s\t%d\n", cls, counts[cls]
            }
        }
    ' "$path" | sort
}

count_by_file() {
    local path="$1"
    awk '
        /warning:/ {
            counts[$1]++
        }
        END {
            for (file in counts) {
                printf "%s\t%d\n", file, counts[file]
            }
        }
    ' "$path" | sort
}

cmake_configure_stdout="${ARTIFACTS_DIR}/${LABEL}-cmake-configure.stdout.txt"
cmake_configure_stderr="${ARTIFACTS_DIR}/${LABEL}-cmake-configure.stderr.txt"
cmake_build_stdout="${ARTIFACTS_DIR}/${LABEL}-cmake-build.stdout.txt"
cmake_build_stderr="${ARTIFACTS_DIR}/${LABEL}-cmake-build.stderr.txt"
ctest_stdout="${ARTIFACTS_DIR}/${LABEL}-ctest.stdout.txt"
ctest_stderr="${ARTIFACTS_DIR}/${LABEL}-ctest.stderr.txt"
make_build_stdout="${ARTIFACTS_DIR}/${LABEL}-make-build.stdout.txt"
make_build_stderr="${ARTIFACTS_DIR}/${LABEL}-make-build.stderr.txt"

area_counts_file="${ARTIFACTS_DIR}/${LABEL}-cmake-warning-counts-by-area.txt"
class_counts_file="${ARTIFACTS_DIR}/${LABEL}-cmake-warning-counts-by-class.txt"
file_counts_file="${ARTIFACTS_DIR}/${LABEL}-cmake-warning-counts-by-file.txt"
summary_md="${ARTIFACTS_DIR}/${LABEL}-workflow-summary.md"

run_step "CMake configure" "$cmake_configure_stdout" "$cmake_configure_stderr" \
    cmake -S . -B "$CMAKE_BUILD_DIR"
run_step "CMake build" "$cmake_build_stdout" "$cmake_build_stderr" \
    cmake --build "$CMAKE_BUILD_DIR" --parallel "$CMAKE_BUILD_JOBS"
run_step "ctest" "$ctest_stdout" "$ctest_stderr" \
    ctest --test-dir "$CMAKE_BUILD_DIR" --output-on-failure
run_step "Makefile library build" "$make_build_stdout" "$make_build_stderr" \
    make "BUILDDIR=$MAKE_BUILD_DIR" all

cmake_configure_warning_count="$(warning_count "$cmake_configure_stderr")"
cmake_build_warning_count="$(warning_count "$cmake_build_stderr")"
make_build_warning_count="$(warning_count "$make_build_stderr")"

count_by_area "$cmake_build_stderr" >"$area_counts_file"
count_by_class "$cmake_build_stderr" >"$class_counts_file"
count_by_file "$cmake_build_stderr" >"$file_counts_file"

cat >"$summary_md" <<EOF_SUMMARY
# Epic 3 Warning Workflow Summary: ${LABEL}

**Date:** $(date +%Y-%m-%d)  
**Repository root:** \`${REPO_ROOT}\`

## Build Paths

- CMake build directory: \`${CMAKE_BUILD_DIR}\`
- Makefile build directory: \`${MAKE_BUILD_DIR}\`
- Artifact directory: \`${ARTIFACTS_DIR}\`
- CMake build jobs: \`${CMAKE_BUILD_JOBS}\`

## Commands Run

1. \`cmake -S . -B ${CMAKE_BUILD_DIR}\`
2. \`cmake --build ${CMAKE_BUILD_DIR} --parallel ${CMAKE_BUILD_JOBS}\`
3. \`ctest --test-dir ${CMAKE_BUILD_DIR} --output-on-failure\`
4. \`make BUILDDIR=${MAKE_BUILD_DIR} all\`

## Warning Counts

- CMake configure warnings: \`${cmake_configure_warning_count}\`
- CMake build warnings: \`${cmake_build_warning_count}\`
- Makefile build warnings: \`${make_build_warning_count}\`

## Derived Summaries

- \`${LABEL}-cmake-warning-counts-by-area.txt\`
- \`${LABEL}-cmake-warning-counts-by-class.txt\`
- \`${LABEL}-cmake-warning-counts-by-file.txt\`

## Raw Logs

- \`${LABEL}-cmake-configure.stdout.txt\`
- \`${LABEL}-cmake-configure.stderr.txt\`
- \`${LABEL}-cmake-build.stdout.txt\`
- \`${LABEL}-cmake-build.stderr.txt\`
- \`${LABEL}-ctest.stdout.txt\`
- \`${LABEL}-ctest.stderr.txt\`
- \`${LABEL}-make-build.stdout.txt\`
- \`${LABEL}-make-build.stderr.txt\`

## Notes

- The CMake path is the authoritative full-tree warning inventory for Sprint 30.
- The Makefile \`all\` path remains a library-only cross-check unless a wider Makefile target is used.
- The workflow defaults to a serialized CMake warning capture (\`WARNING_WORKFLOW_JOBS=1\`) so area/file attribution remains stable and non-interleaved.
- Re-run this workflow with a new label before and after edits to compare summaries without overwriting prior artifacts.
EOF_SUMMARY

echo "epic3_warning_workflow: complete"
echo "  summary: $summary_md"
echo "  cmake warnings: $cmake_build_warning_count"
echo "  make warnings: $make_build_warning_count"
