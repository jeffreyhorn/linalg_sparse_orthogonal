# Sprint 30 Day 7 Rebuild Workflow Automation

**Date:** 2026-05-16  
**Branch:** `sprint-30`

## Objective

Create a repeatable local workflow for Epic 3 warning-baseline reproduction and post-edit validation so maintainers no longer need to reconstruct the Sprint 30 command sequence by hand.

## Deliverables

Added:

- `scripts/epic3_warning_workflow.sh`
- `Makefile` target: `warning-workflow`
- `docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`

Produced by the workflow validation run:

- `artifacts/day7-workflow-workflow-summary.md`
- `artifacts/day7-workflow-cmake-configure.stdout.txt`
- `artifacts/day7-workflow-cmake-configure.stderr.txt`
- `artifacts/day7-workflow-cmake-build.stdout.txt`
- `artifacts/day7-workflow-cmake-build.stderr.txt`
- `artifacts/day7-workflow-ctest.stdout.txt`
- `artifacts/day7-workflow-ctest.stderr.txt`
- `artifacts/day7-workflow-make-build.stdout.txt`
- `artifacts/day7-workflow-make-build.stderr.txt`
- `artifacts/day7-workflow-cmake-warning-counts-by-area.txt`
- `artifacts/day7-workflow-cmake-warning-counts-by-class.txt`
- `artifacts/day7-workflow-cmake-warning-counts-by-file.txt`

## Workflow Design

The new Day 7 workflow is invoked through:

- `make warning-workflow WARNING_WORKFLOW_LABEL=<label>`

For each label, it:

1. removes the matching clean-room build directories
2. runs a clean CMake configure
3. runs a clean CMake build
4. runs full `ctest`
5. runs a clean Makefile `all` build in a separate build directory
6. derives warning summaries from the CMake build stderr

This preserves the Sprint 30 distinction between:

- the authoritative full-tree warning inventory: CMake build
- the library-only cross-check: Makefile `all`

## Validation Performed

Validation run:

- `make warning-workflow WARNING_WORKFLOW_LABEL=day7-workflow`
- `make format`
- `make lint`
- `make test`

Observed result:

- CMake configure warnings: `0`
- CMake build warnings: `112`
- Makefile build warnings: `0`
- `ctest`: `52/52` passed
- total CTest time: `205.75 sec`
- end-of-day validation sequence: passed

Derived CMake warning counts from the validated workflow run:

- by area
  - `tests`: `98`
  - `benchmarks`: `13`
  - `examples`: `1`
- by warning class
  - `-Wmissing-field-initializers`: `72`
  - `-Wdouble-promotion`: `34`
  - `-Wunused-function`: `3`
  - `-Wimplicit-function-declaration`: `2`
  - `-Wswitch`: `1`

The workflow reproduced the same post-Day-5 warning profile expected from the current branch state.

## Day 7 Conclusion

Day 7 closed the workflow-automation goal:

- baseline reproduction no longer depends on manually retyping the Sprint 30 command sequence
- warning-capture locations are stable and label-based
- the workflow preserves both the authoritative CMake warning inventory and the Makefile cross-check
- the run can be repeated before and after edits without overwriting prior evidence
