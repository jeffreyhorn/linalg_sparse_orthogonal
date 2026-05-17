# Sprint 30 Rebuild Workflow

**Status:** Draft  
**Date:** 2026-05-16  
**Sprint:** 30

## Purpose

This document defines the repeatable local rebuild and warning-capture workflow used during Epic 3 compile-hygiene work. It exists so a maintainer can reproduce the warning baseline, validate a cleanup branch after edits, and compare before/after warning summaries without reconstructing the command sequence by hand.

## Workflow Entry Point

Preferred entry point:

```sh
make warning-workflow WARNING_WORKFLOW_LABEL=<label>
```

Default artifact location:

- `docs/planning/EPIC_3/SPRINT_30/artifacts`

Default build directories:

- `build/<label>-cmake`
- `build/<label>-make`

Default warning-capture parallelism:

- `WARNING_WORKFLOW_JOBS=1`

The Makefile wrapper provides a default label, so `make warning-workflow`
will still run if no override is provided. In practice, a custom label is
strongly recommended because it distinguishes one capture from another and
prevents accidental reuse of the default artifact names. Use labels such as:

- `pre-fix`
- `post-fix`
- `day7-workflow`
- `day8-recheck`

## What The Workflow Runs

For each label, the workflow:

1. deletes the matching CMake and Makefile build directories
2. runs a clean `cmake -S . -B build/<label>-cmake`
3. runs a clean `cmake --build build/<label>-cmake --parallel <jobs>`
4. runs `ctest --test-dir build/<label>-cmake --output-on-failure`
5. runs `make BUILDDIR=build/<label>-make all`
6. derives warning summaries from the CMake build stderr

This keeps the authoritative full-tree warning inventory on the CMake path while preserving the Makefile library-build cross-check used earlier in Sprint 30.

The default `WARNING_WORKFLOW_JOBS=1` is intentional. Serialized warning capture avoids interleaved compiler stderr lines, which keeps area/file attribution stable in the derived warning summaries. If a maintainer wants a faster build-only rerun, they can override the job count, but the Sprint 30 baseline artifacts should use the serialized default.

## Artifacts Produced

Each run writes:

- raw configure/build/test logs
- Makefile build logs
- warning counts by area
- warning counts by warning class
- warning counts by file
- a one-file workflow summary

For a label `example-run`, the artifact set is:

- `example-run-cmake-configure.stdout.txt`
- `example-run-cmake-configure.stderr.txt`
- `example-run-cmake-build.stdout.txt`
- `example-run-cmake-build.stderr.txt`
- `example-run-ctest.stdout.txt`
- `example-run-ctest.stderr.txt`
- `example-run-make-build.stdout.txt`
- `example-run-make-build.stderr.txt`
- `example-run-cmake-warning-counts-by-area.txt`
- `example-run-cmake-warning-counts-by-class.txt`
- `example-run-cmake-warning-counts-by-file.txt`
- `example-run-workflow-summary.md`

## Comparison Workflow

To compare a cleanup before and after edits:

1. run `make warning-workflow WARNING_WORKFLOW_LABEL=<before-label>`
2. make the code changes
3. run `make warning-workflow WARNING_WORKFLOW_LABEL=<after-label>`
4. compare the two `*-workflow-summary.md` files and the derived warning-count files

The label-based naming avoids accidental overwrites and keeps Sprint 30 evidence stable.

## Relationship To Other Validation

This workflow is not a replacement for end-of-day validation. After edits, the Sprint 30 standard remains:

```sh
make format && make lint && make test
```

The warning workflow answers:

- what warnings exist on the authoritative full-tree build path
- whether those warnings moved after a change
- whether the CMake validation path still passes

The end-of-day standard answers:

- whether formatting is normalized
- whether the Makefile lint path still passes
- whether the Makefile test path still passes

Both are required for Sprint 30 work.
