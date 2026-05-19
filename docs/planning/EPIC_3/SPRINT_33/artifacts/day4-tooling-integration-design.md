# Sprint 33 Day 4 Tooling Integration Design

**Date:** 2026-05-18  
**Branch:** `sprint-33`

## Objective

Define the concrete workflow shape for Sprint 33 dead-code tooling before any Makefile edits land: target topology, build paths, artifact paths, prerequisite behavior, and the boundary between raw evidence gathering and later report/enforcement work.

## Design Summary

Sprint 33 should ship dead-code tooling in layers:

1. compilation-database generation
2. raw evidence gathering
3. readable report generation
4. optional later enforcement

The main reason for this layering is the Day 3 policy:

- raw scanner output is evidence, not oracle
- the current compile-db coverage is incomplete for part of the tooling surface
- public/documented code must not be collapsed into the same bucket as internal dead code

## Chosen Workflow Shape

### Makefile target layer

Operator-facing targets:

- `deadcode-compile-db`
- `deadcode`
- `deadcode-report`
- `deadcode-check`

Day 4 expectation:

- Day 5 implements `deadcode-compile-db` and `deadcode`
- Day 6 / Day 7 implement the reporting/check layer

### Helper-script pattern

Day 4 recommends reusing the same general pattern as `warning-workflow`:

- Makefile target stays thin
- nontrivial flow can move into `scripts/` as soon as the shell logic becomes noisy

Why:

- prerequisite checks
- multiple commands
- stable artifact paths
- easier future CI reuse

## Dedicated Build Path

Chosen CMake build directory:

- `build/deadcode-cmake`

Why:

- guarantees a predictable `compile_commands.json`
- avoids reliance on prior sprint-specific build directories
- isolates dead-code workflow state from normal build/test work

Day 4 rule:

- do not depend on `build/sprint33-day1-cmake`
- always generate or refresh `build/deadcode-cmake/compile_commands.json` from the target itself

## Local Artifact Paths

Chosen artifact root:

- `build/deadcode/`

Initial raw artifacts:

- `build/deadcode/cppcheck.txt`
- `build/deadcode/xunused.txt`
- `build/deadcode/coverage-notes.txt`

Why:

- these are local workflow outputs, not long-term planning records
- they should be cheap to overwrite on rerun
- sprint-history copies can be summarized separately into `docs/planning/.../artifacts/`

## Raw Command Contract

Required Day 5 raw inputs:

1. `cmake -S . -B build/deadcode-cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
2. `cppcheck --enable=all --quiet src/`
3. `xunused build/deadcode-cmake/compile_commands.json`

Day 4 interpretation:

- those commands satisfy the project-plan shape
- they do not yet define the final operator-readable report

## Behavior Of `make deadcode`

Chosen Day 4 behavior:

- gather evidence
- do not fail on ordinary findings
- fail only on infrastructure/tool execution problems during the first implementation pass

Examples of failure-worthy conditions:

- `cmake` configure fails
- `compile_commands.json` is not produced
- `cppcheck` executable missing or invocation fails
- `xunused` executable missing or invocation fails

Examples of non-failure findings in Sprint 33 Day 5:

- reported unused-function candidates
- reachability candidates
- known coverage-gap notes

Why:

- Day 3 policy rejects treating raw findings as auto-delete proof
- a fail-on-any-finding gate would be noisy and misleading before the report layer exists

## Coverage-Gap Reporting Requirement

The workflow must explicitly surface that the current compile-db under-covers part of the Makefile tooling surface.

Known Day 4 gap list:

- missing benchmark:
  - `bench_svd`
- missing examples:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

Why this must be explicit:

- it changes what `xunused` can prove
- otherwise local users may assume “not reported” means “not present”

Recommended placement:

- `build/deadcode/coverage-notes.txt`

## Separation From Existing Quality Flow

Day 4 recommends keeping dead-code tooling separate from `lint` for Sprint 33.

Reason:

- `lint` is already stable and understood
- `deadcode` still has:
  - known coverage gaps
  - missing local `xunused`
  - policy/report semantics still being built

So the correct Sprint 33 behavior is:

- explicit opt-in dead-code target
- no automatic `lint` integration yet

## Day 6 / Day 7 Follow-On Contract

This design intentionally leaves later work for:

- `deadcode-report`
  - normalize raw tool output
  - preserve policy categories
  - surface false-positive buckets
- `deadcode-check`
  - add narrower enforcement only after the report semantics are stable

## Day 4 Conclusion

The important Day 4 decision is to keep the workflow layered:

- compile-db generation
- raw evidence gathering
- categorized reporting
- later enforcement

That keeps Sprint 33 aligned with the Day 3 conservative policy while still making Day 5 implementation straightforward and reusable.
