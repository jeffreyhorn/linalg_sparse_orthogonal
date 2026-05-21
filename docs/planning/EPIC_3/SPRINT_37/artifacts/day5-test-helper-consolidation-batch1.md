# Sprint 37 Day 5 Test-Helper Consolidation Batch I

**Date:** 2026-05-20  
**Branch:** `sprint-37`

## Objective

Convert the Day 2 test-helper audit into the safest first consolidation slice
by extracting the repeated L2 residual helpers used across the
iterative/preconditioner/integration cluster, while preserving readable
single-file test ownership.

## Executive Summary

Day 5 shipped a narrow shared helper layer:

- `tests/test_solver_helpers.h`

It consolidates the repeated residual/norm logic that had drifted across
multiple solver and integration tests without introducing a new linked helper
module or a broad global framework.

The new helper surface is intentionally small:

- `tf_vec_norm2(...)`
- `tf_relative_residual_l2(...)`
- `tf_block_relative_residual_l2(...)`

This batch removed duplicated local residual helpers from eight test files and
kept file-local failure semantics explicit through a caller-provided allocation
failure sentinel.

## Touched Test Cluster

The first consolidation batch covered:

- `tests/test_iterative.c`
- `tests/test_bicgstab.c`
- `tests/test_ilu.c`
- `tests/test_minres.c`
- `tests/test_sprint5_integration.c`
- `tests/test_sprint10_integration.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint13_integration.c`

What was removed:

- duplicated single-RHS residual helpers
- one duplicated block-RHS residual helper
- one duplicated local vector-norm helper used only by that residual logic

What was added:

- shared header-only helper functions with explicit names and explicit scope

## Why This Landing Shape Was Chosen

The Day 2 audit already established the build-model constraint:

- each test binary is still built from its own source file
- both Makefile and CMake treat tests as one-binary-per-source

That makes a shared `.c` support module a worse first move because it would add
link-time ownership and broaden the refactor surface.

So Day 5 deliberately chose:

- a header-only helper layer
- small pure functions
- no behavior changes
- no registration or truthfulness-policy changes

This keeps the batch reviewable and low risk.

## New Helper Contract

### `tf_vec_norm2(...)`

Purpose:

- shared L2 vector norm helper for solver/integration tests that need the same
  residual math support

### `tf_relative_residual_l2(...)`

Purpose:

- compute `||b - A*x|| / ||b||` using the shared L2 path

Important design choice:

- callers provide `alloc_fail_sentinel`

Why:

- different test files already used different failure conventions
- Sprint 37 should reduce duplication first, not silently normalize unrelated
  behavioral details

Preserved caller behavior:

- `tests/test_iterative.c` keeps `-1.0`
- the other touched files keep `HUGE_VAL`

### `tf_block_relative_residual_l2(...)`

Purpose:

- compute the max-column relative residual for block-RHS tests

This was extracted from the cleanest existing implementation in the touched
cluster rather than generalized beyond current need.

## Validation Notes

Initial validation steps:

- `make format`
- `make lint`
- `make test`

The first `make test` run found one legitimate cleanup fallout:

- `tests/test_sprint10_integration.c` still referenced removed helper
  `local_norm2(...)`

Fix:

- replace that remaining use with `tf_vec_norm2(...)`

Authoritative validation rerun:

- `make format`
- `make lint`
- `make test`

Result:

- all passed

This confirms the batch is behaviorally stable and fits the Sprint 37 narrow
batch contract.

## Residual Queue After Day 5

Still deferred from the broader Day 2 test-helper audit:

- SPD / tridiagonal builder consolidation
- KKT builder consolidation
- file-specific residual helpers in other families such as:
  - `tests/test_qr.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_sprint19_integration.c`

These were deliberately left out because they either:

- have more file-local semantics
- belong to different solver families
- or would have widened the first batch beyond the intended low-risk cluster

## Day 5 Conclusion

Day 5 reduced the safest high-value test-helper duplication without inventing a
new broad test-support layer.

The iterative/preconditioner/integration residual-helper cluster is now
consolidated behind a small explicit header, and the remaining Sprint 37
test-helper queue is narrower and more clearly partitioned for later work.
