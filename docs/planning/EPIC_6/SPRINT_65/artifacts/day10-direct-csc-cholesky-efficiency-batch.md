# Sprint 65 Day 10: Direct CSC/Cholesky Efficiency Batch

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Land one bounded real efficiency follow-through on the CSC supernodal
Cholesky path without widening into LDL^T, iterative/eigensolver reuse,
public headers, or broader benchmark-governance work.

## Landed Scope

This batch intentionally stays limited to:

- `src/sparse_chol_csc_supernodal.c`

It intentionally does not widen into:

- `src/sparse_ldlt_csc.c`
- `src/sparse_ldlt_csc_supernodal.c`
- `src/sparse_iterative.c`
- `src/sparse_eigs.c`
- public headers
- build wiring
- benchmark taxonomy or canonical-surface docs

## Landed Efficiency Follow-Through

The strongest Day 10 remaining waste was repeated row-map restart work on the
sorted supernodal CSC paths:

- extract
- diagonal-block cmod
- writeback

The landed change replaces restart-from-top binary search with a monotonic
forward-only row-map seek:

- `chol_csc_supernode_extract(...)`
- `chol_csc_supernode_eliminate_diag(...)`
- `chol_csc_supernode_writeback(...)`

Stable contract after the landing:

- the row-map cursor only moves forward through sorted `row_map`
- extract and writeback still require exact row membership
- diagonal-block cmod still skips non-panel rows safely
- no behavior, backend, or error-taxonomy widening was introduced

## Why This Batch Matters

Interpretation:

- this is a real hot-path cleanup inside the maintained CSC supernodal
  Cholesky lane
- it aligns with the Sprint 65 direct repeated-run CSC/Cholesky efficiency
  target instead of reopening backend-architecture design
- it keeps the proof burden bounded to the existing Cholesky CSC and
  integration surfaces

## Validation

Because `src/sparse_chol_csc_supernodal.c` changed, the Day 10 validation gate
was:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed. The reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 578.20 sec`

One non-blocking note remains explicit: the reviewed CMake path still spent
most of its wall time in `test_reorder_nd`, but the full reviewed path
completed cleanly and passed all parity gates.

## Retained Benchmark Proof Checks

The maintained direct benchmark proof reruns were:

- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `./build/bench_chol_csc tests/data/suitesparse/nos4.mtx --repeat 1`

Representative retained rows are:

- `bench_refactor_csc,proof,nos4.mtx,chol_spd,100,594,0.640,0.426,0.168,0.011,0.006,2.54,8.24e-16,7.06e-16`
- `bench_chol_csc,proof,nos4.mtx,chol_backend_compare,100,594,scalar,supernodal,builtin,0.620,0.732,0.958,0.013,0.018,0.009,0.85,0.65,7.06e-16,5.89e-16,5.89e-16`

Interpretation:

- the repeated-run direct CSC proof row retained clean residuals and a strong
  `speedup_refactor` row on the sampled `nos4` rerun
- the backend-comparison row stayed honest but remains the narrower one-shot
  backend/path-identification surface rather than the main repeated-run claim

## Day 10 Exit State

Sprint 65 now has:

- one bounded CSC supernodal hot-path efficiency follow-through landed
- one preserved maintained direct benchmark story with fresh retained proof
  rows
- one still-bounded carry-forward queue for any later LDL^T or wider solver
  efficiency work
