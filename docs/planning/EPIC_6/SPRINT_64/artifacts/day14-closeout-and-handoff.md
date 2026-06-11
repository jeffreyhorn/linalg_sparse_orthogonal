# Sprint 64 Day 14: Closeout and Handoff

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Convert the validated Sprint 64 branch into a clear closeout package for the
next Epic 6 backend/performance sprint.

This closeout exists to make the Sprint 64 result read as one coherent Phase 1
backend package rather than as a pile of hotspot/design/benchmark notes.

## Landed Sprint 64 Outcomes

Sprint 64 closes with five concrete backend/performance outcomes:

- the first backend-aware landing target was fixed to the CSC supernodal
  Cholesky dense-kernel lane
- a bounded internal dense-kernel descriptor seam was introduced without
  widening into a repo-wide backend framework
- the public error taxonomy now includes `SPARSE_ERR_BACKEND_CONTRACT` for the
  narrow internal backend-contract failure lane
- `bench_chol_csc` now exposes the active path-identification fields:
  - `csc_scalar_path`
  - `csc_supernodal_path`
  - `csc_supernodal_dense_kernel`
- the public/header/README/maintainer interpretation surfaces now explain the
  landed lane consistently

## Preserved Compatibility and Truthfulness Fence

Sprint 64 preserves the following explicit contract:

- the self-contained default build remains authoritative
- the backend-aware path remains bounded and optional
- fallback correctness remains explicit and proved
- the first backend-aware lane is still local to CSC supernodal Cholesky
- the default dense-kernel descriptor for that lane remains `builtin`
- `SPARSE_ERR_BACKEND_CONTRACT` stays narrow:
  - caller contract valid
  - selected internal backend-owned helper/callback contract failed

What Sprint 64 did **not** do:

- it did not create a general pluggable-backend framework
- it did not widen into packaging/platform work
- it did not turn benchmark refresh into broad benchmark-governance expansion

## Validated Baseline

Sprint 64 closes from the Day 13 validated baseline:

- `make format` passed
- `make lint` passed
- `make test` passed
- `make quality-review-full` passed

Reviewed anchors retained exactly:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 574.42 sec`

Strong retained proof signals:

- `test_chol_csc` = `144 / 144`
- `test_integration` = `47 / 47`
- `bench_chol_csc` still reports:
  - `scalar`
  - `supernodal`
  - `builtin`
- `bench_refactor_csc nos4`: `speedup_refactor = 1.63x`
- `bench_ldlt_csc nos4`: `speedup_csc_native = 1.60x`
- `bench_iterative_reuse`: `cg-tridiag-300 1.07x`, `gmres-unsym-220 1.03x`,
  `minres-kkt-42 1.00x`
- `bench_eigs_reuse`: `growm-nos4-k5 1.05x`, `thick-bcsstk14-k5 1.00x`,
  `lobpcg-diag40-k3 1.04x`

## Sprint 65 Handoff Queue

Ranked carry-forward queue after Sprint 64:

1. LDL^T CSC supernodal backend-aware follow-through
2. bounded shared dense-kernel seam reuse only where it reduces real duplicate
   risk
3. optional build-option or pluggable-kernel widening only if the
   self-contained default build and fallback truthfulness stay explicit
4. later QR / SVD backend layering only if the proof burden is justified
5. broader benchmark-governance consolidation and packaging/platform work stay
   deferred outside this immediate lane

## Non-Blocking Note

The reviewed CMake rebuild still emits the existing
`bench_eigs_reuse.c` double-promotion warnings while rebuilding
`bench_eigs_reuse`.

That remains non-blocking because:

- the full reviewed path completed cleanly
- the parity anchors stayed exact
- all targeted Sprint 64 proof reruns passed

## Exit State

Sprint 64 now hands off:

- one coherent backend-aware Phase 1 package
- one explicit truthfulness and fallback contract
- one fully validated close baseline
- one ranked Sprint 65 queue instead of a generic backend backlog
