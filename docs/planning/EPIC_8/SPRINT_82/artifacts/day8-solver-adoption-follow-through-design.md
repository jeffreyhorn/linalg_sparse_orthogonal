# Sprint 82 Day 8 - Solver Adoption Follow-Through Design

Date: 2026-06-19  
Branch: sprint-82

## Purpose

Fix the exact LDL^T backend/runtime follow-through contract so Day 9 can widen
the first optional accelerated dense-backend seam beyond the Cholesky lane
without reopening broad backend-framework churn.

## Main Result

Sprint 82 now has one exact second implementation contract:

- required Day 9 center:
  - `src/sparse_ldlt.c`
  - `src/sparse_ldlt_csc_supernodal.c`
- strongest support-only code if the implementation truly forces it:
  - `src/sparse_chol_csc.c`
  - `src/sparse_dense.c`
- strongest support-only proof and measurement follow-through:
  - `tests/test_ldlt.c`
  - `benchmarks/bench_refactor_csc.c`
- support-only wording only if the batch truly changes the public reading:
  - `include/sparse_ldlt.h`
  - `README.md`
  - `docs/maintainer_guide.md`

## Why This Is The Exact Next Seam

The next seam is now explicit:

- `src/sparse_ldlt.c` owns the public/backend-dispatch side of the LDL^T CSC
  lane
- `src/sparse_ldlt_csc_supernodal.c` owns the supernodal dense-kernel
  consumption side
- the current LDL^T supernodal diagonal-block factor still routes through
  `ldlt_dense_factor(...)` in `src/sparse_chol_csc.c`, so that file is the
  strongest support-only implementation surface if Day 9 needs to align the
  widened backend/runtime seam

This is stronger than benchmark or docs follow-through because:

- the Cholesky lane already has the new optional runtime selector and family-
  local proof
- the LDL^T lane still lacks matching widened backend/runtime parity
- `bench_refactor_csc.c` already owns the retained repeated-run throughput and
  proof surface, so measurement drift is weaker than solver-path parity drift
- public wording only becomes stale if the Day 9 solver-side contract actually
  changes what callers can rely on

## Preserved Day 9 Fence

Day 9 should preserve:

- builtin default backend as the main product path
- the existing scalar-prepass / supernodal fallback story for the CSC LDL^T
  lane
- benchmark reporting as threshold-free measurement, not a timing gate

Day 9 should not widen into:

- QR or SVD backend work
- package/platform convergence
- shared-library maturity or fake platform-parity claims
- benchmark threshold/gate work
- a generic whole-library backend framework rewrite

## Exit State

- Sprint 82 now has one exact LDL^T backend/runtime follow-through contract.
- Day 9 can land one bounded solver-side batch without reopening the broader
  backend design.
- Support-only proof, benchmark, and wording surfaces are explicitly separated
  from the required implementation center.
