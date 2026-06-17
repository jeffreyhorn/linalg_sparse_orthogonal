# Sprint 75 Day 14: Closeout and Handoff

Date: 2026-06-17
Branch: `sprint-75`

## Purpose

Close Sprint 75 from the Day 13 validated baseline with one explicit
second-phase backend/performance package and a ranked carry-forward queue for
Sprint 76 and beyond.

## Sprint 75 Package Summary

Sprint 75 now closes as one coherent backend/performance package across:

- hotspot re-audit and backend fence
- dense-kernel / CSC supernodal integration
- public callback/runtime follow-through
- benchmark proof refresh
- proof-owner alignment
- validated close state

## What Landed

### 1. The dense-kernel seam now has one clearer shipped owner

- `src/sparse_dense.c` now owns the concrete dense-kernel descriptor for the
  self-contained default build
- that descriptor now includes the explicit batched `solve_panel` seam
- `tests/test_chol_csc.c` proves both correctness and the narrow missing-callback
  backend-contract failure

### 2. The CSC supernodal Cholesky lane now consumes the batched panel seam directly

- `src/sparse_chol_csc_supernodal.c` now uses the batched panel-solve callback
  directly on the touched supernodal path
- `src/sparse_chol_csc.c` / writeback-side ownership stayed coherent after the
  runtime and benchmark follow-through
- the landed change stayed bounded to the Cholesky CSC lane rather than widening
  into a repo-wide backend rewrite

### 3. The public CSC runtime contract now tells the smaller truthful story

- `src/sparse_cholesky.c` and `include/sparse_cholesky.h` now expose:
  - `phase = "cholesky_factor_csc"`
  - `total = 4`
  - bounded orchestration checkpoints instead of fake per-column CSC parity
- cancellation before writeback now has public-path proof that the caller
  matrix shell stays in the original coordinate space
- `tests/test_integration.c` owns that public runtime truth

### 4. The benchmark surface now makes the new seam directly measurable

- `bench_chol_csc` now emits `csc_supernodal_panel_solver`
- the retained default-build reading is:
  - `csc_supernodal_dense_kernel = builtin`
  - `csc_supernodal_panel_solver = batched_panel`
- benchmark proof remains measurement/reporting only; it does not become the
  owner of public callback/cancel truth

## Preserved Fence

Sprint 75 preserved the intended Epic 7 boundary:

- no broad backend abstraction-layer rewrite
- no fake optional-external-backend maturity claim
- no shared-library or plugin-style backend claim
- no benchmark-threshold pass/fail portability story
- no widened reviewed-platform or install-validation claim beyond maintained
  evidence

## Validated Close State

Sprint 75 closes from the Day 13 validated baseline:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- reviewed CMake parity `53`
- Makefile/CMake parity `53 vs 53`
- reviewed CMake `ctest` `53 / 53`
- `Total Test time (real) = 346.76 sec`

Representative retained anchors:

- `test_chol_csc` retained the landed family-local proof set
- `test_integration` retained the landed public CSC runtime proof set
- `example_analysis`: residual `4.44e-16`
- `example_basic_solve`: residual `0.00e+00`
- `bench_refactor_csc nos4`: `speedup_refactor = 1.11`
- `bench_chol_csc nos4`: `builtin`, `batched_panel`, `speedup_csc = 0.48`,
  `speedup_csc_sn = 0.81`
- install/package regressions: installed `pkg-config` version `2.2.0`

## Ranked Carry-Forward Queue

1. benchmark governance and longitudinal reporting, starting from the newly
   strengthened measurable backend surface
2. eigensolver backend/runtime parity as the strongest remaining backend-aware
   second lane
3. QR and SVD backend-aware follow-through only where a bounded proof-backed
   seam justifies movement
4. later packaging, ABI, or platform convergence only where maintained
   evidence supports a stronger claim

## Project Plan Recheck

`docs/planning/EPIC_7/PROJECT_PLAN.md` does not need a Sprint 75 correction.
The landed work still matches the sprint contract.

## Exit State

Sprint 75 closes with:

1. one real second backend-aware landing on the CSC supernodal Cholesky lane
2. one truthful public callback/cancel runtime contract for the widened CSC path
3. one measurable benchmark proof field for the new panel-solver seam
4. one validated handoff package for Sprint 76
