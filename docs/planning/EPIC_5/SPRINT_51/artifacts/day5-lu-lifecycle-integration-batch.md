# Sprint 51 Day 5: LU Lifecycle Integration Batch

## Objective

Route the bounded default LU options path through the shared
`sparse_analysis` / `sparse_factor_numeric` lifecycle seam while preserving:

- the simple/default one-shot LU caller story
- the existing matrix-local `sparse_lu_solve(...)` contract
- legacy custom-pivot / callback behavior where the shared lifecycle contract
  is not yet expressive enough

## Files Changed

- `src/sparse_lu.c`
- `tests/test_integration.c`

## What Landed

### 1. Bounded LU lifecycle routing in `sparse_lu_factor_opts(...)`

The new shared-lifecycle path is enabled only for the bounded option set that
already matches the public direct repeated-run contract:

- `pivot == SPARSE_PIVOT_PARTIAL`
- `tol == 1e-12`
- `progress_cb == NULL`
- matrix still in original row/column state

When those conditions hold, `sparse_lu_factor_opts(...)` now:

1. builds a LU analysis object with the requested reorder mode
2. factors through `sparse_factor_numeric(...)`
3. republishes the factored result back onto the caller-owned LU matrix

### 2. Compatibility-preserving factor republish bridge

The shared lifecycle path naturally keeps reorder information in
`sparse_analysis_t`, but the one-shot LU solve contract expects
`reorder_perm` on the factored matrix itself.

The Day 5 patch therefore added a small internal bridge that:

- steals the factorized working-copy payload from the lifecycle-owned LU matrix
- transfers `analysis.perm` back onto the caller-owned matrix
- republishes the factor-state compatibility mirrors with
  `sparse_factor_state_publish_factored(...)`

That keeps `sparse_lu_solve(...)` and the one-shot LU matrix contract
unchanged for callers.

### 3. Deliberate fallback to the legacy LU path where needed

The pre-existing direct LU route still handles:

- custom pivot/tolerance combinations
- progress / cancellation callbacks
- non-original or already-mutated matrix state

This is intentional. The Sprint 50/51 contract did not promise that the shared
direct lifecycle path already exposes arbitrary LU-specific controls.

### 4. Direct public-surface parity coverage

The integration suite now contains a focused regression that compares:

- bounded `sparse_lu_factor_opts(...)` with AMD reordering
- explicit `sparse_analyze(...)` + `sparse_factor_numeric(...)` +
  `sparse_factor_solve(...)`

The test verifies solution parity on the same matrix/right-hand side pair, so
future drift between the one-shot LU options path and the shared lifecycle path
will fail visibly.

## Why This Is the Right Phase-1 Seam

This batch stayed inside the Sprint 50 design fence:

- no raw internal CSC/native storage exposure
- no generic direct-handle redesign
- no demotion/removal of one-shot LU APIs
- no expansion of the shared lifecycle path to arbitrary LU pivot/tolerance
  controls

The narrow seam was correct because the shared LU lifecycle path already uses
fixed:

- partial pivoting
- `tol = 1e-12`

Routing the simple `sparse_lu_factor(...)` API through that seam directly would
have silently narrowed the caller-visible LU contract, which Sprint 50/51
explicitly avoided.

## Validation

### Required code-day gate

- `make format`
- `make lint`
- `make test`

All passed.

### Stronger reviewed baseline

- `make quality-review-full`

Passed.

Maintained truthfulness anchors:

- reviewed CMake parity remained `53`
- Makefile/CMake parity remained `53` vs `53`
- full reviewed CMake `ctest` passed `53 / 53`
- `Total Test time (real) = 357.29 sec`

### Targeted direct-lifecycle follow-ons completed

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Representative direct results:

- `example_analysis` retained residuals at `4.44e-16`
- `bench_refactor` still completed all listed fixtures and kept the
  analyze-once advantage on `bcsstk04`
- all touched direct structural regression binaries stayed green

## Bottom Line

Sprint 51 Day 5 made the LU repeated-run route real in code, not just in
documentation:

- the bounded default LU options path now uses the shared analyze/factor
  lifecycle seam
- one-shot LU solve behavior remains unchanged for callers
- custom-LU and callback cases remain on the legacy path intentionally
- regression coverage now proves parity with the explicit analysis API
