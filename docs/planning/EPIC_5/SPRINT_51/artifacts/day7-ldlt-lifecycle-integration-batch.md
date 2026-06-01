# Sprint 51 Day 7: LDL^T Lifecycle Integration Batch

## Objective

Route the shared repeated-run direct lifecycle path through the bounded LDL^T
options seam so the explicit analysis/factor path inherits the same linked-list
vs CSC backend dispatch behavior as the public one-shot LDL^T surface, while
preserving the family-local owned-factor contract.

## Files Changed

- `src/sparse_analysis.c`
- `tests/test_integration.c`

## What Landed

### 1. Shared LDL^T numeric routing now uses the normal options seam

The `SPARSE_FACTOR_LDLT` branch inside `sparse_factor_numeric(...)` no longer
calls `sparse_ldlt_factor(...)` directly on the already-permuted working copy.

It now routes through:

- `sparse_ldlt_factor_opts(...)`
- with `.reorder = SPARSE_REORDER_NONE`

That preserves the explicit analysis object as the sole owner of symbolic
reorder choice while reusing the normal one-shot backend-selection logic.

### 2. The repeated-run direct path now inherits linked-list vs CSC dispatch

This was the real remaining Phase-1 gap: the shared direct lifecycle path had
been correct, but it was still bypassing the public one-shot LDL^T options
seam that decides between linked-list and CSC factoring.

After the Day 7 patch:

- the explicit `sparse_analyze(...)` + `sparse_factor_numeric(...)` path
  inherits the same backend-selection behavior as the one-shot LDL^T path
- CSC writeback behavior remains governed by the existing LDL^T options
  surface instead of a separate lifecycle-only implementation branch

### 3. The LDL^T factor-object story remains intact

The batch did not change the caller-facing contract for:

- `sparse_ldlt_factor(...)`
- `sparse_ldlt_factor_opts(...)`
- `sparse_ldlt_solve(...)`
- `sparse_ldlt_free(...)`

The change is limited to how the shared lifecycle contract internally realizes
the numeric LDL^T phase.

### 4. Direct public-surface parity coverage now exists for the CSC-threshold side of the path

The integration suite now includes a focused regression that compares:

- bounded `sparse_ldlt_factor_opts(...)` with AMD reordering
- explicit `sparse_analyze(...)` + `sparse_factor_numeric(...)` +
  `sparse_factor_solve(...)`

on a `200x200` tridiagonal SPD matrix.

That size is intentional: it sits above the default CSC auto-routing threshold,
so the test proves parity across the seam that Day 7 actually changed.

### 5. The batch stayed inside the Sprint 50/51 scope fence

Day 7 did not:

- expose raw internal CSC/native storage layout
- introduce a new generic direct handle
- demote/remove one-shot LDL^T APIs
- broaden the public lifecycle contract to promise backend telemetry
- reopen docs/example conversion before the main source path settled

This remains a bounded Phase-1 direct-lifecycle implementation batch.

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
- `Total Test time (real) = 390.43 sec`

### Targeted direct-lifecycle follow-ons completed

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Representative direct results:

- `example_analysis` retained residuals at `4.44e-16`
- `bench_refactor_csc` continued to show strong CSC refactor wins on larger
  SuiteSparse cases:
  - `bcsstk14`: `speedup_refactor=5.49`
  - `s3rmt3m3`: `speedup_refactor=7.99`
  - `Kuu`: `speedup_refactor=6.17`
  - `Pres_Poisson`: `speedup_refactor=11.31`
- all touched direct structural regression binaries stayed green

## Bottom Line

Sprint 51 Day 7 made the LDL^T repeated-run route real in code:

- the shared explicit analysis path now reuses the normal LDL^T options seam
- linked-list vs CSC dispatch no longer differs between the one-shot options
  path and the shared lifecycle path
- the family-local LDL^T factor-object contract remains intact for callers
- regression coverage now proves parity with the explicit analysis API on the
  CSC-threshold side of the contract
