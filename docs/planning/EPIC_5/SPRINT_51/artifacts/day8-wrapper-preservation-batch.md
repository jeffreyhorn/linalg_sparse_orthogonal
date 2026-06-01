# Sprint 51 Day 8: Wrapper Preservation Batch

## Objective

Preserve the simple/default one-shot direct entry points while routing the
safe default wrappers through the Phase-1 lifecycle-aware options seams where
that contract already exists, without broadening the Sprint 50 scope fence.

## Files Changed

- `src/sparse_cholesky.c`
- `src/sparse_ldlt.c`
- `tests/test_integration.c`

## What Landed

### 1. The default Cholesky wrapper now reuses the normal options seam

`sparse_cholesky_factor(...)` now constructs the bounded default option set and
delegates to `sparse_cholesky_factor_opts(...)` with:

- `.reorder = SPARSE_REORDER_NONE`
- `.backend = SPARSE_CHOL_BACKEND_AUTO`
- no telemetry output
- no progress callback

That means the simple/default one-shot Cholesky path now inherits the same
backend-dispatch and validation behavior as the explicit options surface.

### 2. The default LDL^T wrapper now reuses the normal options seam

`sparse_ldlt_factor(...)` now constructs the bounded default option set and
delegates to `sparse_ldlt_factor_opts(...)` with:

- `.reorder = SPARSE_REORDER_NONE`
- `.tol = 0.0`
- `.backend = SPARSE_LDLT_BACKEND_AUTO`
- no telemetry output
- no progress callback

That means the simple/default one-shot LDL^T path now inherits the same
linked-list vs CSC dispatch behavior as the explicit options surface.

### 3. LU intentionally did not make the same final wrapper hop

The first Day 8 attempt also routed `sparse_lu_factor(...)` through
`sparse_lu_factor_opts(...)`, but that exposed a real recursion seam:

- `sparse_lu_factor_opts(...)`
- shared lifecycle route
- `sparse_factor_numeric(..., SPARSE_FACTOR_LU)`
- `sparse_lu_factor(...)`

So the LU wrapper change was explicitly backed out before final validation.

The final Day 8 outcome is therefore intentional:

- Cholesky and LDL^T safe-wrapper delegation landed
- LU remained on the family-local one-shot wrapper path

This keeps the batch bounded and avoids broad LU redesign inside Sprint 51 Day
8.

### 4. Focused wrapper-parity coverage now exists for Cholesky and LDL^T

The integration suite now includes:

- `test_cholesky_default_wrapper_matches_default_opts`
- `test_ldlt_default_wrapper_matches_default_opts`

Each regression compares:

- the one-shot default wrapper
- the explicit default options form

and checks that the solved outputs are bit-identical on the same tridiagonal
SPD case.

LU already had direct default-wrapper parity coverage from the earlier Sprint
51 LU batch, so Day 8 did not need another LU-specific parity test.

### 5. The batch stayed inside the Sprint 50/51 scope fence

Day 8 did not:

- expose raw internal CSC/native storage layout
- introduce a new generic direct handle
- demote/remove one-shot direct APIs
- promise reuse of old numeric factor state
- reopen broad docs/example conversion before the source seam stabilized

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
- `Total Test time (real) = 427.72 sec`

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
- `bench_refactor` stayed behavior-stable:
  - `tridiag-50`: `speedup=1.14x`
  - `bcsstk04`: `speedup=1.02x`
- `bench_refactor_csc` preserved the larger CSC refactor wins:
  - `bcsstk04`: `speedup_refactor=5.78`
  - `bcsstk14`: `speedup_refactor=5.48`
  - `s3rmt3m3`: `speedup_refactor=7.97`
  - `Kuu`: `speedup_refactor=6.96`
  - `Pres_Poisson`: `speedup_refactor=12.14`

## Bottom Line

Sprint 51 Day 8 made the wrapper-preservation rule explicit in code:

- default Cholesky and LDL^T wrappers now reuse their normal options seams
- direct regression coverage now proves wrapper-vs-default-options parity for
  those families
- LU intentionally remains on the family-local one-shot path until a later
  routing refactor removes the current recursion seam

This is a bounded, compatibility-preserving Phase-1 result rather than a broad
direct-solver API redesign.
