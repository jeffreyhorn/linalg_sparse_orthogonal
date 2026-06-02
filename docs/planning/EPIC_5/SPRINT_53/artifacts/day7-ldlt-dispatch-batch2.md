# Sprint 53 Day 7: LDL^T Dispatch Batch II

## Purpose

Day 7 tightens the next LDL^T CSC dispatch seam without redesigning the public
direct-solver surface. The target is the shared analysis-aware CSC completion
helper: once the outer dispatch has already selected the CSC path, the code
still needs to distinguish between:

- the intended batched-supernodal rejection fallback
- real helper failures or contract violations

Day 7 makes that distinction explicit, aligns the public CSC wording with that
layered reality, and adds direct proof that invalid completion configuration is
rejected rather than silently treated as fallback.

## Main Day 7 Result

Day 7 tightened `ldlt_csc_factor_with_resolved_analysis(...)` in
`src/sparse_ldlt.c` so the CSC completion seam is now explicit:

1. `SPARSE_OK`
   - retain the batched supernodal completion
2. `SPARSE_ERR_BADARG`
   - fall back to the resolved scalar pre-pass factor because the batched path
     rejected the cached pivot pattern
3. any other error
   - propagate directly instead of being masked as dispatch fallback

This is a real dispatch cleanup rather than just wording. Before Day 7, a bad
internal completion configuration could be silently converted into “success via
scalar fallback.”

## Code Changes

### 1. Shared CSC completion helper contract tightened

`src/sparse_ldlt.c` now validates that:

- `analysis->type == SPARSE_FACTOR_LDLT`
- `min_size >= 1`

inside `ldlt_csc_factor_with_resolved_analysis(...)`.

It also narrows the fallback path so only the intended batched-path rejection
case (`SPARSE_ERR_BADARG` from the supernodal attempt) falls back to the
resolved scalar pre-pass factor. Other failures now propagate.

### 2. Public CSC wording now matches the actual dispatch layering

`include/sparse_ldlt.h` now states the correct layering more explicitly:

- `SPARSE_LDLT_BACKEND_CSC` means the CSC pipeline, not an unconditional
  promise of batched supernodal completion
- once selected, the CSC pipeline may complete via:
  - batched supernodal completion
  - resolved scalar-prepass fallback

Related commentary in `tests/test_ldlt.c` and
`tests/test_sprint20_integration.c` was tightened to match that same contract.

## Regression Proof Added

Day 7 added a focused internal/helper regression in `tests/test_ldlt_csc.c`:

- `test_s53_with_analysis_invalid_min_size_rejected`

The test builds a valid resolved-analysis KKT setup, then calls:

- `ldlt_csc_factor_with_resolved_analysis(..., min_size = 0, ...)`

and asserts:

- `SPARSE_ERR_BADARG`

This proves the helper no longer silently aliases a contract violation into the
same path used for intended supernodal rejection fallback.

## Preserved Contract

Day 7 intentionally preserved the bounded Sprint 50-53 semantics:

- one-shot LDL^T remains first-class
- repeated direct runs remain analysis/factors-centric
- `used_csc_path` still reports CSC-vs-linked-list selection only
- the scalar BK pre-pass remains the authoritative indefinite permutation
  resolution step
- no raw CSC/native storage is exposed publicly

This is completion-seam clarification and proof-strengthening, not a public API
expansion.

## Validation

Because `*.c` / `*.h` changed, Day 7 ran the full required gate:

- `make format`
- `make lint`
- `make test`

All passed.

Day 7 also ran the touched follow-ons justified by the batch:

- `./build/test_ldlt`
- `./build/test_ldlt_csc`
- `./build/test_sprint20_integration`
- `./build/example_analysis`

Representative direct results:

- `test_ldlt` = `84 / 84`
- `test_ldlt_csc` = `96 / 96`
- `test_sprint20_integration` = `20 / 20`
- `example_analysis` residual = `4.44e-16`

## What Day 7 Solved

- narrowed the shared LDL^T CSC completion fallback seam
- stopped invalid helper configuration from being silently treated as scalar
  fallback
- made the public CSC wording more accurate about what “CSC selected” means
- added direct proof on the helper contract Day 7 changed

## What Day 7 Did Not Solve

- LDL^T-specific factor-many benchmark proof is still later work
- measured CSC factor-many claims are still a separate benchmark batch
- the scalar BK pre-pass still remains the authoritative indefinite
  permutation-resolution step

## Operational Result

Sprint 53 now has a cleaner LDL^T CSC dispatch base before the benchmark days:

- outer CSC-vs-linked-list dispatch is already explicit from Day 6
- inner CSC completion fallback is now narrower and less error-prone
- public wording no longer overclaims “supernodal” when it really means
  “CSC selected”
