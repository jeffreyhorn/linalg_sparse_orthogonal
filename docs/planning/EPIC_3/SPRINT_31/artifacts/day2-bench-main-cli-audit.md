# Sprint 31 Day 2 `bench_main` CLI And Reorder Audit

**Date:** 2026-05-17  
**Branch:** `sprint-31`

## Objective

Audit `bench_main` against the actual reorder API and the project’s benchmark reference tools, then choose the canonical Sprint 31 CLI contract before any implementation work begins.

## Current `bench_main` State

Observed from `benchmarks/bench_main.c`:

- file-header usage text advertises `--reorder rcm|amd|none`
- parser accepts only `none`, `rcm`, and `amd`
- unknown-reorder diagnostics mention only `none`, `rcm`, and `amd`
- `reorder_name()` handles only `SPARSE_REORDER_NONE`, `SPARSE_REORDER_RCM`, and `SPARSE_REORDER_AMD`

User-visible consequence:

- the main benchmark harness omits at least one reorder mode the current LU/Cholesky solver paths really support
- the current code also emits a `-Wswitch` warning because `reorder_name()` is not exhaustive for the enum values it may encounter

## Reference Comparison

### `bench_reorder`

Observed from `benchmarks/bench_reorder.c`:

- benchmark set includes `NONE`, `RCM`, `AMD`, `COLAMD`, and `ND`
- the tool is explicitly a cross-ordering benchmark rather than a solver harness
- it treats the broader reorder enum surface as normal

Interpretation:

- `bench_reorder` is a useful consistency reference for naming
- it is **not** proof that every solver-oriented benchmark path should accept every reorder enum the project exposes

### Public solver option contracts

Observed from the public headers and implementation:

- `sparse_lu_opts_t`
  - documented: `NONE`, `RCM`, `AMD`, `ND`
  - implementation in `src/sparse_lu.c`: accepts `RCM`, `AMD`, `ND`; rejects `COLAMD`
- `sparse_cholesky_opts_t`
  - documented: `NONE`, `RCM`, `AMD`, `ND`
  - implementation in `src/sparse_cholesky.c`: accepts `RCM`, `AMD`, `ND`; rejects `COLAMD`
- `sparse_ldlt_opts_t`
  - documented: `NONE`, `RCM`, `AMD`, `ND`
  - implementation in `src/sparse_ldlt.c`: accepts `RCM`, `AMD`, `ND`; rejects `COLAMD`

Interpretation:

- `ND` is clearly part of the real solver contract for the `bench_main` paths
- `COLAMD` is not part of the current LU/Cholesky/LDLT factorization-option contract

### Public QR / analysis contracts

Observed from the public headers and implementation:

- `sparse_qr_opts_t` treats `COLAMD` as the recommended reorder for unsymmetric QR and also accepts `AMD`, `RCM`, and `ND`
- `sparse_analysis()` accepts `COLAMD`, but only through its own analysis-time symmetric-permutation path

Interpretation:

- `COLAMD` is a real project capability
- `COLAMD` is not currently a real capability of the specific LU/Cholesky factorization-option paths used by `bench_main`

## Sprint 31 Contract Decision

Canonical `bench_main` contract for the existing LU and `--cholesky` modes:

- accepted reorder spellings: `none`, `rcm`, `amd`, `nd`
- printed reorder labels: `none`, `rcm`, `amd`, `nd`
- help text should advertise exactly those four values
- unknown-reorder diagnostics should name exactly those four values

`COLAMD` decision:

- do not add `colamd` to the current `bench_main` reorder parser in Sprint 31
- reason: the underlying LU/Cholesky solver option paths do not support it
- if the project later wants `COLAMD` in a main benchmark harness, that should be a QR-oriented or analyze-oriented mode rather than a parser-only change that overpromises solver support

## Why This Resolves The Day 2 Ambiguity

Sprint 30’s handoff and Sprint 31’s planning text treated `COLAMD` and `ND` together as “missing reorder coverage” in `bench_main`. The code audit shows those two cases are not equivalent:

- `ND` is a genuine missing `bench_main` feature relative to the real LU/Cholesky solver contract
- `COLAMD` is a broader library capability, but not a capability of the current `bench_main` solver paths

That means Sprint 31 Day 3 should fix a real correctness gap (`nd`) and close the stale help/label drift, while **not** introducing a misleading `colamd` CLI path for solver modes that would reject it underneath.

## Day 2 Conclusion

The canonical Sprint 31 `bench_main` reorder contract is now explicit:

- `none`
- `rcm`
- `amd`
- `nd`

This is the smallest contract that is both user-honest and aligned with the current LU/Cholesky implementations.
