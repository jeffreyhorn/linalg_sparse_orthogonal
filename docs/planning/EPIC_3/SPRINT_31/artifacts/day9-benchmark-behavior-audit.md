# Sprint 31 Day 9 Benchmark Behavior Audit

**Date:** 2026-05-17  
**Branch:** `sprint-31`

## Objective

Verify that the Sprint 31 benchmark/example tools now behave consistently after the CLI, portability, and initializer cleanup work, and close any small remaining user-facing drift that survived those earlier batches.

## Audited Tools

- `benchmarks/bench_main.c`
- `benchmarks/bench_reorder.c`
- `benchmarks/bench_colamd.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`
- `examples/example_colamd.c`

## Post-Cleanup Behavior Matrix

| tool | reorder / flag surface | post-Day-9 behavior |
|---|---|---|
| `bench_main` | main LU / Cholesky harness; reorder accepts `none`, `rcm`, `amd`, `nd` | accepts `nd`; prints lowercase reorder label; rejects `colamd` with an explicit contract error |
| `bench_reorder` | reorder-comparison bench; exposes `none`, `rcm`, `amd`, `colamd`, `nd`; supports `--nd-threshold`, `--skip-factor`, `--only`, `--reorder-via-analyze` | CSV rows use lowercase reorder names consistently across the five-mode set |
| `bench_colamd` | QR ordering comparison bench; compares `none`, `amd`, `colamd` | title, headers, row labels, and `% vs none` wording are all aligned |
| `bench_chol_csc` | backend-comparison bench; fixed AMD reorder baseline | continues to emit backend-comparison CSV only; does not advertise general reorder CLI |
| `bench_ldlt_csc` | backend-comparison bench; fixed AMD reorder baseline | continues to emit backend-comparison CSV only; does not advertise general reorder CLI |
| `example_colamd` | public COLAMD example; compares `none (natural order)` vs `colamd` | uses lowercase labels consistently and now prints `0% change` when the fill counts are equal |

## Live Checks

Observed behavior from the rerun artifacts:

- `bench_main --size 8 --repeat 1 --reorder nd`
  - succeeded
  - printed `Reorder: nd`
- `bench_main --size 8 --repeat 1 --reorder colamd`
  - exited nonzero
  - printed:
    - `Error: unknown reorder mode 'colamd' (use 'none', 'rcm', 'amd', or 'nd')`
- `bench_reorder --only nos4 --skip-factor`
  - printed lowercase rows for:
    - `none`
    - `rcm`
    - `amd`
    - `colamd`
    - `nd`
- `bench_colamd`
  - printed:
    - `=== QR Fill-In Comparison: colamd vs amd vs none ===`
    - `nnz(R)none`
    - `nnz(R)amd`
    - `nnz(R)colamd`
- `bench_chol_csc --small-corpus --repeat 1`
  - printed the expected backend-comparison CSV rows
- `bench_ldlt_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - printed the expected backend-comparison CSV row
- `example_colamd`
  - printed:
    - `none (natural order)`
    - `colamd`
    - `0% change` when `nnz(LU)` matched exactly

## Day 9 Fix

One small user-facing drift survived the earlier cleanup:

- `example_colamd` printed `(-0% increase)` when the fill counts were equal

Day 9 changed that case to:

- `0% change`

This keeps the public example aligned with the cleaned-up lowercase wording and avoids a misleading negative-zero presentation.

## Intentional Asymmetries

Two apparent differences remain, but both are intentional rather than leftover drift:

### `bench_main` vs `bench_reorder`

`bench_main` does not accept `colamd`, while `bench_reorder` does.

Reason:

- `bench_main` drives LU / Cholesky factorization paths
- those paths do not actually support `SPARSE_REORDER_COLAMD`
- Day 2 established that `bench_main` must stay aligned to the narrower real solver contract:
  - `none`
  - `rcm`
  - `amd`
  - `nd`

### Backend-comparison benches

`bench_chol_csc` and `bench_ldlt_csc` do not expose broad reorder selection.

Reason:

- their purpose is apples-to-apples backend timing
- they intentionally hold the fill-reducing reorder constant (AMD)
- broadening those tools into general reorder harnesses would be a separate feature scope, not a Sprint 31 cleanup correction

## Validation

Validation commands:

- `make format`
- `cmake --build build/sprint31-day1-cmake --parallel 1 --target example_colamd bench_main bench_reorder bench_colamd bench_chol_csc bench_ldlt_csc`
- reran all audited tools listed above
- `cmake --build build/sprint31-day1-cmake --parallel 1 --clean-first`

Compile-state result:

- full-tree warnings remained `99`
- benchmark/example warnings remained `1`
- no new warning class was introduced by the Day 9 output fix

Remaining benchmark/example warning:

- `benchmarks/bench_convergence.c`
  - `1` `-Wdouble-promotion`

## Day 9 Conclusion

Sprint 31’s behavior-consistency work now has a validated post-cleanup state:

- the main reorder-facing tools present lowercase labels consistently
- `bench_main`’s narrower reorder contract is explicit and correctly enforced
- the public COLAMD example no longer has the negative-zero output glitch
- the remaining Sprint 31 benchmark/example queue is no longer behavior drift
