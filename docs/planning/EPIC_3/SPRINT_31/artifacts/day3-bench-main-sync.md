# Sprint 31 Day 3 `bench_main` Sync Fixes

**Date:** 2026-05-17  
**Branch:** `sprint-31`

## Objective

Implement the first `bench_main` contract-alignment batch by bringing its usage text, reorder parser, and reorder-label helper into line with the canonical Day 2 LU/Cholesky solver contract.

## Edits Landed

Edited file:

- `benchmarks/bench_main.c`

Changes:

1. Updated the file-header usage text from:
   - `--reorder rcm|amd|none`
   to:
   - `--reorder none|rcm|amd|nd`
2. Extended `reorder_name()` to print:
   - `none`
   - `rcm`
   - `amd`
   - `colamd`
   - `nd`
3. Extended the `--reorder` parser to accept `nd`.
4. Updated the unknown-reorder diagnostic to:
   - `use 'none', 'rcm', 'amd', or 'nd'`

## Why `reorder_name()` Now Handles `colamd`

Day 2 established that `bench_main` should **not** accept `colamd` in its current LU/Cholesky solver modes, because the underlying factorization-option paths reject `SPARSE_REORDER_COLAMD`.

That parser decision did **not** mean `reorder_name()` should remain non-exhaustive.

Reasons to handle `SPARSE_REORDER_COLAMD` in the label helper anyway:

- it removes the `-Wswitch` warning
- it prevents stale `"unknown"` labeling if that enum value reaches the helper through future refactors or non-parser call paths
- it does not widen the user-facing parser contract

So after Day 3:

- parser contract: `none|rcm|amd|nd`
- label helper coverage: `none|rcm|amd|colamd|nd`

## Validation

### Clean rebuild delta

Using a clean comparable CMake rebuild:

- full-tree warnings: `112 -> 111`
- benchmark/example warnings: `14 -> 13`
- `bench_main.c` warnings: `5 -> 4`

Removed warning class:

- `-Wswitch`: `1 -> 0`

Remaining `bench_main.c` warnings:

- `3` `-Wmissing-field-initializers`
- `1` `-Wimplicit-function-declaration`

### Behavioral smoke checks

Successful new path:

- command:
  - `./build/sprint31-day1-cmake/bench_main --size 8 --repeat 1 --reorder nd`
- observed output includes:
  - `Reorder: nd`

Still-rejected unsupported path:

- command:
  - `./build/sprint31-day1-cmake/bench_main --size 8 --repeat 1 --reorder colamd`
- exit code:
  - `1`
- stderr:
  - `Error: unknown reorder mode 'colamd' (use 'none', 'rcm', 'amd', or 'nd')`

## Day 3 Conclusion

The first `bench_main` sync batch landed cleanly:

- `nd` is now exposed consistently in usage text, parsing, and printed labels
- the stale switch warning is gone
- the harness contract now matches the real LU/Cholesky reorder support established in Day 2

The remaining `bench_main` cleanup is now narrower and more mechanical:

- designated-initializer drift
- `snprintf` portability debt
