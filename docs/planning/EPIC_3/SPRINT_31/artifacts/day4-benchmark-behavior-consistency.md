# Sprint 31 Day 4 Benchmark Behavior Consistency

**Date:** 2026-05-17  
**Branch:** `sprint-31`

## Objective

Align reorder-mode presentation across the main benchmark harness and the smaller reorder-related benchmark/example tools so the user-visible labels match after Day 3’s `bench_main` contract update.

## Pre-Day-4 Drift

Observed from live tool output and source:

- `bench_main`
  - used lowercase CLI-style spellings: `none`, `rcm`, `amd`, `nd`
- `bench_reorder`
  - emitted uppercase enum-style labels in CSV rows: `NONE`, `RCM`, `AMD`, `COLAMD`, `ND`
- `bench_colamd`
  - emitted row values using `none=`, `amd=`, `colamd=`
  - but titled the comparison as `COLAMD vs AMD vs Natural`
  - and used header shorthand `nnz(R)nat` / `nnz(R)col`
- `example_colamd`
  - described the LU comparison as `Natural ordering` vs `COLAMD ordering`

Interpretation:

- the reorder behavior itself was already stable
- the user-facing naming was not
- after Day 3, the most obvious remaining inconsistency was that the tools were mixing:
  - lowercase CLI spellings
  - uppercase enum labels
  - `natural` prose
  - abbreviated column headers like `nat` and `col`

## Day 4 Changes

### `bench_reorder`

Changed the emitted CSV reorder labels to lowercase CLI-style spellings:

- `none`
- `rcm`
- `amd`
- `colamd`
- `nd`

Result:

- the CSV output now uses the same spellings a user would type into `bench_main --reorder ...`

### `bench_colamd`

Changed presentation-only labels:

- title:
  - from `COLAMD vs AMD vs Natural`
  - to `colamd vs amd vs none`
- headers:
  - from `nnz(R)nat`, `nnz(R)amd`, `nnz(R)col`
  - to `nnz(R)none`, `nnz(R)amd`, `nnz(R)colamd`
- percentage text:
  - from `vs natural`
  - to `vs none`

Result:

- the header terminology now matches the row labels printed by the program itself

### `example_colamd`

Changed presentation-only labels in the LU fill-in comparison:

- from `Natural ordering`
- to `none (natural order)`

and:

- from `COLAMD ordering`
- to `colamd`

Result:

- the example now uses the same naming convention as the benchmark tools while still preserving the explanatory note that `none` corresponds to natural order

## Post-Day-4 Consistency Matrix

| tool | supported / shown reorder names | user-facing style after Day 4 |
|---|---|---|
| `bench_main` | `none`, `rcm`, `amd`, `nd` | lowercase CLI spellings |
| `bench_reorder` | `none`, `rcm`, `amd`, `colamd`, `nd` | lowercase CSV labels |
| `bench_colamd` | `none`, `amd`, `colamd` | lowercase title, headers, row labels |
| `example_colamd` | `none (natural order)`, `colamd` | lowercase labels with explanatory parenthetical |

## Validation

Post-edit live outputs confirmed:

- `bench_reorder --only nos4 --skip-factor` now prints rows with lowercase reorder names
- `bench_colamd` now prints `nnz(R)none`, `nnz(R)amd`, `nnz(R)colamd` and `% vs none`
- `example_colamd` now prints `none (natural order)` and `colamd`

Targeted rebuilds for:

- `bench_reorder`
- `bench_colamd`
- `example_colamd`

all completed successfully.

## Day 4 Conclusion

Day 4 closed the remaining naming drift without changing actual reorder behavior:

- the main benchmark tools now share a consistent lowercase spelling convention
- the specialized COLAMD tools no longer mix `none` labels with `natural`/`nat` presentation
- the next Sprint 31 work can focus on portability and initializer cleanup rather than output-label confusion
