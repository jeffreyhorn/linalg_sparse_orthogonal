# Sprint 32 Day 10 Double-Promotion Batch Design

**Date:** 2026-05-18  
**Branch:** `sprint-32`

## Objective

Audit the residual Sprint 32 `-Wdouble-promotion` queue and turn it into a precise implementation plan for Days 11 and 12.

## Authoritative Current State

Day 10 reused the clean serialized Apple Clang CMake rebuild captured after Day 9:

- full-tree warnings: `33`
- `src`: `0`
- `tests`: `33`
- `benchmarks`: `0`
- `examples`: `0`
- `-Wdouble-promotion`: `33`

Compared to the Day 1 baseline:

- full-tree warnings: `98 -> 33`
- `-Wmissing-field-initializers`: `62 -> 0`
- `-Wunused-function`: `3 -> 0`

That leaves Sprint 32 with a single active warning class and no mixed warning debt.

## Residual Queue by File

| File | Count | Dominant pattern |
| --- | ---: | --- |
| `tests/test_sprint6_integration.c` | 6 | `INFINITY` return and `double ... = INFINITY` sentinels |
| `tests/test_svd.c` | 6 | `INFINITY` return/comparisons plus one `NAN` local |
| `tests/test_sprint20_integration.c` | 4 | `INFINITY` return sentinels |
| `tests/test_bidiag.c` | 3 | `INFINITY` return sentinels |
| `tests/test_sprint18_integration.c` | 3 | `INFINITY` return and local sentinels |
| `tests/test_qr.c` | 2 | `INFINITY` return sentinels |
| `tests/test_sprint10_integration.c` | 2 | `INFINITY` return sentinels |
| `tests/test_bicgstab.c` | 1 | `INFINITY` return sentinel |
| `tests/test_block_solvers.c` | 1 | `INFINITY` return sentinel |
| `tests/test_colamd.c` | 1 | `INFINITY` return sentinel |
| `tests/test_ilu.c` | 1 | `INFINITY` return sentinel |
| `tests/test_lu_csr.c` | 1 | `INFINITY` return sentinel |
| `tests/test_sprint19_integration.c` | 1 | `INFINITY` return sentinel |
| `tests/test_sprint5_integration.c` | 1 | `INFINITY` return sentinel |

Total: `33`

## Pattern Audit

The warning-bearing files were searched specifically for alternative double-promotion causes:

- `fabsf`
- `sqrtf`
- `powf`
- float-suffixed literals such as `0.0f` / `1.0f`
- mixed float-temporary arithmetic

None of those patterns explain the active Sprint 32 warnings.

The residual queue breaks down as:

- `32` `INFINITY` sites in double contexts
- `1` `NAN` site in a double context

That split matters because it keeps the implementation scope honest:

- this is not a numeric-algorithm audit
- this is not tolerance retuning
- this is not mixed-precision arithmetic cleanup
- this is sentinel-macro cleanup

## Chosen Replacement Idioms

### Helper return / local sentinel rule

Use `HUGE_VAL` instead of `INFINITY` whenever the target type is `double`.

Representative conversions:

- `return INFINITY;` -> `return HUGE_VAL;`
- `double rel = INFINITY;` -> `double rel = HUGE_VAL;`

### NaN local rule

Use `nan("")` instead of `NAN` when initializing a `double`.

Representative conversion:

- `double recon = NAN;` -> `double recon = nan("");`

### Infinity assertion rule

Use a semantic infinity check instead of exact equality against the macro.

Representative conversion:

- `ASSERT_TRUE(c == INFINITY);` -> `ASSERT_TRUE(isinf(c) && c > 0.0);`

Why this is the chosen assertion idiom:

- it avoids the float-to-double promotion warning
- it states the actual test contract
- it keeps the expectation on positive infinity explicit

## Chosen Batch Order

### Day 11 Batch I: helper-sentinel cleanup

Files:

- `tests/test_ilu.c`
- `tests/test_sprint5_integration.c`
- `tests/test_qr.c`
- `tests/test_sprint6_integration.c`
- `tests/test_bidiag.c`
- `tests/test_lu_csr.c`
- `tests/test_block_solvers.c`
- `tests/test_sprint10_integration.c`
- `tests/test_colamd.c`
- `tests/test_bicgstab.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_sprint20_integration.c`

Count:

- `27` warnings

Why this batch is first:

- every site uses the same `HUGE_VAL` replacement rule
- most are helper functions for residual or reconstruction measurement
- the behavioral risk is low and validation can be kept local to the touched binaries

Expected end state after Day 11:

- all non-SVD Sprint 32 double-promotion warnings are gone
- residual warning queue should shrink from `33` to `6`
- the only remaining warning file should be `tests/test_svd.c`

### Day 12 Batch II: SVD sentinel and assertion cleanup

File:

- `tests/test_svd.c`

Count:

- `6` warnings

Why this file stands alone:

- it is the only file mixing:
  - `INFINITY` helper returns
  - a `NAN` local sentinel
  - infinity assertions in condition-number coverage
- the replacements are still mechanical, but the file uses two distinct Day 10 idioms beyond the generic `HUGE_VAL` conversion

Expected end state after Day 12:

- Sprint 32's `-Wdouble-promotion` queue closes completely
- warning reconciliation can happen against a single class with no ambiguity

## Validation Plan

### Day 11 validation

- `make format`
- targeted build of the touched Batch I test binaries
- direct execution of the touched highest-signal tests
- clean serialized Apple Clang CMake rebuild for the warning delta

### Day 12 validation

- `make format`
- targeted `test_svd` build and run
- clean serialized Apple Clang CMake rebuild
- reconcile final warning counts against:
  - Day 1 baseline
  - Day 9 post-initializer state
  - Day 11 post-helper cleanup state

## Conclusion

Day 10 narrowed Sprint 32's remaining warning work to a very small and honest scope:

- `33` warnings remain
- `32` are `INFINITY` in double contexts
- `1` is `NAN` in a double context
- no active warning site requires a broader numeric refactor

That leaves a simple two-step finish:

- Day 11 removes the `27` helper-sentinel sites with the `HUGE_VAL` rule
- Day 12 closes the `6` specialized `tests/test_svd.c` sites and reconciles the warning inventory
