# Sprint 32 Day 7 Initializer Batch Design

**Date:** 2026-05-18  
**Branch:** `sprint-32`

## Objective

Audit the remaining test-side designated-initializer warning queue after the Day 5 `test_reorder_nd.c` cleanup and turn it into a tight two-day implementation plan for Sprint 32 Days 8 and 9.

## Authoritative Current State

Day 7 reran the clean serialized Apple Clang CMake build for the current branch:

- full-tree warnings: `91`
- `src`: `0`
- `tests`: `91`
- `benchmarks`: `0`
- `examples`: `0`
- `-Wmissing-field-initializers`: `58`
- `-Wdouble-promotion`: `33`

Compared to the Day 1 baseline:

- full-tree warnings: `98 -> 91`
- `-Wmissing-field-initializers`: `62 -> 58`
- `-Wdouble-promotion`: unchanged at `33`
- `-Wunused-function`: `3 -> 0`

That means Sprint 32's truthfulness work is complete enough that the remaining initializer queue can be planned as pure mechanical cleanup.

## Remaining Initializer Queue

The current `-Wmissing-field-initializers` sites are concentrated in `10` files:

| File | Count | Main struct family |
| --- | ---: | --- |
| `tests/test_ldlt.c` | 18 | `sparse_ldlt_opts_t` |
| `tests/test_chol_csc.c` | 8 | `sparse_cholesky_opts_t` |
| `tests/test_colamd.c` | 7 | `sparse_qr_opts_t` |
| `tests/test_cholesky.c` | 5 | `sparse_cholesky_opts_t` |
| `tests/test_sprint12_integration.c` | 5 | `sparse_ldlt_opts_t` |
| `tests/test_sprint20_integration.c` | 5 | `sparse_ldlt_opts_t` |
| `tests/test_reorder.c` | 4 | `sparse_lu_opts_t` |
| `tests/test_sprint18_integration.c` | 3 | `sparse_cholesky_opts_t` |
| `tests/test_sprint19_integration.c` | 2 | `sparse_cholesky_opts_t` |
| `tests/test_etree.c` | 1 | `sparse_lu_opts_t` |

Total: `58`

## Pattern Audit

The remaining warnings are the same family of ABI-growth fallout already fixed in Sprint 31 benchmarks/examples:

### Cholesky family

Struct growth that matters:

- `backend`
- `used_csc_path`
- `progress_cb`
- `progress_user`

Observed remaining forms:

- one-field positional reordering-only forms such as `{SPARSE_REORDER_AMD}`
- three-field backend/telemetry forms such as `{SPARSE_REORDER_AMD, SPARSE_CHOL_BACKEND_AUTO, &used}`
- one brittle two-field form in `tests/test_chol_csc.c` where `0.0` currently lands in the enum-valued `backend` slot only because zero maps to `SPARSE_CHOL_BACKEND_AUTO`

### LDLT family

Struct growth that matters:

- `backend`
- `used_csc_path`
- `progress_cb`
- `progress_user`

Observed remaining forms:

- two-field legacy forms such as `{SPARSE_REORDER_AMD, 0.0}`
- four-field backend/telemetry forms such as `{SPARSE_REORDER_NONE, 0.0, SPARSE_LDLT_BACKEND_AUTO, &used_csc}`

### QR family

Struct growth that matters:

- `progress_cb`
- `progress_user`

Observed remaining forms:

- `{SPARSE_REORDER_COLAMD, 0, 0}`
- `{SPARSE_REORDER_COLAMD, 0, 1}`

The intent is already clear in source terms:

- `.reorder` is the main override
- `.economy` and `.sparse_mode` are occasionally intentional overrides
- callback/context should stay at default `NULL`

### LU family

Struct growth that matters:

- `progress_cb`
- `progress_user`

Observed remaining forms:

- `{SPARSE_PIVOT_PARTIAL, SPARSE_REORDER_AMD, 1e-12}`
- same pattern in the etree companion file

These are straightforward designated-init conversions:

- `.pivot`
- `.reorder`
- `.tol`

## Chosen Batch Order

### Day 8 Batch I: LDLT family

Files:

- `tests/test_ldlt.c`
- `tests/test_sprint12_integration.c`
- `tests/test_sprint20_integration.c`

Why this batch is first:

- same option struct across all touched files
- same two mechanical sub-patterns
- highest single-family payoff: `28 / 58` warnings
- `test_sprint20_integration.c` is mixed-class, but its initializer cleanup is still LDLT-specific and should land with the LDLT family rather than be deferred to the later double-promotion work

Expected end state for the batch:

- every touched LDLT test names only the non-default fields it intends to override
- legacy positional `{reorder, tol}` and `{reorder, tol, backend, used_csc_path}` forms disappear
- residual warnings in `test_sprint20_integration.c` after Day 8 should be only its `-Wdouble-promotion` sites

### Day 9 Batch II: Cholesky + QR + LU companion family

Files:

- `tests/test_chol_csc.c`
- `tests/test_cholesky.c`
- `tests/test_sprint18_integration.c`
- `tests/test_sprint19_integration.c`
- `tests/test_colamd.c`
- `tests/test_reorder.c`
- `tests/test_etree.c`

Why these belong together:

- the remaining queue after Day 8 is `30` warnings, small enough for one focused day
- the files separate naturally into three small same-API subgroups:
  - Cholesky backend/callback family
  - QR callback family
  - LU callback family
- `tests/test_etree.c` has only one warning and matches `tests/test_reorder.c` exactly in struct family, so there is no value in stranding it for a later sprint

Expected end state for the batch:

- all residual Sprint 32 `-Wmissing-field-initializers` sites are gone
- the remaining warning queue becomes purely double-promotion work

## Reused Coding Rule

Sprint 32 should reuse the Sprint 31 designated-init rule exactly:

1. Name only the fields the test intentionally overrides.
2. Let default-valued trailing backend, telemetry, callback, and context fields zero-initialize.
3. Do not mirror evolving public option-struct layouts positionally.

Examples of the intended conversion style:

- LU: `{.pivot = SPARSE_PIVOT_PARTIAL, .reorder = SPARSE_REORDER_AMD, .tol = 1e-12}`
- QR reorder-only: `{.reorder = SPARSE_REORDER_COLAMD}`
- QR sparse-mode: `{.reorder = SPARSE_REORDER_COLAMD, .sparse_mode = 1}`
- Cholesky reorder-only: `{.reorder = SPARSE_REORDER_AMD}`
- Cholesky forced backend: `{.reorder = SPARSE_REORDER_AMD, .backend = SPARSE_CHOL_BACKEND_CSC, .used_csc_path = &used}`
- LDLT forced backend: `{.reorder = SPARSE_REORDER_NONE, .backend = SPARSE_LDLT_BACKEND_AUTO, .used_csc_path = &used_csc}`

## Validation Plan

### Day 8 validation

- `make format`
- targeted build of:
  - `test_ldlt`
  - `test_sprint12_integration`
  - `test_sprint20_integration`
- run those binaries directly
- clean serialized CMake rebuild for the warning delta

### Day 9 validation

- `make format`
- targeted build of:
  - `test_chol_csc`
  - `test_cholesky`
  - `test_sprint18_integration`
  - `test_sprint19_integration`
  - `test_colamd`
  - `test_reorder`
  - `test_etree`
- run the highest-signal touched binaries directly
- clean serialized CMake rebuild for final initializer closure

## Conclusion

The Day 7 audit leaves a clean two-step plan:

- Day 8 closes the LDLT-family initializer debt (`28` warnings)
- Day 9 closes the remaining Cholesky, QR, and LU companion debt (`30` warnings)

That is the tightest sequencing that keeps each edit batch coherent, keeps validation local to the edited API family, and avoids mixing initializer cleanup with the later double-promotion work before the initializer queue is fully closed.
