# Sprint 32 Day 5 `test_reorder_nd.c` Cleanup

**Date:** 2026-05-17  
**Branch:** `sprint-32`

## Objective

Apply the Sprint 32 truthfulness model directly to `tests/test_reorder_nd.c`: remove the dormant historical scaffold, preserve the real active contracts, and clear the file’s remaining initializer and unused-function warnings in the same focused pass.

## What Changed

### Removed dormant historical scaffold

Day 2 identified exactly three dormant compiled test bodies:

1. `test_finest_fm_annealing_pres_poisson_close_to_target`
2. `test_nd_root_spectral_pres_poisson_close_to_target`
3. `test_non_pipeline_pres_poisson_close_to_target`

All three are now gone from `tests/test_reorder_nd.c`.

The corresponding commented-out `RUN_TEST(...)` lines are also gone, so the file no longer represents inactivity via commented-out test registration.

### Preserved the truthful active surface

The active advisory-path checks remain:

- `test_finest_fm_annealing_differs_from_baseline`
- `test_nd_root_spectral_pres_poisson_smoke`
- `test_finest_fm_thick_restart_returns_to_anchor`
- `test_hcc_kuu_safe_corpus_parity`
- `test_per_vertex_fixed_k_three_schemes_differentiate`

Those tests continue to express live current behavior:

- dispatch and non-default-path execution
- parity budgets
- differentiation contracts

That is the coverage the suite actually provides today, and Day 5 keeps it active.

### Closed the file’s initializer warnings

The remaining active option structs in the file now use designated initializers:

- Cholesky ND/AMD path options
- LU ND dispatch options
- LDLT ND dispatch options

This closes the file’s `-Wmissing-field-initializers` debt at the same time as the dormant-scaffold cleanup.

### Removed dead include usage

Deleting the retired supernodal close-to-target stub also made `<errno.h>` unnecessary, so the include was removed.

## Why Deletion Was Correct

Day 3’s policy drew a line between:

- live active or opt-in contracts
- historical evidence

The removed trio belonged to the second category:

- two stubs encoded known-missed Sprint 27 Pres_Poisson close-to-target claims
- one encoded a Sprint 28 target that was formally retired

Keeping them compiled would have preserved exactly the anti-pattern Sprint 32 was meant to eliminate:

- compiled test bodies
- no execution in the active suite
- coverage implied by source shape but not by real test behavior

The relevant measurements and rationale already exist in the Sprint 27 and Sprint 28 decision docs, so deleting the code does not lose project memory.

## Validation

### Focused runtime validation

Validated commands:

- `make format`
- `make build/test_framework_optin build/test_reorder_nd`
- `./build/test_framework_optin`
- `./build/test_reorder_nd`

Results:

- `test_framework_optin` passed
- `test_reorder_nd` passed all `23` active tests

### Clean warning-baseline validation

Validated command:

- `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first`

Measured warning delta versus the Day 1 baseline:

- full-tree warnings: `98 -> 91`
- `-Wmissing-field-initializers`: `62 -> 58`
- `-Wdouble-promotion`: `33 -> 33`
- `-Wunused-function`: `3 -> 0`

Most importantly:

- `tests/test_reorder_nd.c` no longer appears in the clean-build warning output at all

So the file’s original Day 1 queue:

- `4` initializer warnings
- `3` unused-function warnings

is now fully closed.

## End State

`tests/test_reorder_nd.c` now truthfully represents the active suite:

- no dormant compiled historical target stubs
- no commented-out `RUN_TEST(...)`
- no file-local warning debt
- all active ND tests still passing

That closes the highest-signal Sprint 32 truthfulness problem and leaves the remaining sprint work concentrated on the broader test-tree warning queue.
