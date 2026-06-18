# Sprint 79 Day 6 - Differential / Oracle Batch

Date: 2026-06-18  
Branch: sprint-79

## Purpose
Land the first bounded Sprint 79 assurance batch by strengthening the public repeated-run lifecycle oracle and the bounded seeded property lane without widening into broader family-local proof churn or support-surface edits.

## Main Result
Sprint 79 now has one landed first assurance batch in the required implementation center:
- `tests/test_integration.c`
- `tests/test_fuzz.c`

The batch stayed inside the Day 5 fence while strengthening the public repeated-run LDL^T lifecycle assurance story.

## Landed Oracle and Property Work
The public integration oracle now includes:
- `test_public_lifecycle_refactor_same_pattern_matches_one_shot_ldlt`

That test proves:
- the explicit repeated-run LDL^T lifecycle stays aligned with the one-shot CSC-backed LDL^T lane
- same-pattern refactors on a large indefinite KKT family keep the public lifecycle truthful
- the one-shot comparison lane is actually using the CSC path

The bounded seeded property lane now includes:
- `test_property_large_n_ldlt_public_lifecycle_same_pattern_csc`

That property test:
- uses seeds `809u`, `1451u`, and `2029u`
- builds large KKT-style indefinite matrices with:
  - `n_top = SPARSE_CSC_THRESHOLD + 12`
  - `n_bot = 8`
- compares the public repeated-run lifecycle against one-shot LDL^T CSC-backed solves across same-pattern value perturbations
- retains the direct proof output:
  - `large-n LDLT CSC lifecycle property: 3/3 passed`

The supporting local helper seam added in `tests/test_fuzz.c` is intentionally bounded:
- `build_large_kkt(...)`
- `perturb_large_kkt_values_in_place(...)`

## Preserved Fence
The Day 6 batch preserved:
- current public callback/cancel behavior
- current family/path-local caveat reading
- current Windows fuzz exclusion truth
- current benchmark/reporting, install/export, and workflow ownership splits

The batch explicitly did not widen into:
- family-local support proof owners:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
- support-surface wording edits:
  - `docs/maintainer_guide.md`
  - `README.md`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- benchmark/reporting surfaces
- install/export proof scripts
- workflow YAML surfaces
- unrelated implementation or API work

## Validation
Because `*.c` changed, ran:
- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Reviewed anchors:
- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 344.29 sec`

Representative retained proof:
- `test_integration` = `51 / 51`
- `test_fuzz` = `26 / 26`
- retained new oracle:
  - `test_public_lifecycle_refactor_same_pattern_matches_one_shot_ldlt`
- retained new property:
  - `test_property_large_n_ldlt_public_lifecycle_same_pattern_csc`

## Exit State
- The first Sprint 79 assurance batch is landed.
- The public repeated-run LDL^T lifecycle now has both a bounded oracle test and a bounded seeded large-`n` property test.
- Sprint 79 can now rerank the next final-assurance seam from a stronger public lifecycle baseline.
