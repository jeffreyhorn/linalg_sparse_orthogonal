# Sprint 33 Day 7 Report Wiring And First Report

**Date:** 2026-05-18  
**Branch:** `sprint-33`

## Objective

Implement the reporting targets planned on Day 6 and generate the first classified dead-code report so Sprint 33 has a concrete, auditable candidate queue rather than only raw scanner output.

## What Shipped

Day 7 added:

- `make deadcode-report`
- `make deadcode-check`
- `scripts/deadcode_report.py`

The new flow is:

1. `make deadcode`
2. `python3 scripts/deadcode_report.py build/deadcode`
3. optionally `python3 scripts/deadcode_report.py --check build/deadcode`

Operator-facing effect:

- `make deadcode-report`
  - refreshes the raw artifacts
  - writes:
    - `build/deadcode/report.md`
    - `build/deadcode/report.tsv`
- `make deadcode-check`
  - regenerates the report
  - validates report completeness/category hygiene
  - does not fail merely because findings still exist this sprint

## Report Shape

Generated human report:

- `build/deadcode/report.md`

Generated stable findings table:

- `build/deadcode/report.tsv`

The report implementation follows the Day 6 bucket model:

- `coverage-gap`
- `definitely-unused-internal-candidate`
- `public-surface-review`
- `secondary-candidate-signal`
- `non-deadcode-static-analysis-noise`

## `deadcode-check` Contract

Current enforced behavior:

1. report generation must succeed
2. the coverage-gap section must be present in `report.md`
3. every `xunused` finding must be categorized into either:
   - `definitely-unused-internal-candidate`
   - `public-surface-review`

Current intentionally *not* enforced:

- empty findings set
- empty cleanup queue
- zero `cppcheck` secondary/noise counts

That matches the Sprint 33 sequence:

- Day 8 still needs public-surface review
- Day 9 through Day 11 still need cleanup batching and removals

## First Classified Report Results

Current normalized bucket counts:

- `coverage-gap`: `7`
- `definitely-unused-internal-candidate`: `1`
- `public-surface-review`: `4`
- `secondary-candidate-signal`: `35`
- `non-deadcode-static-analysis-noise`: `6`

### Coverage-gap bucket

Still missing from the current compile-db:

- `bench_svd`
- `example_basic_solve`
- `example_condition`
- `example_iterative`
- `example_least_squares`
- `example_matrix_free`
- `example_svd_lowrank`

### Definitely-unused internal candidate

The first cleanup-ready internal queue is currently one symbol:

- `chol_csc_dump_supernodes`

Day 7 interpretation:

- this is the starting point for Day 9 cleanup design
- it stays pending until Day 8 confirms there is no public-surface counter-evidence nearby

### Public-surface review queue

The Day 8 audit set is now explicit:

- `givens_apply_right`
- `sparse_print_dense`
- `sparse_print_entries`
- `sparse_print_info`

All four are currently deferred because their declarations live in installed headers.

### Secondary `cppcheck` evidence

Top files by secondary dead-code-adjacent signal in the report:

- `src/sparse_chol_csc.c`: `20`
- `src/sparse_ldlt_csc.c`: `17`
- `src/sparse_matrix.c`: `16`
- `src/sparse_qr.c`: `14`
- `src/sparse_graph.c`: `14`

Interpretation:

- these are review-priority hints only
- they are not part of the Day 7 cleanup-ready queue

## Validation

Day 7 validation:

1. `python3 -m py_compile scripts/deadcode_report.py`
2. `make deadcode-report`
3. `make deadcode-check`

Results:

- report generation: passed
- completeness check: passed
- first classified report: generated successfully

## Day 7 Conclusion

Sprint 33 now has a working report layer rather than only raw scanner logs.

The most important result is not the number of rows in `report.tsv`; it is the narrowed action queue:

- one internal cleanup candidate for Day 9 planning
- four public-surface items for Day 8 review
- compile-db blind spots preserved explicitly
- broad `cppcheck` signal held in a supporting bucket instead of being mistaken for a direct deletion list
