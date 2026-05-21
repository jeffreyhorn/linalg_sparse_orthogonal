# Sprint 37 Day 9 Large-File Maintainability Batch I

**Date:** 2026-05-20  
**Branch:** `sprint-37`

## Objective

Implement the first large-file maintainability refactor batch chosen on Day 8,
keep the cleanup structural rather than semantic, and make the residual
large-file queue smaller and clearer.

## Batch Scope

Chosen Day 9 files:

1. `Makefile`
2. `scripts/deadcode_report.py`

Unchanged by design:

- giant feature-owner tests
- benchmark-owner CLIs
- `scripts/deadcode_workflow.sh`

## What Changed

### 1. `Makefile`: shared all-test execution helpers

The maintained test-run loop previously existed in three near-copies:

- `test`
- `coverage-lcov`
- `coverage-gcovr`

Day 9 introduced two named helpers:

- `RUN_TEST_BINS_WITH_BANNERS`
- `RUN_TEST_BINS_QUIET`

Landing points:

- helper definitions at lines `213` and `223`
- `test` uses the bannered helper
- both coverage targets use the quiet helper

Why this improves maintainability:

- one authoritative maintained test-run loop now owns failure handling
- direct and coverage flows can share the same behavior without repeated edits
- future loop-shape changes no longer require three synchronized patches

Behavior preserved:

- `test` still prints per-binary banners and `All tests passed.`
- coverage targets still fail if any test binary fails

### 2. `scripts/deadcode_report.py`: report rendering now reads by phase

The markdown report assembly previously lived in one long `write_markdown(...)`
block.

Day 9 extracted the main report sections into dedicated helpers:

- `append_run_metadata`
- `append_coverage_gaps`
- `append_internal_candidates`
- `append_public_surface_items`
- `append_secondary_signals`
- `append_noise_summary`
- `append_next_action_queue`

Support helpers were added for common structure:

- `append_section`
- `append_symbol_rows`
- `public_bucket_reviewed_keeps`

Why this improves maintainability:

- report rendering is now organized by visible report section
- policy-review reads can focus on the specific section being changed
- the script’s Sprint 33/34/36 layering is easier to audit without scanning one
  long render function

Behavior preserved:

- CLI unchanged
- report/check paths unchanged
- report content contract unchanged

## Before / After Shape

### `Makefile`

Before:

- repeated test-bin loops in `test`, `coverage-lcov`, and `coverage-gcovr`

After:

- one bannered shared loop helper
- one quiet shared loop helper
- repeated loop logic removed from the three targets

### `scripts/deadcode_report.py`

Before:

- `472` lines
- `15` top-level functions
- monolithic markdown-render phase

After:

- `504` lines
- `25` top-level functions
- sectioned markdown-render helpers

Interpretation:

- raw line count increased, but the extra lines are structural boundaries and
  helper names rather than new workflow semantics
- this is a maintainability improvement, not a size-reduction exercise

## Validation

Direct touched-path validation passed:

- `python3 -m py_compile scripts/deadcode_report.py`
- `make test`
- `make deadcode-report`
- `make deadcode-check`

Meaning:

- the shared test-loop helpers are behavior-stable on the full maintained suite
- the dead-code report/check flow still produces and validates the expected
  artifacts after the renderer refactor

## Residual Queue After Day 9

Still deferred:

- `scripts/deadcode_workflow.sh`
- `benchmarks/bench_eigs.c`
- `benchmarks/bench_main.c`
- large feature-owner tests:
  - `tests/test_chol_csc.c`
  - `tests/test_svd.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_etree.c`
  - `tests/test_iterative.c`
- wider `Makefile` target-graph density outside the shared test loop
- broader parse/classify/validate layering in `scripts/deadcode_report.py`

## Day 9 Conclusion

The first large-file maintainability batch landed in the right place:

- `Makefile` now has one authoritative maintained test-run loop
- `scripts/deadcode_report.py` now reads by report section instead of through
  one long renderer

Behavior did not change, the maintained support paths stayed green, and the
remaining large-file queue is now smaller and more explicit.
