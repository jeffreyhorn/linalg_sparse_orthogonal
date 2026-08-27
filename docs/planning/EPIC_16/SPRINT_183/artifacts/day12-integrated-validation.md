# Sprint 183 Day 12: Integrated Validation

## Scope

Day 12 ran the integrated local validation pass for the selected Cholesky SPD
tridiagonal comparison family and the retained non-claim surfaces touched by
Sprint 183.

## Validation Summary

| Command | Status | Notes |
| --- | --- | --- |
| `python3 tests/test_chol_external_dense_reference.py` | Pass | Cholesky helper fixture, CLI, missing-file skip, and unknown-fixture diagnostics. |
| `python3 tests/test_run_external_comparison.py` | Pass | Runner target, rows, helper dispatch, metadata, and unsupported-target diagnostics. |
| `python3 tests/test_selected_comparison_workflow.py` | Pass | Linux/macOS selected comparison workflow guards and Windows non-promotion checks. |
| `python3 tests/test_selected_report_targets_manifest.py` | Pass | Selected target manifest shape and metadata checks. |
| `python3 tests/test_normalize_report_index.py` | Pass | Selected row IDs, artifact diagnostics, and report-index normalization fixtures. |
| `python3 scripts/validate_corpus_schema.py` | Pass | Corpus schema and manifest validation. |
| `make test_cholesky` | Superseded | No such Makefile target exists in this repository. |
| `build/test_cholesky` | Pass | Actual Cholesky test binary; 21 tests passed, 0 failed. |
| `make report-index-comparison-freshness` | Pass | Selected freshness passed with 39 comparison rows. |
| `bash scripts/static_package_deferral_check.sh` | Pass | Static package and shared ABI non-claim guard. |
| `bash scripts/package_manager_deferral_check.sh` | Pass | Package-manager deferral and public non-claim guard. |
| `make format` | Pass | No tracked C/header diffs introduced. |
| `make lint` | Pass | Strict warning syntax check, clang-tidy, and cppcheck completed. |
| `make test` | Pass | Full repository test suite passed. |

## Generated Artifact Check

`make report-index-comparison-freshness` regenerated local selected comparison
outputs under `build/comparison/`, including
`build/comparison/cholesky_spd_tridiag_5/`. Those generated outputs remain
ignored and unstaged. `build/report-index` also remains unstaged.

## Residual Risk

- Day 12 validation is local on the current development machine; hosted
  Linux/macOS lanes will still provide the reviewed CI execution proof.
- Windows report freshness remains intentionally deferred. Day 12 confirmed
  the guard path still rejects Windows selected freshness promotion.
- The selected Cholesky claim remains fixture-local to
  `cholesky_spd_tridiag_5`; it does not widen to broad Cholesky correctness,
  broad SPD coverage, reordering parity, CSC-vs-linked-list parity, fill
  superiority, external-library parity, package/platform proof, performance,
  release, or state-of-the-art evidence.

## Validation Hygiene

Python `__pycache__` files created by validation were removed. No generated
comparison or report-index outputs were staged.

## Final Checks

| Command | Status |
| --- | --- |
| `git status --short -- build/comparison build/report-index` | Pass |
| `git diff --check` | Pass |
