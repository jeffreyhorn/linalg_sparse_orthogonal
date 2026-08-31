# Sprint 190 Day 7: Freshness Guard

## Purpose

Implement the narrow freshness guard needed for the Sprint 190 Windows selected
report decision without promoting Windows workflow or manifest metadata before
hosted evidence exists.

## Guard Implemented

`scripts/normalize_report_index.py` now accepts a target-specific selected
comparison freshness argument:

```sh
python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5
```

The guard limits selected comparison freshness validation to the matching
`target_key` from `tests/corpus/manifests/selected_report_targets.tsv`.

For `cholesky-spd-tridiag-5`, the guard checks:

- selected target identity from the manifest;
- expected Cholesky row IDs and expected row count;
- generated-local row freshness against the current Git commit;
- non-pass generated rows;
- artifact path identity for
  `build/comparison/cholesky_spd_tridiag_5/study.tsv`;
- stale `source_commit` diagnostics with the exact target-specific remediation
  command.

## Diagnostic Contract

The target-specific guard emits Cholesky-only diagnostics for missing output,
stale output, failed rows, row-set mismatch, duplicate rows, and unexpected
rows.

Required remediation is target-specific:

```sh
run python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5
```

This avoids directing a Windows-only Cholesky lane to run the broader
Linux/macOS selected comparison freshness target.

## Code Changes

- `selected_comparison_contracts()` can filter selected comparison contracts
  by `target_key`.
- Selected comparison expected row IDs, row counts, artifacts, subfamilies, and
  artifact diagnostics all accept the same optional target filter.
- Generated comparison policy diagnostics filter to selected artifacts while
  accepting both repo-relative and temporary build-root artifact paths.
- Freshness diagnostics skip non-selected comparison subfamilies when
  `--selected-target` is provided.
- Stale and failed generated comparison diagnostics include the exact
  target-specific remediation command.

## Test Coverage

`tests/test_normalize_report_index.py` now covers:

- missing target-specific Cholesky output reports Cholesky artifact paths only;
- Cholesky-only generated output passes without requiring QR, LU, or partial-SVD
  rows;
- stale Cholesky `source_commit` output fails with target-specific
  remediation;
- failed Cholesky generated rows fail with Cholesky artifact diagnostics;
- broader selected comparison row-set mismatch coverage remains active.

The synthetic selected comparison writer gained an `only_subfamilies` filter so
tests can model the future Windows Cholesky-only lane without weakening the
full selected comparison freshness tests.

## Non-Promotion Boundary

Day 7 does not add:

- Windows selected report manifest metadata;
- a Windows selected comparison workflow job;
- a Windows selected comparison artifact upload;
- broad Windows report freshness claims.

The guard is intentionally available before workflow promotion so Day 8 can
wire hosted integration to a reviewed command instead of inventing new
freshness semantics in YAML.

## Validation

Commands run:

- `python3 tests/test_normalize_report_index.py`
- `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_validate_windows_powershell.py`

All focused validation commands passed.

No `.c` or `.h` files were modified, so the full `make format && make lint &&
make test` C gate is not required for Day 7.
