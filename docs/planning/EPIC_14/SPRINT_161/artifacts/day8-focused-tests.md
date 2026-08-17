# Day 8 Focused Tests

## Summary

Day 8 promotes the selected partial-SVD comparison rows into the normalizer's
selected comparison freshness gate and adds focused tests for the expanded row
set. The selected comparison family now requires the existing QR rows plus the
ten `partial_svd_diag6_k2` rows.

## Files Changed

| File | Change |
| --- | --- |
| `scripts/normalize_report_index.py` | Added the ten selected `partial_svd_diag6_k2` comparison row IDs and the partial-SVD study artifact to selected comparison freshness. |
| `tests/test_normalize_report_index.py` | Expanded the synthetic selected comparison row set and row-state tests to include partial-SVD rows. |
| `Makefile` | Added `python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2` to `report-index-comparison-freshness`. |
| `docs/planning/EPIC_14/SPRINT_161/WORKING_NOTES.md` | Added Day 8 log entry. |

## Selected Comparison Row Set

The selected comparison freshness set now contains `22` generated rows:

- six `qr_underdetermined_minnorm_2x4` rows;
- six `qr_overdetermined_compatible_5x3` rows;
- ten `partial_svd_diag6_k2` rows.

The artifact diagnostic now names:

```text
artifacts=build/comparison/qr_minnorm/study.tsv,build/comparison/qr_compatible_ls/study.tsv,build/comparison/partial_svd_diag6_k2/study.tsv
```

## Focused Row-State Coverage

`tests/test_normalize_report_index.py` now verifies:

| Case | Coverage |
| --- | --- |
| Complete selected row set | QR plus partial-SVD rows pass required comparison freshness. |
| Missing selected row | Missing `comparison_partial_svd_diag6_k2_v_projector_diag_v1` fails with row-set mismatch. |
| Unexpected selected-family row | Unexpected `comparison_partial_svd_diag6_k2_unexpected_metric_v1` fails with row-set mismatch. |
| Duplicate selected row | Duplicate `comparison_partial_svd_diag6_k2_project_status_v1` fails duplicate normalized row detection. |
| Stale selected row | Stale partial-SVD selected row fails source-commit freshness. |
| Failing selected row | Failing partial-SVD selected row fails `comparison_selected_status`. |
| Deferred selected row | Deferred partial-SVD selected row fails and remains non-proof context. |
| Skipped selected row | Skipped partial-SVD selected row fails and remains non-proof context. |

## Makefile Freshness Alignment

`make report-index-comparison-freshness` now regenerates:

```sh
python3 scripts/run_external_comparison.py --target qr-minnorm
python3 scripts/run_external_comparison.py --target qr-compatible-ls
python3 scripts/run_external_comparison.py --target partial-svd-diag6-k2
```

Then it runs the existing selected comparison freshness check. This keeps the
Makefile target aligned with the expanded normalizer selected row set.

## C Proof-Owner Decision

No C proof-owner tests were added. Day 8 touched Python, Makefile, tests, and
planning docs only. Public solver behavior and headers were unchanged.

## Validation

Commands run:

```sh
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
python3 -m py_compile scripts/run_external_comparison.py tests/test_run_external_comparison.py scripts/validate_corpus_schema.py scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_run_external_comparison.py
python3 scripts/validate_corpus_schema.py
make report-index-comparison-freshness
```

Results:

- Normalizer compile check passed.
- Focused normalizer test passed.
- Expanded Python compile check passed.
- Focused external-comparison runner test passed.
- Corpus schema validation passed.
- `make report-index-comparison-freshness` passed and reported
  `normalize-report-index: freshness ok (25 rows)`.

No `.c` or `.h` files were modified.

## Day 9 Handoff

Day 9 should document the normalized row design and evidence-tier
classification now that the selected partial-SVD comparison family is wired
into local-only generated comparison freshness.
