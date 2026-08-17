# Day 10 Report Integration

## Summary

Day 10 verifies the selected partial-SVD comparison rows in normalized report
output and strengthens the focused normalizer regression test. The selected
comparison freshness gate now has end-to-end local coverage for generated QR
and partial-SVD comparison rows.

## Files Changed

| File | Change |
| --- | --- |
| `tests/test_normalize_report_index.py` | Added explicit assertions that normalized output contains all ten generated `partial_svd_diag6_k2` comparison rows with `pass` status, `local_only` support tier, the selected study artifact, and raw-vector-identity non-claims. |
| `docs/planning/EPIC_14/SPRINT_161/WORKING_NOTES.md` | Added Day 10 log entry. |

The core report-index wiring was completed on Day 8; Day 10 verifies and
hardens the observable normalized output.

## Normalized Output Verification

Command:

```sh
python3 scripts/normalize_report_index.py --family comparison --output /tmp/sprint161-day10-normalized.tsv
```

Observed partial-SVD rows included:

- `report_contract_comparison_partial_svd_diag6_k2_external_process_dense_reference_comparison_v1`
- `comparison_partial_svd_diag6_k2_project_status_v1`
- `comparison_partial_svd_diag6_k2_baseline_status_v1`
- `comparison_partial_svd_diag6_k2_singular_value_0_v1`
- `comparison_partial_svd_diag6_k2_singular_value_1_v1`
- `comparison_partial_svd_diag6_k2_singular_values_max_abs_delta_v1`
- `comparison_partial_svd_diag6_k2_residual_norm_v1`
- `comparison_partial_svd_diag6_k2_u_orthogonality_v1`
- `comparison_partial_svd_diag6_k2_v_orthogonality_v1`
- `comparison_partial_svd_diag6_k2_u_projector_diag_v1`
- `comparison_partial_svd_diag6_k2_v_projector_diag_v1`

The generated rows normalize as `comparison` / `partial_svd_diag6_k2`,
`generated_local`, `pass`, and `local_only`, with
`build/comparison/partial_svd_diag6_k2/study.tsv` as the selected artifact.

## Freshness Validation

Command:

```sh
make report-index-comparison-freshness
```

Result:

```text
normalize-report-index: freshness ok (25 rows)
report-index-comparison-freshness: passed (local-only generated comparison freshness)
```

The `25` normalized freshness rows are:

- three source-controlled comparison contract rows;
- twelve selected QR generated comparison rows;
- ten selected partial-SVD generated comparison rows.

## Focused Regression Coverage

`tests/test_normalize_report_index.py` now verifies the complete selected
comparison case contains all ten normalized partial-SVD rows and preserves:

- `pass` status;
- `local_only` support tier;
- selected partial-SVD study artifact identity;
- raw singular-vector identity non-claim boundary.

The existing focused cases continue to cover stale, missing, duplicate,
unexpected, fail, defer, and skip behavior for the expanded selected row set.

## Validation

Commands run:

```sh
make report-index-comparison-freshness
python3 scripts/normalize_report_index.py --family comparison --output /tmp/sprint161-day10-normalized.tsv
python3 -m py_compile scripts/normalize_report_index.py tests/test_normalize_report_index.py
python3 tests/test_normalize_report_index.py
python3 tests/test_run_external_comparison.py
python3 scripts/validate_corpus_schema.py
```

Results:

- Selected comparison freshness passed locally.
- Normalized output contains the selected partial-SVD comparison rows.
- Python compile check passed.
- Focused normalizer test passed.
- Focused external-comparison runner test passed.
- Corpus schema validation passed.

No `.c` or `.h` files were modified.

## Day 11 Handoff

Day 11 should update public and maintainer documentation that still describes
the selected comparison freshness gate as QR-only. The correct wording is
selected fixture-local QR plus partial-SVD comparison freshness, with
`partial_svd_diag6_k2` remaining `local_only` and non-parity.
