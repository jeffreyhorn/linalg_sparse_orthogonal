# Sprint 192 Day 11: Failure and Drift Coverage

## Summary

Day 11 hardened failure coverage for the selected methodology-bound
performance lane. The selected benchmark freshness checker now validates the
selected CSV artifact contents instead of only checking that the file exists,
and the documentation guard now has targeted failures for threshold-free policy
drift and hosted timing-gate overclaims.

## Source Changes

| Surface | Day 11 change |
| --- | --- |
| `scripts/check_bench_canonical_freshness.py` | Added selected CSV artifact validation for required columns, row count, stable fixture fields, and matrix-size agreement. |
| `tests/test_bench_canonical_freshness.py` | Added negative tests for wrong selected CSV fixture, missing required CSV column, and duplicate selected CSV rows. |
| `tests/test_selected_performance_docs.py` | Added negative tests for missing `threshold=n/a` policy text and hosted timing-gate overclaim wording. |

## Checker Coverage

The selected checker now validates `bench_refactor_csc.csv` for:

- required selected fields: `benchmark`, `matrix`, `n`, `scenario`,
  `ldlt_dense_backend_request`, `ldlt_dense_backend_selected`, and
  `ldlt_dense_backend_fallback`;
- exactly one selected CSV data row;
- `benchmark=bench_refactor_csc`;
- `matrix=nos4.mtx`;
- `n=100`;
- `scenario=chol_spd`;
- LDLT backend fields set to `n/a`;
- `index.tsv` `matrix_size` agreement with CSV `n`.

This keeps malformed selected CSV data from passing the selected freshness
gate and then appearing as valid normalized report context.

## Drift Tests

New benchmark freshness regressions fail clearly when:

- the selected CSV fixture is changed to an unsupported matrix;
- a required selected CSV column is missing;
- the selected CSV contains more than one data row.

New docs guard regressions fail clearly when:

- `threshold=n/a` disappears from report-index schema guidance;
- docs claim hosted selected performance is a timing gate.

## Non-Goals

Day 11 did not add a hosted timing threshold, broaden artifact upload scope,
change benchmark C sources, or promote unselected canonical benchmark rows.
The lane remains selected-only and threshold-free.

## Validation

Commands run:

```sh
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_selected_performance_docs.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 -m py_compile scripts/check_bench_canonical_freshness.py tests/test_bench_canonical_freshness.py tests/test_selected_performance_docs.py tests/test_selected_comparison_workflow.py tests/test_normalize_report_index.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected benchmark freshness tests passed, including the new selected CSV
  failure coverage;
- selected-performance docs guard passed, including new threshold-free and
  timing-gate drift coverage;
- selected workflow guard tests passed;
- report-index normalization tests passed;
- selected target schema validation passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 11.

## Day 12 Inputs

Day 12 integrated validation should inspect the generated selected CSV,
`index.tsv`, and `manifest.txt` together and confirm the selected metadata
matches the hosted threshold-free policy recorded by Days 8-11.
