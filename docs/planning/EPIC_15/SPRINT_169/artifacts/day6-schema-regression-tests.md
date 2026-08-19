# Sprint 169 Day 6: Report Schema Regression Tests

## Purpose

Day 6 adds focused regression coverage for the selected canonical benchmark
freshness checker. The goal is to make the Day 5 schema normalization
enforceable with clear failures for malformed selected performance rows,
manifest mismatches, row-width drift, and unselected-row claim-boundary
regressions.

## Added Test Surface

Added:

- `tests/test_bench_canonical_freshness.py`
- `make bench-canonical-report-freshness-tests`

The test script is directly executable and also available through the Makefile
target. It generates a fresh canonical report bundle, copies generated report
artifacts to temporary directories, mutates one failure at a time, and checks
that `scripts/check_bench_canonical_freshness.py` emits stable failure
signals.

Generated report artifacts remain under ignored `build/` paths or temporary
directories and are not source-controlled.

## Positive Coverage

| Test | Coverage |
| --- | --- |
| `test_positive_local_report` | Confirms the current local generated report passes selected freshness checks. |
| `test_positive_hosted_report_keeps_unselected_rows_local` | Confirms hosted-mode selected row metadata passes while all unselected rows remain `local_only` / `local_threshold_free`. |

## Negative Coverage

| Test | Mutated field or condition | Expected protection |
| --- | --- | --- |
| `test_selected_matrix_size_is_required` | selected row `matrix_size=not_recorded` | rejects anything other than `matrix_size=n=100` |
| `test_selected_warmup_is_required` | selected row `warmup=not_recorded` | rejects anything other than `warmup=none_configured` |
| `test_selected_variance_is_required` | selected row `variance=not_recorded` | rejects anything other than `variance=not_computed_single_sample` |
| `test_manifest_selected_matrix_size_must_match` | manifest `selected_matrix_size=n=101` | rejects selected row / manifest disagreement |
| `test_row_width_mismatch_is_rejected` | truncated selected `index.tsv` row | rejects row-width drift before field validation |
| `test_unselected_rows_cannot_be_hosted_selected` | unselected row `support_tier=hosted_selected` | rejects hosted-selected metadata on unselected canonical rows |

## Schema Invariants Protected

The tests protect these selected-row invariants:

- exactly one selected `bench_refactor_csc` row;
- selected `matrix_size=n=100`;
- selected `warmup=none_configured`;
- selected `variance=not_computed_single_sample`;
- selected row and manifest agreement for `selected_matrix_size`;
- stable `index.tsv` row width;
- hosted mode selected row can be `hosted_selected` /
  `hosted_selected_threshold_free`;
- unselected rows remain `local_only` / `local_threshold_free`.

## Validation

Day 6 changed Python tests, the Makefile, planning artifacts, and earlier Day 5
script/docs changes. No `.c` or `.h` files were modified, so the full C
quality gate is not required for this day.

Validation run:

```sh
bash -n scripts/bench_canonical_report.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/check_bench_canonical_freshness.py \
  tests/test_bench_canonical_freshness.py
git diff --check
PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report-freshness-tests
```

The regression target passed all eight positive and negative cases:

```text
test_positive_local_report: passed
test_selected_matrix_size_is_required: passed
test_selected_warmup_is_required: passed
test_selected_variance_is_required: passed
test_manifest_selected_matrix_size_must_match: passed
test_row_width_mismatch_is_rejected: passed
test_unselected_rows_cannot_be_hosted_selected: passed
test_positive_hosted_report_keeps_unselected_rows_local: passed
```

## Day 6 Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Schema regressions fail with clear messages. | Complete | Negative tests mutate selected fields, manifest values, row width, and unselected boundaries. |
| Malformed selected performance reports are rejected. | Complete | Matrix size, warmup, variance, manifest mismatch, and row-width cases fail as expected. |
| Focused validation passes locally. | Complete | `make bench-canonical-report-freshness-tests` passed. |
