# Sprint 192 Day 8: Hosted Lane Implementation

## Summary

Day 8 completed the source-controlled hosted selected performance lane
implementation review by tightening CI output, workflow guard coverage, and
maintainer documentation around the selected-only artifact contract.

The lane remains bounded to Linux hosted freshness for one selected benchmark
row: `bench_refactor_csc` on `tests/data/suitesparse/nos4.mtx --repeat 1`.

## Implemented Workflow Contract

| Field | Implemented value |
| --- | --- |
| Workflow | `.github/workflows/ci.yml` |
| Job id | `hosted-performance-freshness` |
| Job name | `Linux reviewed hosted selected performance freshness` |
| Runner | `ubuntu-latest` |
| Timeout | `10` minutes |
| Report label | `sprint-168-hosted-performance` |
| Support tier | `hosted_selected` |
| Claim boundary | `hosted_selected_threshold_free` |
| Benchmark generation | `make bench-canonical-report` |
| Hosted freshness check | `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted` |
| Upload artifact | `sprint168-selected-performance-freshness` |
| Retention | `7` days |
| Missing-file behavior | `if-no-files-found: error` |

The workflow summary now prints the exact hosted upload paths so CI logs expose
the selected artifact scope during human review.

## Exact Upload Scope

Uploaded paths:

```text
build/bench-reports/canonical/bench_refactor_csc.csv
build/bench-reports/canonical/index.tsv
build/bench-reports/canonical/manifest.txt
```

Forbidden broad or unselected upload paths:

```text
build/bench-reports/**
build/bench-reports/canonical/**
build/bench-reports/canonical/bench_chol_csc.csv
build/bench-reports/canonical/bench_iterative_reuse.csv
build/bench-reports/canonical/bench_eigs_reuse.csv
```

The canonical generator may still emit unselected CSV files locally because it
owns the report bundle, but hosted publication remains selected-only.

## Guard Coverage

`tests/test_selected_comparison_workflow.py` now verifies:

- hosted selected performance job name and `timeout-minutes: 10`;
- selected hosted metadata environment variables;
- benchmark generation and hosted freshness commands;
- selected row identifier in the summary script;
- exact uploaded path summary text;
- manifest-derived workflow artifact name;
- `retention-days: 7`;
- `if-no-files-found: error`;
- required selected upload files;
- rejection of broad benchmark upload globs;
- rejection of unselected canonical benchmark CSV uploads.

Negative drift tests cover missing timeout, wrong artifact name, broad upload
patterns, unselected upload reintroduction, missing retention, and missing
required files.

## Documentation Update

`docs/maintainer_guide.md` now names the exact selected hosted artifact bundle,
its retention policy, and the fact that unselected canonical benchmark CSV
files are not uploaded as reviewed hosted performance evidence.

## Non-Claims

This lane does not establish:

- portable performance;
- timing superiority;
- benchmark regression thresholds;
- external-library parity;
- package or ABI support;
- broad platform support;
- release benchmark proof;
- state-of-the-art sparse linear algebra status.

## Validation

Commands run:

```sh
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 scripts/normalize_report_index.py --family benchmark --check-freshness
python3 -m py_compile tests/test_selected_comparison_workflow.py tests/test_bench_canonical_freshness.py tests/test_normalize_report_index.py scripts/check_bench_canonical_freshness.py scripts/normalize_report_index.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected workflow guard tests passed;
- benchmark canonical freshness regression tests passed;
- report-index normalization regression tests passed;
- selected target schema validation passed;
- benchmark report-index freshness passed with advisory local measurement rows;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 8.

## Day 9 Inputs

Day 9 can now focus on the regression-policy decision without reopening hosted
publication scope. The current implementation is explicitly threshold-free and
selected-only.
