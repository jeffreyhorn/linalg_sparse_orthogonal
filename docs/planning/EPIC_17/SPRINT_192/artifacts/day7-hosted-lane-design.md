# Sprint 192 Day 7: Hosted Lane Design

## Summary

Day 7 defined the hosted selected performance freshness lane as a bounded Linux
CI job for one methodology-bound benchmark row:
`bench_refactor_csc` on `tests/data/suitesparse/nos4.mtx --repeat 1`.

The lane is allowed to prove hosted freshness of selected threshold-free
benchmark metadata. It is not allowed to claim portable runtime performance,
algorithmic superiority, platform parity, package or ABI support, external
library parity, release benchmark status, or state-of-the-art sparse linear
algebra status.

## CI Contract

| Field | Contract |
| --- | --- |
| Workflow | `.github/workflows/ci.yml` |
| Job id | `hosted-performance-freshness` |
| Job name | `Linux reviewed hosted selected performance freshness` |
| Runner | `ubuntu-latest` |
| Timeout | `10` minutes |
| Report label | `sprint-168-hosted-performance` |
| Support tier | `hosted_selected` |
| Claim boundary | `hosted_selected_threshold_free` |
| Runner context | `github-actions-ubuntu-latest` |
| Build flags | `default_make_flags` |
| Build mode | `serial` |
| Benchmark command | `make bench-canonical-report` |
| Freshness command | `python3 scripts/check_bench_canonical_freshness.py --report-dir build/bench-reports/canonical --mode hosted` |

The job intentionally runs the canonical benchmark report generator because
that script owns the canonical index and manifest. The hosted claim remains
selected-only because the checker validates only the selected row and the
workflow uploads only the selected artifacts listed below.

## Exact Artifact Scope

The hosted upload artifact is:

```text
sprint168-selected-performance-freshness
```

The exact uploaded paths are:

```text
build/bench-reports/canonical/bench_refactor_csc.csv
build/bench-reports/canonical/index.tsv
build/bench-reports/canonical/manifest.txt
```

The workflow must not upload:

```text
build/bench-reports/**
build/bench-reports/canonical/**
build/bench-reports/canonical/bench_chol_csc.csv
build/bench-reports/canonical/bench_iterative_reuse.csv
build/bench-reports/canonical/bench_eigs_reuse.csv
```

The unselected canonical CSV files may be generated as local context by
`make bench-canonical-report`, but they are not part of the hosted reviewed
artifact scope.

## Guard Tests

`tests/test_selected_comparison_workflow.py` now verifies the selected
performance lane contract:

- job name and `timeout-minutes: 10`;
- selected hosted benchmark environment metadata;
- benchmark generation command;
- hosted freshness checker command with `--mode hosted`;
- manifest-derived workflow artifact name;
- exact required upload paths from the selected target manifest;
- fail-closed upload behavior via `if-no-files-found: error`;
- rejection of broad benchmark upload patterns;
- rejection of unselected benchmark CSV uploads;
- rejection of missing required upload files.

The guard now has drift tests for:

- missing timeout;
- wrong upload artifact name;
- broad benchmark upload paths;
- reintroduced unselected benchmark uploads;
- missing `manifest.txt` from the selected upload set.

## Review Risks

| Risk | Mitigation |
| --- | --- |
| Canonical generator emits unselected rows | The selected checker and upload scope remain selected-only. |
| Hosted timing is misread as a performance claim | Environment, manifest, and docs keep `hosted_selected_threshold_free` and non-claim wording explicit. |
| Artifact scope silently broadens | Guard tests reject broad paths and unselected benchmark CSV uploads. |
| Runner timing noise creates false product claims | The lane validates metadata freshness only; timing thresholds remain absent. |
| Later workflow edits drop selected evidence files | Guard tests derive required files from the selected target manifest. |

## Validation

Commands run:

```sh
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_bench_canonical_freshness.py
python3 tests/test_normalize_report_index.py
python3 scripts/validate_corpus_schema.py
python3 -m py_compile tests/test_selected_comparison_workflow.py
git diff --check
git diff --name-only -- '*.c' '*.h'
```

Results:

- selected workflow guard tests passed;
- benchmark canonical freshness regression tests passed;
- report-index normalization regression tests passed;
- selected target schema validation passed;
- Python syntax compilation passed;
- `git diff --check` passed;
- no `.c` or `.h` files changed, so `make format && make lint && make test`
  is not required for Day 7.

## Day 8 Inputs

Day 8 implementation should preserve the selected-only upload contract above
and keep any additional hosted behavior behind explicit guard coverage before
claim wording changes.
