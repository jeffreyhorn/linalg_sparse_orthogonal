# Sprint 169 Day 12: Integrated Local Validation

## Purpose

Day 12 runs the integrated local validation sweep for the selected performance
methodology work completed so far. The sweep covers changed shell and Python
scripts, selected local freshness, hosted-mode local metadata validation, S6
sentinel behavior, normalized report-index integration, generated-output
handling, and claim-safe documentation.

## Validation Summary

| Check | Result | Notes |
| --- | --- | --- |
| Shell syntax | Passed | `bench_canonical_report.sh` and `performance_sentinels.sh` parsed cleanly. |
| Python compile | Passed | Freshness checker, normalizer, and focused tests compiled cleanly. |
| Selected freshness regression tests | Passed | All eight positive/negative `test_bench_canonical_freshness.py` cases passed. |
| Local selected freshness | Passed | `make bench-canonical-report-freshness` passed in local mode. |
| Hosted-style local metadata validation | Passed | Hosted mode passed with hosted-selected support/claim metadata and non-local runner context. |
| Sentinel validation | Passed | `make performance-sentinels` passed and emitted S5/S6 hard gates plus S2/S3 context. |
| Normalized report-index tests | Passed | `tests/test_normalize_report_index.py` passed. |
| Generated benchmark/sentinel normalized view | Passed | `normalize_report_index.py --family benchmark --family sentinel --check-freshness` exited 0. |
| Claim scan | Passed | Scan found scoped non-claim wording and selected/local evidence caveats only. |
| Generated-output policy | Passed | Generated report output remains under ignored `build/`. |
| Whitespace hygiene | Passed | `git diff --check` passed. |

## Commands Run

```sh
bash -n scripts/bench_canonical_report.sh scripts/performance_sentinels.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/check_bench_canonical_freshness.py \
  scripts/normalize_report_index.py \
  tests/test_bench_canonical_freshness.py \
  tests/test_normalize_report_index.py
PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report-freshness-tests
PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report-freshness
BENCH_CANONICAL_REPORT_LABEL=sprint-169-hosted-style-local \
  SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
  SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
  SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-ubuntu-latest \
  SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
  SPARSE_CANONICAL_BUILD_MODE=serial \
  SPARSE_CANONICAL_CPU_MODEL=local-hosted-style \
  PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical --mode hosted
PYTHONDONTWRITEBYTECODE=1 make performance-sentinels
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_normalize_report_index.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/normalize_report_index.py \
  --family benchmark --family sentinel \
  --output build/report-index/normalized-index.tsv
PYTHONDONTWRITEBYTECODE=1 python3 scripts/normalize_report_index.py \
  --family benchmark --family sentinel --check-freshness
rg -n "portable performance|portable speed|performance guarantee|state-of-the-art performance|hosted benchmark result|platform parity|OpenMP speedup|backend superiority|external-library parity|release benchmark proof|runtime-loader" \
  README.md benchmarks/README.md docs/maintainer_guide.md \
  docs/planning/EPIC_15/SPRINT_169 -g '*.md'
git status --ignored --short build/bench-reports/canonical \
  build/bench-reports/sentinels build/report-index
PYTHONDONTWRITEBYTECODE=1 make bench-canonical-report-freshness
git diff --check
```

The hosted-style check was rerun sequentially after an initial parallel
validation attempt conflicted with the local freshness target because both
commands write `build/bench-reports/canonical`. The sequential hosted-style
check passed.

## Selected Freshness Evidence

`make bench-canonical-report-freshness` passed in local mode:

```text
bench-canonical-freshness: passed (mode=local; artifact=bench_refactor_csc; report_dir=build/bench-reports/canonical)
bench-canonical-report-freshness: passed (selected threshold-free performance report freshness)
```

The focused regression test target passed all eight cases:

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

## Hosted-Mode Local Evidence

The hosted-style local metadata validation passed with:

- `BENCH_CANONICAL_REPORT_LABEL=sprint-169-hosted-style-local`;
- `SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected`;
- `SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free`;
- `SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-ubuntu-latest`;
- `SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags`;
- `SPARSE_CANONICAL_BUILD_MODE=serial`;
- `SPARSE_CANONICAL_CPU_MODEL=local-hosted-style`.

Result:

```text
bench-canonical-freshness: passed (mode=hosted; artifact=bench_refactor_csc; report_dir=build/bench-reports/canonical)
```

This is a local hosted-style metadata check, not hosted CI proof.

## Sentinel And Report-Index Evidence

`make performance-sentinels` passed and wrote:

```text
sentinels.tsv
manifest.txt
wall_check.txt
bench_refactor_csc_nos4.csv
bench_chol_csc_nos4.csv
bench_refactor_csc_kkt.csv
```

`tests/test_normalize_report_index.py` passed.

The generated benchmark/sentinel normalized view wrote 27 rows and
`--check-freshness` exited 0. The output preserved benchmark rows as advisory,
S2/S3 rows as advisory, and S5/S6 hard-gate rows as generated-present
unchecked warnings rather than portable performance claims.

## Generated-Output Handling

Generated output remains ignored:

```text
!! build/
```

No generated report artifact was staged or promoted to source-controlled
evidence. The final local freshness run restored the canonical generated
bundle to local metadata after the hosted-style validation.

## Day 12 Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected methodology policy passes local checks. | Complete | Local freshness, hosted-style metadata validation, and freshness regression tests passed. |
| Generated report output remains ignored unless intentionally published. | Complete | `git status --ignored --short build/...` reports ignored `build/` output only. |
| All required focused checks pass. | Complete | Syntax, compile, freshness, sentinel, normalizer, claim scan, and whitespace checks passed. |
