# Sprint 168 Day 6: Report Metadata Implementation

## Purpose

Day 6 implements the methodology metadata contract designed on Day 5 for the
selected canonical performance report path. The implementation keeps the
existing canonical report behavior and output location while adding hosted-lane
metadata hooks needed for later freshness publication.

The selected lane remains threshold-free. The new metadata fields describe
where and how the report was generated; they do not create a timing threshold,
portable performance guarantee, backend superiority claim, external comparison
claim, or state-of-the-art sparse linear algebra claim.

## Implementation Summary

Updated `scripts/bench_canonical_report.sh` to:

- preserve the existing five-argument interface used by
  `make bench-canonical-report`;
- preserve the existing four generated benchmark CSV files under
  `build/bench-reports/canonical/`;
- keep generated timing CSV row formats unchanged;
- allow hosted CI to override `support_tier`, `claim_boundary`, and
  `methodology_notes`;
- add `runner_context`, `build_flags`, and `cpu_model` to both `index.tsv`
  and `manifest.txt`;
- reject tabs and newlines in all newly configurable metadata fields before
  emitting TSV or manifest content.

## Environment Hooks

| Variable | Purpose | Local default |
| --- | --- | --- |
| `SPARSE_CANONICAL_SUPPORT_TIER` | Classify the report support tier. | `local_only` |
| `SPARSE_CANONICAL_CLAIM_BOUNDARY` | Record the report claim boundary. | `local_threshold_free` |
| `SPARSE_CANONICAL_RUNNER_CONTEXT` | Identify local or hosted runner context. | `local` |
| `SPARSE_CANONICAL_BUILD_FLAGS` | Record build flags or CI build policy. | `not_recorded` |
| `SPARSE_CANONICAL_CPU_MODEL` | Record CPU model or hosted runner CPU note. | `unknown` |
| `SPARSE_CANONICAL_METHODOLOGY_NOTES` | Override methodology-note tokens. | `threshold_free_local_measurement;not_portable_performance_claim` |

## Output Schema Change

`index.tsv` now inserts three fields after `compiler`:

- `runner_context`;
- `build_flags`;
- `cpu_model`.

The resulting `index.tsv` header and emitted rows have 29 tab-separated fields.
The selected `bench_refactor_csc` row still records `nos4.mtx`,
`configured_repeat_1`, `baseline=n/a`, `threshold=n/a`, and
`methodology_notes=threshold_free_local_measurement;not_portable_performance_claim`.

## Compatibility Notes

- Existing local invocations can continue to run `make bench-canonical-report`
  without setting any new variables.
- Existing report files are still written below the ignored
  `build/bench-reports/canonical/` path.
- Existing CSV timing files keep their current benchmark-owned schema.
- Local defaults remain conservative: `local_only`,
  `local_threshold_free`, `not_recorded`, and `unknown`.
- Hosted CI can supply stronger selected-lane metadata without adding a second
  benchmark runner.

## Local Validation

Ran:

```sh
bash -n scripts/bench_canonical_report.sh
```

Result: passed.

Ran:

```sh
env BENCH_CANONICAL_REPORT_LABEL=sprint-168-day6-local \
  SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
  SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
  SPARSE_CANONICAL_RUNNER_CONTEXT=local-day6-dry-run \
  SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
  SPARSE_CANONICAL_CPU_MODEL=unknown \
  SPARSE_CANONICAL_BUILD_MODE=serial \
  make bench-canonical-report
```

Result: passed and wrote the canonical CSV bundle, `index.tsv`, and
`manifest.txt`.

Checked generated `index.tsv` field counts:

```text
1:29
2:29
```

Checked generated metadata values in `index.tsv` and `manifest.txt`:

- `report_label=sprint-168-day6-local`;
- `runner_context=local-day6-dry-run`;
- `build_flags=default_make_flags`;
- `cpu_model=unknown`;
- `build_mode=serial`;
- `support_tier=hosted_selected`;
- `claim_boundary=hosted_selected_threshold_free`.

Generated report output remains ignored under `build/` and is not intended to
be staged.

## Quality Gate

Day 6 changed a shell script and planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate (`make format && make lint && make test`)
is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected report emits required metadata locally. | Complete | Local hosted-style `make bench-canonical-report` emitted runner, flags, CPU, support tier, and claim-boundary fields. |
| Existing report commands still run for local workflows. | Complete | The existing Make target and script argument contract were preserved. |
| Generated build/report artifacts are not staged unintentionally. | Complete | Report output remains under ignored `build/bench-reports/canonical/`. |
| CSV timing rows remain benchmark-owned. | Complete | Day 6 changed `index.tsv` and `manifest.txt` metadata only; CSV schemas are unchanged. |
