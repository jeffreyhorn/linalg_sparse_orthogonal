# Sprint 168 Day 8: Freshness Check Implementation

## Purpose

Day 8 implements and validates the selected performance freshness check
designed on Day 7. The implementation is intentionally narrow: it validates
the selected `bench_refactor_csc` canonical report row and its methodology
metadata without promoting all benchmark rows into hosted evidence and without
comparing timing values.

## Implementation Summary

Added `scripts/check_bench_canonical_freshness.py`.

The checker validates:

- required report artifacts exist:
  - `bench_refactor_csc.csv`;
  - `index.tsv`;
  - `manifest.txt`;
- `index.tsv` has the required Day 6 methodology columns and consistent row
  widths;
- exactly one selected row exists with `artifact=bench_refactor_csc`;
- the selected row points at an existing `relative_path`;
- selected row identity fields match the Sprint 168 lane:
  - `command=tests/data/suitesparse/nos4.mtx --repeat 1`;
  - `fixture_or_workload=nos4.mtx`;
  - `repeat_semantics=configured_repeat_1`;
  - `report_family=benchmark`;
  - `status=measurement`;
  - `baseline=n/a`;
  - `threshold=n/a`;
- `methodology_notes` includes `not_portable_performance_claim`;
- required metadata fields are present and non-empty;
- `generated_at_utc` matches `YYYY-MM-DDTHH:MM:SSZ`;
- `support_tier` and `claim_boundary` stay within selected threshold-free
  values;
- `manifest.txt` agrees with the selected row for report label, commit,
  branch, platform, compiler, runner context, build flags, CPU model, build
  mode, thread setting, support tier, claim boundary, baseline, threshold, and
  methodology notes.

## Make Target

Added `make bench-canonical-report-freshness`.

The target:

1. regenerates the canonical benchmark report bundle through the existing
   `bench-canonical-report` target;
2. checks the selected report row with
   `python3 scripts/check_bench_canonical_freshness.py --report-dir "$(BENCH_CANONICAL_REPORT_DIR)" --mode local`;
3. prints a clear pass banner for selected threshold-free performance report
   freshness.

## Modes

| Mode | Use | Strictness |
| --- | --- | --- |
| `local` | Maintainer/local target and pre-CI dry runs. | Allows the selected row to use `local_only` or `hosted_selected` support tier, `local_threshold_free` or `hosted_selected_threshold_free` claim boundary, `not_recorded` build flags, and `unknown` CPU model; unselected rows must remain `local_only` / `local_threshold_free`. |
| `hosted` | Future hosted CI invocation after Day 9/Day 10 wiring. | Requires the selected row to use `hosted_selected`, `hosted_selected_threshold_free`, non-`local` runner context, build flags other than `not_recorded`, and a report label other than `unlabeled`; unselected rows must remain `local_only` / `local_threshold_free`. |

## Failure Behavior

The checker emits stable error prefixes and remediation text. Examples
validated locally:

```text
freshness: error: benchmark_selected_artifact_missing: artifact=bench_refactor_csc.csv ...
```

```text
freshness: error: benchmark_selected_claim_boundary: field=support_tier expected=hosted_selected observed=local_only ...
```

All freshness errors end with:

```text
run make bench-canonical-report-freshness
```

## Local Validation

Ran:

```sh
python3 -m py_compile scripts/check_bench_canonical_freshness.py
```

Result: passed.

Ran:

```sh
python3 scripts/check_bench_canonical_freshness.py --help
```

Result: passed and displayed `--report-dir` plus `--mode {local,hosted}`.

Ran:

```sh
make bench-canonical-report-freshness
```

Result: passed. The target regenerated the canonical report bundle and checked
the selected `bench_refactor_csc` row in local mode.

Ran missing-artifact failure-mode check against an empty temporary report
directory:

```text
freshness: error: benchmark_selected_artifact_missing: artifact=bench_refactor_csc.csv ...
```

Result: failed as expected with a remediation command.

Ran hosted-mode failure check against a local-default report:

```text
freshness: error: benchmark_selected_claim_boundary: field=support_tier expected=hosted_selected observed=local_only ...
```

Result: failed as expected with a remediation command.

Ran hosted-style positive dry run:

```sh
env BENCH_CANONICAL_REPORT_LABEL=sprint-168-day8-hosted-dry-run \
  SPARSE_CANONICAL_SUPPORT_TIER=hosted_selected \
  SPARSE_CANONICAL_CLAIM_BOUNDARY=hosted_selected_threshold_free \
  SPARSE_CANONICAL_RUNNER_CONTEXT=github-actions-ubuntu-dry-run \
  SPARSE_CANONICAL_BUILD_FLAGS=default_make_flags \
  SPARSE_CANONICAL_CPU_MODEL=unknown \
  SPARSE_CANONICAL_BUILD_MODE=serial \
  make bench-canonical-report

python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical \
  --mode hosted
```

Result: passed.

## Non-Claim Preservation

The checker does not:

- compare raw timing values;
- compare speedup values;
- define a regression threshold;
- compare against external libraries;
- infer broad solver correctness;
- infer package, shared-library, ABI, or platform support;
- promote `bench_chol_csc`, `bench_iterative_reuse`, or `bench_eigs_reuse` to
  hosted selected evidence.

## Quality Gate

Day 8 changed a Python script, `Makefile`, and planning artifacts. No `.c` or
`.h` files were modified, so the full C quality gate
(`make format && make lint && make test`) is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected freshness command passes locally. | Complete | `make bench-canonical-report-freshness` passed. |
| Missing/stale selected report cases fail clearly. | Complete | Missing selected artifact and hosted metadata/claim-boundary failures emitted stable `freshness: error:` diagnostics with remediation text. |
| Unselected report families are not accidentally promoted. | Complete | The checker selects only `artifact=bench_refactor_csc`; other canonical rows are ignored except for TSV schema validation. |
