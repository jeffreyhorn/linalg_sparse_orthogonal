# Sprint 168 Day 7: Freshness Check Design

## Purpose

Day 7 designs the strict freshness check for the selected hosted performance
report. The check is intentionally narrow: it proves that the selected
`bench_refactor_csc` canonical report bundle was generated with required
methodology metadata and claim boundaries. It does not compare timing values or
promote generic benchmark rows into broad performance evidence.

## Existing Freshness Patterns Reviewed

| Existing owner | Pattern | Sprint 168 decision |
| --- | --- | --- |
| `make report-index-oracle-freshness` | Regenerate selected oracle output, then run `normalize_report_index.py --family oracle --require-generated oracle --check-freshness`. | Reuse the regenerate-then-check Make target shape and clear banner messages. |
| `make report-index-comparison-freshness` | Regenerate selected comparison studies, then require expected generated rows and statuses. | Reuse selected-row strictness and actionable remediation messages. |
| `scripts/normalize_report_index.py` | Normalizes report families and applies strict freshness only to selected oracle/comparison families and guardrails. | Keep generic benchmark rows advisory for now; add a focused selected-performance check instead of changing every benchmark row. |
| `tests/corpus/manifests/report_families.tsv` | Describes canonical benchmark rows as local/advisory generated evidence. | Preserve this broad contract until hosted CI owns the selected performance lane. |
| `scripts/bench_canonical_report.sh` | Generates the selected report bundle and Day 6 methodology metadata. | Treat `index.tsv` and `manifest.txt` as the primary freshness inputs. |

## Selected Freshness Scope

| Field | Required value |
| --- | --- |
| Report directory | `build/bench-reports/canonical` by default, with a script argument override for tests and CI diagnostics. |
| Required artifacts | `bench_refactor_csc.csv`, `index.tsv`, and `manifest.txt`. |
| Selected row | Exactly one `index.tsv` row with `artifact=bench_refactor_csc`. |
| Command | `tests/data/suitesparse/nos4.mtx --repeat 1`. |
| Fixture | `nos4.mtx`. |
| Repeat semantics | `configured_repeat_1`. |
| Report family | `benchmark`. |
| Row status | `measurement`. |
| Baseline | `n/a`. |
| Threshold | `n/a`. |
| Methodology note token | Must include `not_portable_performance_claim`. |

## Required Metadata

The selected row must contain non-empty values for:

- `surface`;
- `category`;
- `report_label`;
- `generated_at_utc`;
- `git_commit`;
- `git_branch`;
- `platform`;
- `compiler`;
- `runner_context`;
- `build_flags`;
- `cpu_model`;
- `build_mode`;
- `omp_num_threads`;
- `artifact`;
- `relative_path`;
- `command`;
- `report_family`;
- `status`;
- `support_tier`;
- `claim_boundary`;
- `fixture_or_workload`;
- `repeat_semantics`;
- `warmup`;
- `variance`;
- `baseline`;
- `threshold`;
- `backend_context`;
- `methodology_notes`.

The check should validate that `generated_at_utc` matches the UTC timestamp
shape `YYYY-MM-DDTHH:MM:SSZ`. It should not require a specific timestamp value.

## Local Versus Hosted Invocation

### Local Target

Day 8 should add a local target shaped like:

```make
.PHONY: bench-canonical-report-freshness
bench-canonical-report-freshness: bench-canonical-report
	@echo "bench-canonical-report-freshness: checking selected canonical performance report"
	@python3 scripts/check_bench_canonical_freshness.py --report-dir "$(BENCH_CANONICAL_REPORT_DIR)" --mode local
	@echo "bench-canonical-report-freshness: passed (selected threshold-free performance report freshness)"
```

Local mode should accept:

- `support_tier=local_only` or `hosted_selected`;
- `claim_boundary=local_threshold_free` or
  `hosted_selected_threshold_free`;
- `runner_context=local` or a non-empty local dry-run label;
- `build_flags=not_recorded` or a non-empty configured value;
- `cpu_model=unknown` or a non-empty configured value.

This keeps existing local report generation usable while allowing the Day 6
hosted-style dry run to pass the same checker.

### Hosted Target

Day 9/Day 10 should wire a hosted invocation that generates the report with
explicit metadata and then runs:

```sh
python3 scripts/check_bench_canonical_freshness.py \
  --report-dir build/bench-reports/canonical \
  --mode hosted
```

Hosted mode should require:

- `support_tier=hosted_selected`;
- `claim_boundary=hosted_selected_threshold_free`;
- `runner_context` is not `local`;
- `build_flags` is not `not_recorded`;
- `report_label` is not `unlabeled`.

Hosted mode may allow `cpu_model=unknown` because GitHub-hosted runner CPU
assignment can vary, but the value must still be present.

## Strict Checks

The Day 8 checker should:

1. Fail if any required artifact is missing.
2. Fail if `index.tsv` is missing required columns or has malformed row widths.
3. Fail if the selected `bench_refactor_csc` row is missing or duplicated.
4. Fail if the selected row points at a missing `relative_path`.
5. Fail if command, fixture, repeat semantics, report family, row status,
   baseline, threshold, or methodology notes do not match the selected scope.
6. Fail if required metadata fields are blank, `unknown` where disallowed, or
   `not_recorded` where hosted mode disallows it.
7. Fail if `support_tier` or `claim_boundary` is broader than the selected
   threshold-free values.
8. Fail if `manifest.txt` disagrees with selected row fields for
   `report_label`, `git_commit`, `git_branch`, `platform`, `compiler`,
   `runner_context`, `build_flags`, `cpu_model`, `build_mode`,
   `omp_num_threads`, `support_tier`, `claim_boundary`, `baseline`,
   `threshold`, and `methodology_notes`.

## Non-Checks

The freshness check must not:

- compare raw elapsed times, speedups, or residual timing columns;
- require timing improvement versus a baseline;
- derive a performance regression threshold;
- compare this project against SuiteSparse, Eigen, SciPy, NumPy, LAPACK, or
  another external library;
- infer broad solver correctness from benchmark rows;
- infer broad platform, package, shared-library, or ABI support;
- mark unselected canonical rows as hosted evidence.

`bench_chol_csc`, `bench_iterative_reuse`, and `bench_eigs_reuse` may remain in
the canonical bundle as context, but the strict selected-performance freshness
check should ignore them except for TSV row-width validation.

## Failure Message Requirements

Failure output should be short, machine-searchable, and actionable:

| Failure | Required message content |
| --- | --- |
| Missing report directory | `freshness: error: benchmark_selected_report_dir_missing`, expected path, remediation command. |
| Missing artifact | `freshness: error: benchmark_selected_artifact_missing`, artifact name, expected path, remediation command. |
| Bad TSV schema | `freshness: error: benchmark_selected_schema`, missing columns or row-width detail. |
| Missing selected row | `freshness: error: benchmark_selected_row_missing`, selected artifact name, remediation command. |
| Duplicate selected row | `freshness: error: benchmark_selected_row_duplicate`, observed count. |
| Bad selected value | `freshness: error: benchmark_selected_value`, field name, expected value, observed value. |
| Missing metadata | `freshness: error: benchmark_selected_metadata_missing`, field name. |
| Over-broad support or claim | `freshness: error: benchmark_selected_claim_boundary`, offending field and value. |
| Manifest mismatch | `freshness: error: benchmark_selected_manifest_mismatch`, field name, row value, manifest value. |

Every error should end with:

```text
run make bench-canonical-report-freshness
```

or the equivalent hosted command once the hosted lane exists.

## Report-Index Behavior

For Day 8, the strict selected freshness check should be independent of
`normalize_report_index.py`. The normalized report index can continue to treat
the benchmark family as advisory, matching `report_families.tsv`.

Later Sprint 168 days may add a normalized-index row or CI docs pointer for the
selected hosted performance lane, but that should happen only after the focused
freshness checker and hosted CI invocation exist.

## Day 8 Implementation Shape

Recommended Day 8 files and commands:

- add `scripts/check_bench_canonical_freshness.py`;
- add `make bench-canonical-report-freshness`;
- run `python3 scripts/check_bench_canonical_freshness.py --help`;
- run `make bench-canonical-report-freshness`;
- run at least one focused failure-mode self-check using a temporary copied
  report directory or synthetic minimal report fixture;
- run `git diff --check`.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected report freshness has objective pass/fail rules. | Complete | Strict checks define artifacts, selected row identity, metadata, manifest agreement, and claim-boundary constraints. |
| Failure output will be actionable in CI. | Complete | Failure-message table defines stable prefixes, observed/expected detail, and remediation command text. |
| Freshness rules do not convert timing into a superiority gate. | Complete | Non-checks explicitly exclude raw timing comparisons, thresholds, external parity, and broad performance claims. |
