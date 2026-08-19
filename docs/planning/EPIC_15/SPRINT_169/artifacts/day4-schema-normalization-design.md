# Sprint 169 Day 4: Schema Normalization Design

## Purpose

Day 4 designs the selected performance report schema changes needed to make
the Day 3 statistical policy machine-readable. The design keeps the selected
`bench_refactor_csc` publication row narrow, preserves unselected canonical
rows as local/advisory, and prepares Day 5 implementation work.

## Current Selected Row

The current generated selected row has this shape in `index.tsv`:

| Field | Current value |
| --- | --- |
| `artifact` | `bench_refactor_csc` |
| `relative_path` | `bench_refactor_csc.csv` |
| `command` | `tests/data/suitesparse/nos4.mtx --repeat 1` |
| `fixture_or_workload` | `nos4.mtx` |
| `matrix_size` | `not_recorded` |
| `repeat_semantics` | `configured_repeat_1` |
| `warmup` | `not_recorded` |
| `variance` | `not_recorded` |
| `baseline` | `n/a` |
| `threshold` | `n/a` |

Day 3 decided that `repeat_semantics`, `baseline`, and `threshold` should
remain stable, while `warmup` and `variance` should move from ambiguous
`not_recorded` values to explicit policy values.

## Schema Normalization Decision

| Field | Decision | Implementation owner |
| --- | --- | --- |
| `repeat_semantics` | Keep `configured_repeat_1`. | `scripts/bench_canonical_report.sh`; checker selected-value expectation. |
| `warmup` | Normalize selected row to `none_configured`. | generator default for canonical rows; checker selected-value expectation. |
| `variance` | Normalize selected row to `not_computed_single_sample`. | generator default for canonical rows; checker selected-value expectation. |
| `baseline` | Keep `n/a`. | generator and checker. |
| `threshold` | Keep `n/a`. | generator and checker. |
| `matrix_size` | Normalize selected `nos4.mtx` row to `n=100`. | generator row metadata; checker selected-value expectation. |
| sample count | Do not add a new column in Sprint 169 Day 5. | Express single-sample policy through `repeat_semantics` and `variance`. |
| nonzero count | Do not add a new column in Sprint 169 Day 5. | Leave as future optional schema work if report consumers need it. |

This keeps the schema stable at 29 columns while replacing weak values with
more precise machine-readable values.

## Matrix-Size Decision

The selected fixture has two relevant facts:

| Source | Observed value | Interpretation |
| --- | --- | --- |
| `tests/data/suitesparse/nos4.mtx` Matrix Market size line | `100 100 347` | Stored symmetric-coordinate fixture dimensions and stored-entry count. |
| `bench_refactor_csc.csv` selected row | `n=100`, `nnz=594` | Benchmark-side expanded problem size and nonzero count. |

Sprint 169 should set:

```text
matrix_size=n=100
```

Rationale:

- `n=100` is already emitted by the selected benchmark row;
- it is stable for the selected fixture and command;
- it avoids adding fixture parsing to the shell report generator;
- it avoids ambiguity between Matrix Market stored entries (`347`) and
  expanded symmetric benchmark nonzeros (`594`);
- it fits the existing `matrix_size` field without changing the index header.

If future report consumers need nonzero-count semantics, add a separate
`nonzero_count` or `workload_size` field rather than overloading
`matrix_size`.

## Stable Formatting Rules

| Field | Format rule |
| --- | --- |
| `matrix_size` | `n=<integer>` for selected square benchmark rows; use `not_recorded` only when the benchmark row does not expose stable dimensions. |
| `repeat_semantics` | `configured_repeat_<integer>` when the selected command has an explicit `--repeat <integer>`. |
| `warmup` | `none_configured` when no separate warmup phase is configured; future counted warmups should use a stable token such as `configured_warmup_<integer>`. |
| `variance` | `not_computed_single_sample` for one selected report observation; future computed variance should name the statistic and sample source. |
| `baseline` | `n/a` for threshold-free publication rows. |
| `threshold` | `n/a` for threshold-free publication rows. |

All values must remain single-line, tab-free, deterministic strings suitable
for TSV output and direct textual diffs.

## Manifest Agreement Requirements

The selected row and `manifest.txt` should agree on:

- `report_label`;
- `git_commit`;
- `git_branch`;
- `platform`;
- `compiler`;
- `runner_context`;
- `build_flags`;
- `cpu_model`;
- `build_mode`;
- `omp_num_threads`;
- `support_tier`;
- `claim_boundary`;
- `baseline`;
- `threshold`;
- `warmup`;
- `variance`;
- `matrix_size`;
- `methodology_notes`.

Day 5 should extend checker manifest agreement to include `warmup`,
`variance`, and `matrix_size`. These fields are currently emitted in the
manifest, but the checker does not compare them against the selected row.

## Selected And Unselected Row Invariants

Selected row invariants:

- exactly one row with `artifact=bench_refactor_csc`;
- `relative_path=bench_refactor_csc.csv`;
- `command=tests/data/suitesparse/nos4.mtx --repeat 1`;
- `fixture_or_workload=nos4.mtx`;
- `matrix_size=n=100`;
- `repeat_semantics=configured_repeat_1`;
- `warmup=none_configured`;
- `variance=not_computed_single_sample`;
- `baseline=n/a`;
- `threshold=n/a`;
- local mode allows local or hosted-style dry-run selected boundaries;
- hosted mode requires `hosted_selected` /
  `hosted_selected_threshold_free`.

Unselected row invariants:

- support tier remains `local_only`;
- claim boundary remains `local_threshold_free`;
- unselected rows are not included in the selected hosted evidence claim;
- unselected rows may receive the normalized warmup and variance policy if
  the generator applies it globally, but they must remain local/advisory;
- freshness checks should continue to reject hosted-selected metadata on any
  unselected row.

## Implementation Plan For Day 5

1. Change `warmup` in `scripts/bench_canonical_report.sh` from
   `not_recorded` to `none_configured`.
2. Change `variance` in `scripts/bench_canonical_report.sh` from
   `not_recorded` to `not_computed_single_sample`.
3. Set selected-row `matrix_size` to `n=100` while preserving
   `not_recorded` for rows without stable dimensions.
4. Update `scripts/check_bench_canonical_freshness.py` selected-value
   expectations for `matrix_size`, `warmup`, and `variance`.
5. Extend manifest agreement checks to include `matrix_size`, `warmup`, and
   `variance`.
6. Update README, benchmark docs, and maintainer docs to replace stale
   `not_recorded` warmup/variance wording for canonical selected performance
   rows.
7. Run focused script, Python, local freshness, hosted-mode freshness, and
   `git diff --check` validation.

## Day 4 Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Schema changes are planned before implementation. | Complete | Field decisions and Day 5 implementation steps are listed. |
| Selected and unselected row behavior is explicitly separated. | Complete | Selected and unselected invariants are defined separately. |
| Generated fields remain machine-readable and diff-friendly. | Complete | Stable formatting rules keep values deterministic and tab-free. |

## Validation

Day 4 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Run after writing this artifact:

```sh
git diff --check
```
