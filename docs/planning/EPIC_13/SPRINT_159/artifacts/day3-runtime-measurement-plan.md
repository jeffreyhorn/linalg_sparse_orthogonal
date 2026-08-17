# Day 3 Runtime Measurement Plan

## Scope

Day 3 defines how Sprint 159 will measure selected oracle and comparison
freshness candidates before adding hosted CI work. This artifact is a plan,
not runtime evidence. Day 4 owns execution, measured timings, output sizes,
and promotion/demotion decisions.

## Selected Measurement Targets

| Target ID | Candidate | Primary command | Measurement objective |
| --- | --- | --- | --- |
| S159-H01 | QR `oracle/solver_backed` rows | `make report-index-oracle-freshness` | Determine whether selected QR oracle rows can run in hosted CI with stable runtime and bounded output. |
| S159-H02 | Partial-SVD `oracle/solver_backed` rows | `make report-index-oracle-freshness` | Determine whether selected partial-SVD oracle rows can share the hosted oracle lane without excessive runtime. |
| S159-H03 | `comparison/qr_minnorm` rows | `make report-index-comparison-freshness` | Determine whether the QR minimum-norm comparison lane can run as reviewed hosted evidence. |
| S159-S01 | `oracle/generated_reference` rows | `make report-index-oracle-freshness` | Determine whether generated-reference rows should be included in hosted artifacts as supplemental context. |

The oracle Make target measures S159-H01, S159-H02, and S159-S01 together
because the maintained command generates the selected combined oracle family.
Day 4 should not split that command unless measured runtime proves a need for
a narrower hosted command.

## Measurement Matrix

| Measurement | Command | Cleanup/precondition | Captures |
| --- | --- | --- | --- |
| Oracle cold gate | `make clean && make report-index-oracle-freshness` | Start from clean build and regenerated generated outputs. | Full hosted-like cost: build, oracle generation, normalizer freshness check. |
| Oracle warm gate | `make report-index-oracle-freshness` | Run after cold gate without `make clean`. | Steady-state regeneration plus freshness-check cost. |
| Oracle generator-only | `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd` | Static library exists. | Oracle generator cost and produced artifact size/count. |
| Oracle normalizer-only | `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness` | Oracle generated outputs exist from the same commit. | Strict selected oracle freshness-check cost and diagnostics. |
| Comparison cold gate | `make clean && make report-index-comparison-freshness` | Start from clean build and regenerated generated outputs. | Full hosted-like cost: build, comparison generation, normalizer freshness check. |
| Comparison warm gate | `make report-index-comparison-freshness` | Run after cold gate without `make clean`. | Steady-state comparison regeneration plus freshness-check cost. |
| Comparison generator-only | `python3 scripts/run_external_comparison.py --target qr-minnorm` | Static library exists. | Comparison generator cost, dependency behavior, and artifact size/count. |
| Comparison normalizer-only | `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` | Comparison generated outputs exist from the same commit. | Strict selected comparison freshness-check cost and diagnostics. |

## Timing Protocol

Day 4 should capture at least:

- one cold run for each selected Make gate;
- two warm runs for each selected Make gate;
- one generator-only run for each selected generator;
- one normalizer-only run for each selected strict freshness check.

Use `/usr/bin/time -p` where available so output has stable `real`, `user`,
and `sys` fields. If `/usr/bin/time` is unavailable on a platform, use the
shell `time` output and record the shell/platform in the evidence artifact.

Recommended wrapper shape:

```sh
/usr/bin/time -p make report-index-oracle-freshness
/usr/bin/time -p make report-index-comparison-freshness
```

Do not use parallel Make for these measurements. The hosted lane should remain
serial until artifact paths and generator cleanup behavior are proven safe.

## Generated Output Inventory

### Oracle Gate Outputs

Expected paths:

- `build/corpus/oracle/corpus.oracle.tsv`
- `build/corpus-reports/index.tsv`
- `build/corpus-reports/skips.tsv`
- `build/corpus-reports/manifest.txt`

Expected selected row counts from current maintainer guidance:

- `3` generated-reference rows
- `23` `solver_family=qr` rows
- `26` `solver_family=partial_svd` rows
- `52` generated oracle rows total

Day 4 should record:

- file existence;
- line count for each TSV;
- byte size for each output file;
- selected solver-family counts;
- selected fixture-key coverage;
- generator command recorded in the manifest;
- source commit and branch recorded in generated rows or manifest;
- diagnostics emitted by the strict normalizer check.

### Comparison Gate Outputs

Expected paths:

- `build/comparison/qr_minnorm/project_observations.tsv`
- `build/comparison/qr_minnorm/baseline_observations.tsv`
- `build/comparison/qr_minnorm/dependency_status.tsv`
- `build/comparison/qr_minnorm/study.tsv`
- `build/comparison/qr_minnorm/summary.md`
- `build/comparison/qr_minnorm/manifest.tsv`

Expected selected rows from `scripts/normalize_report_index.py`:

- `comparison_qr_underdetermined_minnorm_2x4_project_status_v1`
- `comparison_qr_underdetermined_minnorm_2x4_baseline_status_v1`
- `comparison_qr_underdetermined_minnorm_2x4_residual_norm_v1`
- `comparison_qr_underdetermined_minnorm_2x4_solution_norm_v1`
- `comparison_qr_underdetermined_minnorm_2x4_solution_values_v1`
- `comparison_qr_underdetermined_minnorm_2x4_project_vs_baseline_max_abs_delta_v1`

Day 4 should record:

- file existence;
- line count for each TSV and summary file;
- byte size for each output file;
- dependency status and whether any dependency was skipped;
- selected row count and selected row status;
- source commit, branch, worktree state, platform, compiler, and baseline
  command from the manifest/study rows;
- diagnostics emitted by the strict normalizer check.

## Hosted Timeout Draft

Day 4 should compute final timeout recommendations from measured local
runtime. Until measurements exist, use these draft rules:

| Lane | Draft timeout rule | Promotion threshold |
| --- | --- | --- |
| Oracle freshness | `max(10 minutes, 4x cold local real time)` | Promote only if cold local run is stable and projected hosted runtime remains comfortably below the timeout. |
| Comparison freshness | `max(10 minutes, 4x cold local real time)` | Promote only if dependency behavior is deterministic and generated comparison rows pass without optional skip ambiguity. |
| Combined hosted job | `max(15 minutes, 4x combined cold local real time)` | Use only if both selected gates are stable and artifact output remains small. |

If either selected gate requires more than 10 minutes locally, Day 4 should
pause CI implementation and decide whether to demote the family, split the
lane, or ask for review.

## Rerun Policy Draft

- A failed hosted selected row should be rerun only after inspecting artifact
  diagnostics, not by blindly retrying.
- A runner-service or dependency-fetch outage can be rerun as infrastructure
  failure if the generated rows were not produced.
- A stale, missing, failing, partial, missing-solver-family,
  missing-fixture-key, row-set-mismatch, or non-pass selected row is a product
  failure, not a retry-only failure.
- Optional dependency skips in comparison output are not pass evidence unless
  Day 9/10 explicitly define selected skip semantics.
- Day 4 should record whether comparison baseline execution has any external
  dependency risk on hosted runners.

## Pass, Skip, Stale, And Fail Timing Criteria

| State | Criteria before CI work |
| --- | --- |
| Pass | Command exits `0`, selected row counts match, selected rows are fresh for the current commit, output size is bounded, and summary/artifacts are deterministic enough for hosted review. |
| Skip | Skip is allowed only for documented optional dependency state; selected hosted proof cannot treat a skip as pass unless later semantics explicitly select that behavior. |
| Stale | Any source-commit mismatch, stale generated row, or stale manifest for selected rows blocks CI promotion. |
| Fail | Any non-pass selected oracle/comparison row, missing selected fixture, missing solver family, row-count mismatch, failed dependency, or malformed output blocks CI promotion. |
| Too slow | Runtime exceeds the Day 4 threshold or varies enough to threaten hosted flakiness. |
| Too noisy | Artifacts are too large, unstable, or hard to interpret from hosted logs. |

## Artifact Retention Planning Inputs

Day 4 should recommend:

- whether to upload raw generated TSVs;
- whether to upload summary-only artifacts for normal passing runs;
- whether failure runs should always upload raw generated outputs;
- artifact retention days;
- artifact names that distinguish selected hosted oracle and comparison rows
  from advisory/local-only report families.

Initial artifact naming proposal:

| Artifact | Contents |
| --- | --- |
| `sprint159-oracle-freshness` | Oracle TSV, corpus report index, skips, manifest, and selected-row summary. |
| `sprint159-comparison-qr-minnorm` | Comparison TSVs, dependency status, study, summary, manifest, and selected-row summary. |
| `sprint159-report-index-diagnostics` | Normalizer diagnostics and command log for selected hosted rows only. |

## Day 4 Execution Checklist

1. Record branch, commit, worktree state, platform, compiler, and date.
2. Run the cold and warm measurement matrix.
3. Capture `real`, `user`, and `sys` time for each command.
4. Capture output file size and line counts.
5. Capture selected row counts and selected row statuses.
6. Capture normalizer diagnostics.
7. Decide whether each selected candidate remains reviewed-hosted candidate,
   becomes supplemental-hosted, stays advisory-local, or is deferred.
8. Stop before CI edits if runtime, dependency, artifact, or semantics evidence
   is unclear.

## Completion Check

- Selected rows have measurable local commands.
- Cold, warm, generator-only, and normalizer-only measurements are defined.
- Hosted timeout and rerun expectations are stated before workflow edits.
- Artifact-retention planning has concrete output names and paths.
- Day 4 has an executable runtime evidence checklist.
