# Day 4 Runtime Budget Evidence

## Scope

Day 4 executes the Sprint 159 runtime measurement plan for selected oracle and
comparison freshness candidates. The goal is to decide whether selected rows
fit a realistic hosted CI budget before any workflow edits are made.

## Measurement Context

| Field | Value |
| --- | --- |
| Branch | `sprint-159` |
| Commit | `b53810ba514b030a0cbe6153cd92e9760a51b5b3` |
| Platform | `darwin-x86_64` |
| Compiler from generated manifests | `Apple clang version 11.0.0 (clang-1100.0.33.17)` |
| Timing tool | `/usr/bin/time -p` |
| Worktree state during comparison manifest generation | `dirty` because Sprint 159 planning artifacts are untracked/in progress |
| Parallelism | serial Make and serial script execution |

## Runtime Measurements

| Measurement | Command | Result | real | user | sys | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Oracle cold gate | `make clean`, then `make report-index-oracle-freshness` | pass | 26.22s | 11.98s | 4.11s | Includes full static-library rebuild, oracle generation, and strict freshness check. |
| Oracle warm gate 1 | `make report-index-oracle-freshness` | pass | 6.65s | 1.30s | 0.99s | Regenerates selected oracle output and runs strict freshness check. |
| Oracle warm gate 2 | `make report-index-oracle-freshness` | pass | 5.67s | 1.41s | 1.08s | Second warm sample remained stable. |
| Oracle generator-only | `python3 scripts/run_corpus_oracle.py --include-solver-qr --include-partial-svd` | pass | 4.56s | 1.02s | 0.77s | Static library already existed. |
| Oracle normalizer-only | `python3 scripts/normalize_report_index.py --family oracle --require-generated oracle --check-freshness` | pass | 0.26s | 0.14s | 0.06s | Strict selected oracle freshness check. |
| Comparison cold gate | `make clean`, then `make report-index-comparison-freshness` | pass | 21.94s | 14.17s | 4.27s | Includes full static-library rebuild, comparison generation, and strict freshness check. |
| Comparison warm gate 1 | `make report-index-comparison-freshness` | pass | 1.94s | 0.67s | 0.50s | Regenerates selected comparison output and runs strict freshness check. |
| Comparison warm gate 2 | `make report-index-comparison-freshness` | pass | 1.77s | 0.63s | 0.45s | Second warm sample remained stable. |
| Comparison generator-only | `python3 scripts/run_external_comparison.py --target qr-minnorm` | pass | 1.03s | 0.32s | 0.22s | Static library already existed. |
| Comparison normalizer-only | `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness` | pass | 0.26s | 0.14s | 0.06s | Strict selected comparison freshness check. |

## Row Counts And Status

### Oracle Rows

| Field | Count |
| --- | ---: |
| Total generated oracle rows | 52 |
| `solver_family=qr` | 23 |
| `solver_family=partial_svd` | 26 |
| `solver_family=unknown` generated-reference rows | 3 |
| `comparison_status=pass` | 52 |

Observed fixture keys:

- `partial_svd_clustered_repeated_diag8x6_k3_v1`
- `partial_svd_fail_closed_diag6_k2_v1`
- `partial_svd_lowrank_rect5x7_k3_sparse_output_v1`
- `partial_svd_rankdef_diag6x4_k2_range_projector_v1`
- `qr_minnorm_3x6_exact_values`
- `qr_minnorm_5x10_exact_values`
- `qr_rank_deficient_6x4_nullspace_v1`
- `qr_rankdef_dependent_row_4x3_v1`
- `qr_rankdef_duplicate_5x4_v1`
- `qr_underdetermined_minnorm_2x4`

The strict oracle normalizer emitted advisory source-controlled contract lines
and generated-present warnings, then exited successfully with:

```text
normalize-report-index: freshness ok (54 rows)
```

The generated-present warning wording is acceptable for Day 4 because the
selected strict oracle policy still returned success and the final row-count,
solver-family, fixture-key, and status checks matched expected values. Day 9
and Day 10 should revisit whether hosted selected rows need less confusing
normalizer wording.

### Comparison Rows

| Field | Count |
| --- | ---: |
| Total selected comparison generated rows | 6 |
| `status=pass` | 6 |
| Source-controlled contract row in normalizer output | 1 |

Selected comparison rows:

- `comparison_qr_underdetermined_minnorm_2x4_project_status_v1`
- `comparison_qr_underdetermined_minnorm_2x4_baseline_status_v1`
- `comparison_qr_underdetermined_minnorm_2x4_residual_norm_v1`
- `comparison_qr_underdetermined_minnorm_2x4_solution_norm_v1`
- `comparison_qr_underdetermined_minnorm_2x4_solution_values_v1`
- `comparison_qr_underdetermined_minnorm_2x4_project_vs_baseline_max_abs_delta_v1`

The strict comparison normalizer exited successfully with:

```text
normalize-report-index: freshness ok (7 rows)
```

Dependency status:

| Dependency | Status | Required | Interpretation |
| --- | --- | --- | --- |
| `python3` | pass | yes | selected interpreter available |
| `tests/qr_external_dense_reference.py` | pass | yes | source-controlled dense reference helper available |
| `numpy` | defer | no | optional package baseline not selected; not pass evidence |
| `scipy` | defer | no | optional package baseline not selected; not pass evidence |

## Output Size Inventory

### Oracle Outputs

| Path | Lines | Bytes |
| --- | ---: | ---: |
| `build/corpus/oracle/corpus.oracle.tsv` | 53 | 56129 |
| `build/corpus-reports/index.tsv` | 54 | 64003 |
| `build/corpus-reports/skips.tsv` | 2 | 493 |
| `build/corpus-reports/manifest.txt` | 16 | 4970 |
| **Total** | **125** | **125595** |

### Comparison Outputs

| Path | Lines | Bytes |
| --- | ---: | ---: |
| `build/comparison/qr_minnorm/project_observations.tsv` | 5 | 490 |
| `build/comparison/qr_minnorm/baseline_observations.tsv` | 5 | 446 |
| `build/comparison/qr_minnorm/dependency_status.tsv` | 5 | 475 |
| `build/comparison/qr_minnorm/study.tsv` | 7 | 11382 |
| `build/comparison/qr_minnorm/summary.md` | 36 | 2206 |
| `build/comparison/qr_minnorm/manifest.tsv` | 24 | 1837 |
| **Total** | **82** | **16836** |

## Hosted Timeout Decision

| Lane | Measured cold runtime | Measured warm runtime | Recommended timeout | Decision |
| --- | ---: | ---: | ---: | --- |
| Oracle selected freshness | 26.22s | 5.67s to 6.65s | 10 minutes | Fits hosted PR budget. Keep as reviewed-hosted candidate. |
| Comparison QR minimum-norm freshness | 21.94s | 1.77s to 1.94s | 10 minutes | Fits hosted PR budget. Keep as reviewed-hosted candidate. |
| Combined selected freshness job | about 48.16s if both cold gates rebuild independently; less if library build is shared | about 7.44s to 8.59s warm combined | 15 minutes | Feasible if serialized and artifact paths remain scoped. |

Day 5 should prefer one serialized hosted job with two selected steps if it can
share the built static library safely. Separate jobs are acceptable if clearer
failure attribution is more important than avoiding duplicate rebuild cost.

## Artifact Retention Decision

| Artifact name | Contents | Retention draft | Normal-pass behavior | Failure behavior |
| --- | --- | ---: | --- | --- |
| `sprint159-oracle-freshness` | `corpus.oracle.tsv`, `index.tsv`, `skips.tsv`, `manifest.txt`, selected-row summary | 7 days | Upload or summarize selected rows. Raw TSV upload is acceptable because the total is about 126 KB. | Always upload raw outputs and diagnostics. |
| `sprint159-comparison-qr-minnorm` | comparison observation TSVs, dependency status, study, summary, manifest | 7 days | Upload or summarize selected rows. Raw output is acceptable because the total is about 17 KB. | Always upload raw outputs and diagnostics. |
| `sprint159-report-index-diagnostics` | strict normalizer output for selected rows only | 7 days | Console summary is enough if artifacts above upload. | Upload diagnostics or preserve log excerpt. |

## Promotion And Demotion Decisions

| Candidate | Day 2 target tier | Day 4 decision | Reason |
| --- | --- | --- | --- |
| S159-H01 QR `oracle/solver_backed` rows | reviewed-hosted candidate | keep candidate | Runtime is stable, row counts match, and artifacts are small. |
| S159-H02 partial-SVD `oracle/solver_backed` rows | reviewed-hosted candidate | keep candidate | Runtime is stable even with combined oracle generation; artifacts are small. |
| S159-H03 `comparison/qr_minnorm` rows | reviewed-hosted candidate | keep candidate | Runtime is stable, six selected rows pass, and optional NumPy/SciPy rows remain clearly deferred. |
| S159-S01 `oracle/generated_reference` rows | supplemental-hosted candidate | keep supplemental candidate | Generated-reference rows are small and useful context, but remain non-primary claim evidence. |

No selected candidate was demoted on runtime or artifact-size grounds.

## CI Preconditions For Day 5

Before Day 5 designs hosted workflow changes, preserve these conditions:

- run selected gates serially;
- keep artifact names scoped to selected oracle and QR minimum-norm comparison
  evidence;
- do not promote broad report-index `--check-freshness`;
- do not treat optional NumPy/SciPy deferred comparison rows as pass evidence;
- do not change generated API HTML publication policy;
- keep selected hosted row wording fixture-local and claim-bounded;
- revisit oracle normalizer wording that reports generated-present warnings
  while the strict selected oracle check succeeds.

## Completion Check

- Selected oracle and comparison commands passed locally.
- Runtime fits a realistic hosted CI budget.
- Output sizes are small enough for short-retention artifacts.
- Dependency and skip behavior is documented.
- CI implementation can proceed with concrete timeout and artifact decisions.
