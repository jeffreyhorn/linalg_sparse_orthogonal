# Sprint 208 Day 2: Hosted Artifact Inventory

## Purpose

Locate and inspect the current hosted Windows evidence for the exact selected
Cholesky target, including workflow run provenance, job status, artifact
metadata, artifact membership, selected row IDs, generated row metadata, and
initial promotion blockers.

## Hosted Workflow Run

| Field | Value |
| --- | --- |
| Workflow | `Windows CI` |
| Run ID | `35731703320` |
| Run URL | `https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/35731703320` |
| Event | `push` |
| Display title | `Merge pull request #230 from jeffreyhorn/sprint-207` |
| Branch | `master` |
| Head SHA | `d75118349269c6805654070eb44c9c57608b3e47` |
| Created | `2026-09-22T13:10:04Z` |
| Updated | `2026-09-22T13:12:23Z` |
| Conclusion | `success` |

## Hosted Job Inventory

| Job | Started | Completed | Conclusion |
| --- | --- | --- | --- |
| `Windows PowerShell validation ownership` | `2026-09-22T13:10:09Z` | `2026-09-22T13:10:20Z` | `success` |
| `Windows selected Cholesky comparison freshness (MSVC)` | `2026-09-22T13:10:09Z` | `2026-09-22T13:10:41Z` | `success` |
| `Windows reviewed CMake install/downstream validation path` | `2026-09-22T13:10:10Z` | `2026-09-22T13:11:36Z` | `success` |
| `Windows enforced reviewed CMake consumer subset (MSVC)` | `2026-09-22T13:10:09Z` | `2026-09-22T13:12:22Z` | `success` |

The selected Cholesky job is the only Day 2 hosted report-freshness input. The
other Windows jobs support workflow validation and CMake/install context, but
they do not promote selected report freshness by themselves.

## Workflow Command Evidence

The selected job in `.github/workflows/windows-ci.yml` ran the maintained
bounded command path:

- configured with `cmake -S . -B build -G "Visual Studio 17 2022" -A x64`;
- built `sparse_lu_ortho` with
  `cmake --build build --config Release --target sparse_lu_ortho`;
- generated the selected report with
  `python scripts/run_external_comparison.py --target cholesky-spd-tridiag-5
  --probe-build-system cmake --cmake-generator "Visual Studio 17 2022"
  --cmake-arch x64 --cmake-config Release --library
  build/Release/sparse_lu_ortho.lib`;
- checked freshness with
  `python scripts/normalize_report_index.py --family comparison
  --require-generated comparison --check-freshness --selected-target
  cholesky-spd-tridiag-5`;
- uploaded `sprint190-windows-selected-comparison-cholesky`.

The hosted log reported `normalize-report-index: freshness ok (17 rows)` and
uploaded exactly six files.

## Artifact Metadata

| Field | Value |
| --- | --- |
| Artifact ID | `10696020870` |
| Artifact name | `sprint190-windows-selected-comparison-cholesky` |
| Artifact URL | `https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/35731703320/artifacts/10696020870` |
| Size | `4598` bytes |
| Created | `2026-09-22T13:10:38Z` |
| Updated | `2026-09-22T13:10:38Z` |
| Expires | `2026-12-21T13:10:04Z` |
| Expired | `false` |
| Upload digest | `fab335a3b95e958111d500e74ce52b5630c85e8d3ae29b8967eeb75b73f72dda` |

Day 2 downloaded the artifact with:

```sh
gh run download 35731703320 --name sprint190-windows-selected-comparison-cholesky --dir /tmp/sprint208-day2-windows-artifact-check.compx4
```

## Artifact File Set

| File | Day 2 status |
| --- | --- |
| `project_observations.tsv` | Present. |
| `baseline_observations.tsv` | Present. |
| `dependency_status.tsv` | Present. |
| `study.tsv` | Present. |
| `summary.md` | Present. |
| `manifest.tsv` | Present. |

The file set matches the current workflow upload list and selected manifest
`required_files` contract.

## Row Inventory

`study.tsv` contains six rows. All expected row IDs are present in the
manifest order, no unexpected row IDs were observed, and every row has
`status=pass`.

| Row ID | Metric | Status | Status reason |
| --- | --- | --- | --- |
| `comparison_cholesky_spd_tridiag_5_project_status_v1` | `project_status` | `pass` | `project_status_match` |
| `comparison_cholesky_spd_tridiag_5_baseline_status_v1` | `baseline_status` | `pass` | `baseline_status_success` |
| `comparison_cholesky_spd_tridiag_5_residual_norm_v1` | `residual_norm` | `pass` | `project_baseline_residual_delta_within_tolerance` |
| `comparison_cholesky_spd_tridiag_5_solution_norm_v1` | `solution_norm` | `pass` | `project_baseline_solution_norm_delta_within_tolerance` |
| `comparison_cholesky_spd_tridiag_5_solution_values_v1` | `solution_values` | `pass` | `project_baseline_solution_values_delta_within_tolerance` |
| `comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1` | `project_vs_baseline_max_abs_delta` | `pass` | `project_baseline_max_abs_delta_within_tolerance` |

## Row Provenance

| Field | Value |
| --- | --- |
| `platform` | `windows-amd64` |
| `compiler` | `cmake-probe:Visual Studio 17 2022:Release` |
| `source_commit` | `d75118349269c6805654070eb44c9c57608b3e47` |
| `source_branch` | `master` |
| `worktree_state` | `clean` |
| `generated_at_utc` | `2026-09-22T13:10:35+00:00` |
| `artifact_path` | `build\comparison\cholesky_spd_tridiag_5\study.tsv` |
| `support_tier` | `local_only` |
| `claim_scope` | `fixture-local Cholesky SPD tridiagonal solve comparison only` |

## Observation Files

`project_observations.tsv` records:

| Metric | Value | Status |
| --- | --- | --- |
| `project_status` | `SPARSE_SUCCESS` | `pass` |
| `residual_norm` | `5.7560540319981793e-15` | `pass` |
| `solution_norm` | `7.4161984870956648` | `pass` |
| `solution_values` | `1,2,3.0000000000000009,4.0000000000000009,5.0000000000000009` | `pass` |

`baseline_observations.tsv` records matching source-controlled dense Cholesky
reference values with `baseline_status=success`.

`dependency_status.tsv` records required `python3` and
`tests/chol_external_dense_reference.py` as `pass`. Optional `numpy` and
`scipy` package baselines are `defer`; deferred package rows are not pass
evidence.

## Manifest Comparison

The selected target manifest row for `cholesky-spd-tridiag-5` currently says:

| Manifest field | Value |
| --- | --- |
| `target_id` | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| `workflow_platforms` | `linux;macos` |
| `support_tier` | `local_only` |
| `artifact_pattern` | `build/comparison/cholesky_spd_tridiag_5/study.tsv` |
| `expected_rows` | `6` |
| `workflow_artifact` | `sprint175-linux-selected-comparison-freshness;sprint175-macos-selected-comparison-freshness` |
| `non_claims` | Includes `no Windows report freshness`, package/ABI non-claims, performance non-claim, and state-of-the-art non-claim. |

The hosted artifact matches the manifest row IDs and row count, but it does
not match the positive manifest platform metadata because the manifest does
not yet list `windows`.

## Promotion Gaps

Day 2 found credible current hosted Windows artifact evidence, but not enough
by itself to promote selected Windows Cholesky freshness:

1. The selected manifest still lists only `linux;macos` workflow platforms.
2. Generated rows still carry `support_tier=local_only`.
3. Generated rows and `summary.md` still include `no hosted CI proof` and
   `no Windows report freshness`.
4. Generated `artifact_path` values use Windows backslashes
   (`build\comparison\cholesky_spd_tridiag_5\study.tsv`), while manifest
   metadata uses forward slashes.

## Day 3 Handoff

Day 3 should trace the selected row IDs and artifact paths across the manifest,
normalizer, generated rows, and hosted artifact. The key question is whether
the remaining gaps are only claim-semantics blockers or whether additional
path-normalization and artifact-mismatch regressions must land before any
promotion decision.

