# Sprint 199 Day 2: Hosted Artifact Inventory

## Purpose

Inspect hosted Windows evidence for the exact selected Cholesky target,
artifact paths, row IDs, artifact metadata, workflow provenance, and initial
promotion gaps. Day 2 records inventory only; Day 3 owns evidence semantics and
Day 4 owns manifest promotion or re-deferral.

## Hosted Workflow Run

| Field | Value |
| --- | --- |
| Workflow | `Windows CI` |
| Run ID | `34269219871` |
| Run URL | `https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/34269219871` |
| Event | `push` |
| Display title | `Merge pull request #220 from jeffreyhorn/sprint-198` |
| Branch | `master` |
| Head SHA | `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450` |
| Created | `2026-09-08T19:29:02Z` |
| Updated | `2026-09-08T19:32:40Z` |
| Conclusion | `success` |

## Hosted Job Inventory

| Job | Started | Completed | Conclusion |
| --- | --- | --- | --- |
| `Windows selected Cholesky comparison freshness (MSVC)` | `2026-09-08T19:29:05Z` | `2026-09-08T19:29:45Z` | `success` |
| `Windows PowerShell validation ownership` | `2026-09-08T19:29:06Z` | `2026-09-08T19:29:20Z` | `success` |
| `Windows reviewed CMake install/downstream validation path` | `2026-09-08T19:29:06Z` | `2026-09-08T19:30:29Z` | `success` |
| `Windows enforced reviewed CMake consumer subset (MSVC)` | `2026-09-08T19:29:07Z` | `2026-09-08T19:32:39Z` | `success` |

The selected job is the only Sprint 199 hosted report-freshness input. The
other jobs support Windows workflow and install/readiness context but do not
promote selected report freshness by themselves.

## Workflow Command Baseline

The selected job in `.github/workflows/windows-ci.yml`:

- configures the library with `cmake -S . -B build -G "Visual Studio 17 2022"
  -A x64`;
- builds `sparse_lu_ortho` with `cmake --build build --config Release --target
  sparse_lu_ortho`;
- generates the selected report with `python
  scripts/run_external_comparison.py --target cholesky-spd-tridiag-5
  --probe-build-system cmake --cmake-generator "Visual Studio 17 2022"
  --cmake-arch x64 --cmake-config Release --library
  build/Release/sparse_lu_ortho.lib`;
- checks freshness with `python scripts/normalize_report_index.py --family
  comparison --require-generated comparison --check-freshness
  --selected-target cholesky-spd-tridiag-5`;
- uploads `sprint190-windows-selected-comparison-cholesky`.

## Artifact Metadata

| Field | Value |
| --- | --- |
| Artifact ID | `10073117703` |
| Artifact name | `sprint190-windows-selected-comparison-cholesky` |
| Size | `4592` bytes |
| Created | `2026-09-08T19:29:41Z` |
| Updated | `2026-09-08T19:29:41Z` |
| Expired | `false` |

Day 2 downloaded the artifact with:

```sh
gh run download 34269219871 --name sprint190-windows-selected-comparison-cholesky --dir /tmp/sprint199-day1-windows-artifact-check
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

The file set matches the selected workflow upload list and the current
manifest `required_files` contract.

## Row Inventory

`study.tsv` contains six rows. All expected row IDs are present, no unexpected
row IDs are present, and every row has `status=pass`.

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
| `source_commit` | `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450` |
| `source_branch` | `master` |
| `worktree_state` | `clean` |
| `generated_at_utc` | `2026-09-08T19:29:37+00:00` |
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

## Promotion Gaps

Day 2 found credible hosted Windows artifact evidence, but not enough by
itself to promote the manifest:

1. Generated `artifact_path` values use Windows backslashes
   (`build\comparison\cholesky_spd_tridiag_5\study.tsv`), while the selected
   manifest uses forward slashes. Days 5 through 7 must add explicit
   normalizer coverage before promotion relies on target-specific artifact
   filtering.
2. Generated rows still carry `support_tier=local_only` and non-claims
   including `no Windows report freshness`. Day 3 and Day 4 must decide
   whether this metadata should be promoted, reworded, or kept as a
   source-controlled generated-row limitation while public docs describe only
   hosted evidence.
3. The artifact proves one hosted `master` run for one target. It does not
   prove broad Windows report freshness, selected oracle freshness, selected
   benchmark freshness, QR incompatible Windows freshness, external-library
   ecosystem parity, package/ABI support, performance superiority, release
   readiness, or state-of-the-art status.

## Day 3 Handoff

Day 3 should convert this inventory into evidence semantics:

- whether the hosted artifact row set, statuses, commit metadata, and workflow
  provenance are sufficient for selected Windows Cholesky freshness promotion;
- whether `support_tier=local_only` in generated rows blocks manifest
  promotion or requires a narrowly scoped metadata change;
- which exact claim text can be promoted without contradicting existing
  non-claims;
- which normalizer/path tests must land before any manifest platform change.
