# Sprint 199 Day 3: Evidence Semantics

## Purpose

Interpret the hosted Windows artifact inventory for
`cholesky-spd-tridiag-5` and define the threshold Day 4 must use before
promoting or re-deferring selected target metadata.

Day 3 does not change the selected target manifest. It records what the hosted
artifact proves, what it does not prove, and which blockers must be resolved
before a source-controlled Windows freshness claim can be promoted.

## Evidence Inputs

| Input | Value |
| --- | --- |
| Selected target | `cholesky-spd-tridiag-5` |
| Manifest target ID | `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` |
| Fixture/subfamily | `cholesky_spd_tridiag_5` |
| Workflow | `Windows CI` |
| Workflow run | `34269219871` |
| Run URL | `https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/34269219871` |
| Selected job | `Windows selected Cholesky comparison freshness (MSVC)` |
| Workflow artifact | `sprint190-windows-selected-comparison-cholesky` |
| Artifact ID | `10073117703` |
| Source commit | `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450` |
| Source branch | `master` |
| Worktree state | `clean` |
| Platform | `windows-amd64` |
| Compiler/build path | `cmake-probe:Visual Studio 17 2022:Release` |
| Generated timestamp | `2026-09-08T19:29:37+00:00` |

## What The Hosted Artifact Proves

The hosted evidence proves this narrow statement:

`cholesky-spd-tridiag-5` generated a selected comparison report on hosted
Windows using the MSVC/CMake probe path, uploaded the expected selected
artifact, and produced the six expected selected comparison rows with
`status=pass` for commit `98d57edc2f73dd0ca8c7e05fbf476e43b30a0450`.

The proof is bounded by all of the following facts:

- the workflow run completed successfully on `master`;
- the selected Windows freshness job completed successfully;
- the uploaded artifact was present, non-expired during review, and named
  `sprint190-windows-selected-comparison-cholesky`;
- the artifact contained `project_observations.tsv`,
  `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`,
  `summary.md`, and `manifest.tsv`;
- `study.tsv` contained exactly the six expected Cholesky row IDs;
- all generated selected rows had `status=pass`;
- row provenance recorded `platform=windows-amd64`,
  `source_branch=master`, clean worktree state, and the reviewed source
  commit;
- the workflow command used `--selected-target cholesky-spd-tridiag-5` for
  target-specific freshness checking.

## What The Hosted Artifact Does Not Prove

The hosted artifact does not prove:

- broad Windows report freshness;
- selected oracle freshness on Windows;
- selected benchmark freshness on Windows;
- QR incompatible Windows freshness;
- Windows freshness for any selected comparison target except
  `cholesky-spd-tridiag-5`;
- Linux or macOS promotion changes;
- NumPy, SciPy, LAPACK, SuiteSparse, Eigen, or external-library ecosystem
  parity;
- package-manager distribution, Homebrew/core readiness, bottles, Linuxbrew,
  shared-library ABI, dynamic loader, release, performance, or
  state-of-the-art claims.

The Windows PowerShell validation job, Windows install/downstream job, and
Windows reviewed CMake consumer job are supporting context only. They are not
generated selected comparison row evidence.

## Freshness Field Interpretation

| Field | Day 3 interpretation |
| --- | --- |
| `target` / `target_key` | Sufficiently scoped to `cholesky-spd-tridiag-5`; no other target can inherit this proof. |
| `platform` | `windows-amd64` supports Windows-hosted evidence for this target only. |
| `compiler` | `cmake-probe:Visual Studio 17 2022:Release` supports the MSVC/CMake probe path only. |
| `library path` | The workflow passes `build/Release/sparse_lu_ortho.lib`; this is static library probe evidence, not shared-library or ABI evidence. |
| `artifact_path` | Generated rows use `build\comparison\cholesky_spd_tridiag_5\study.tsv`; promotion should wait for explicit Windows path normalization coverage. |
| `generated_at_utc` | Timestamp ties the row set to the reviewed hosted run; it is not a recurring freshness guarantee by itself. |
| `support_tier` | Generated rows still say `local_only`; this conflicts with any direct hosted-selected manifest promotion unless metadata wording is updated or explicitly explained. |
| `non_claims` | Generated summary still includes `no hosted CI proof` and `no Windows report freshness`; these must be reconciled before public promotion. |

## Promotion Threshold

Day 4 may promote `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` to include Windows only if
all criteria below are true in the same reviewed change:

1. The source-controlled selected target manifest names the exact target,
   expected files, expected row IDs, workflow file, workflow job, workflow
   artifact, and Windows platform.
2. The hosted artifact evidence remains tied to the reviewed branch or merged
   baseline commit and has no stale, expired, wrong-target, wrong-platform, or
   missing-row ambiguity.
3. `normalize_report_index.py` has tests proving selected artifact filtering
   accepts Windows backslash and mixed-separator paths without accepting
   near-match artifacts.
4. Generated report metadata and public documentation agree on one narrow
   claim: selected Windows Cholesky comparison freshness for
   `cholesky-spd-tridiag-5`.
5. Existing non-claims remain explicit for broad Windows report freshness,
   Windows oracle freshness, Windows benchmark freshness, other comparison
   targets, external-library parity, package/ABI support, performance, release,
   and state-of-the-art status.

If any criterion is false, Day 4 should keep Windows re-deferred for the
selected manifest and record the exact blocker.

## Gap And Ambiguity Classification

| Gap | Classification | Day 4 / Later action |
| --- | --- | --- |
| Backslash `artifact_path` in hosted rows | Promotion blocker | Add explicit normalizer path tests before relying on selected target filtering. |
| Generated row `support_tier=local_only` | Promotion blocker | Decide whether hosted selected Windows evidence needs a new support tier or a narrower claim explanation. |
| Generated summary says `no hosted CI proof` | Promotion blocker | Reconcile generated summary/non-claim text before public promotion. |
| Generated summary says `no Windows report freshness` | Promotion blocker | Replace only if the manifest and docs can preserve the narrow selected-target claim. |
| Optional `numpy` and `scipy` dependencies are `defer` | Context only | Do not count as pass evidence or blocker for source-controlled dense-helper proof. |
| PowerShell validation job passes | Context only | Keep as workflow ownership evidence, not generated report freshness evidence. |
| Windows install/downstream and reviewed CMake jobs pass | Context only | Keep as Windows build/support context, not selected report freshness evidence. |

## Day 4 Inputs

Day 4 should make one of two decisions:

- Promote Windows for `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` only if the path,
  support-tier, generated summary, manifest, and public wording blockers can
  be resolved together.
- Re-defer Windows for the selected manifest and cite the unresolved blockers
  if any of those surfaces cannot be aligned without overstating evidence.
