# Sprint 209 Day 3: Hosted Evidence Inventory

## Purpose

Inspect current hosted Windows evidence for `qr-incompatible-ls` before deciding
whether Sprint 209 can promote selected Windows QR incompatible metadata or must
first add a bounded workflow proof path.

## Commands Run

```text
gh run list --workflow "Windows CI" --branch master --limit 10 --json databaseId,displayTitle,event,headSha,createdAt,updatedAt,status,conclusion,url
gh run view 35768806616 --json databaseId,name,displayTitle,event,headSha,createdAt,updatedAt,status,conclusion,url,jobs
gh api repos/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/35768806616/artifacts --jq '.artifacts[] | [.id, .name, .size_in_bytes, .created_at, .expired, .archive_download_url] | @tsv'
gh run download 35768806616 --name sprint190-windows-selected-comparison-cholesky --dir "$tmpdir"
find "$tmpdir" -type f | sort
rg -n "qr|incompatible|cholesky|comparison_" "$tmpdir"
```

## Latest Hosted Windows Run

| Field | Value |
| --- | --- |
| Workflow | `Windows CI` |
| Run ID | `35768806616` |
| Run URL | `https://github.com/jeffreyhorn/linalg_sparse_orthogonal/actions/runs/35768806616` |
| Event | `push` |
| Display title | `Merge pull request #231 from jeffreyhorn/sprint-208` |
| Branch | `master` |
| Head SHA | `a98c7b593d24c7514f01f7dbe443fbeeb541f5af` |
| Created | `2026-09-22T18:41:08Z` |
| Updated | `2026-09-22T18:44:14Z` |
| Conclusion | `success` |

## Job Inventory

| Job | Started | Completed | Conclusion | Interpretation |
| --- | --- | --- | --- | --- |
| `Windows PowerShell validation ownership` | `2026-09-22T18:41:13Z` | `2026-09-22T18:41:26Z` | `success` | Confirms Windows workflow ownership and claim-boundary validation, not QR report evidence. |
| `Windows selected Cholesky comparison freshness (MSVC)` | `2026-09-22T18:41:13Z` | `2026-09-22T18:41:55Z` | `success` | Existing bounded Cholesky-only selected comparison proof. |
| `Windows reviewed CMake install/downstream validation path` | `2026-09-22T18:41:13Z` | `2026-09-22T18:42:38Z` | `success` | Static CMake install/downstream proof, not selected QR report evidence. |
| `Windows enforced reviewed CMake consumer subset (MSVC)` | `2026-09-22T18:41:13Z` | `2026-09-22T18:44:13Z` | `success` | General Windows CMake test lane, not selected QR report evidence. |

## Artifact Inventory

| Artifact ID | Name | Size | Created | Expired |
| ---: | --- | ---: | --- | --- |
| `10712913903` | `sprint190-windows-selected-comparison-cholesky` | `4598` bytes | `2026-09-22T18:41:51Z` | `false` |

No artifact with `qr`, `qr-incompatible`, `qr_incompatible_ls`, or a Sprint 209
QR-specific name exists in the latest hosted Windows run.

## Downloaded Artifact Membership

The only selected comparison artifact downloaded from run `35768806616` contains
these files:

| File | Observed target |
| --- | --- |
| `project_observations.tsv` | Cholesky SPD tridiagonal. |
| `baseline_observations.tsv` | Cholesky SPD tridiagonal. |
| `dependency_status.tsv` | Cholesky SPD tridiagonal. |
| `study.tsv` | Six Cholesky selected rows. |
| `summary.md` | Cholesky SPD tridiagonal summary. |
| `manifest.tsv` | Cholesky SPD tridiagonal metadata. |

The artifact does not contain `build/comparison/qr_incompatible_ls/`, QR
incompatible file names, QR incompatible row IDs, or QR incompatible manifest
metadata.

## Artifact Content Summary

| Field | Observed value |
| --- | --- |
| Target | `cholesky-spd-tridiag-5` |
| Fixture key | `cholesky_spd_tridiag_5` |
| Source commit | `a98c7b593d24c7514f01f7dbe443fbeeb541f5af` |
| Source branch | `master` |
| Worktree state | `clean` |
| Platform | `windows-amd64` |
| Compiler | `cmake-probe:Visual Studio 17 2022:Release` |
| Support tier | `local_only` |
| Non-claims | Includes `no Windows report freshness`. |

The row set is:

- `comparison_cholesky_spd_tridiag_5_project_status_v1`
- `comparison_cholesky_spd_tridiag_5_baseline_status_v1`
- `comparison_cholesky_spd_tridiag_5_residual_norm_v1`
- `comparison_cholesky_spd_tridiag_5_solution_norm_v1`
- `comparison_cholesky_spd_tridiag_5_solution_values_v1`
- `comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1`

None of the required Day 2 QR row IDs are present.

## Day 3 Decision

The latest hosted Windows evidence does not support QR incompatible selected
Windows freshness promotion. It proves that the Windows workflow remains healthy
after Sprint 208 and that the existing Cholesky lane still uploads its bounded
artifact, but there is no hosted MSVC `qr-incompatible-ls` probe, no QR artifact
bundle, and no QR row-set inspection evidence.

Sprint 209 must therefore proceed to workflow design and implementation before
any manifest promotion decision. Until an exact QR hosted proof exists,
`SRT-COMP-QR-INCOMPATIBLE-LS` remains Linux/macOS-only, `local_only`, and
bounded by `no Windows report freshness`.
