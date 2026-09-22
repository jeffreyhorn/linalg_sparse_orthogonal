# Sprint 209 Day 2: MSVC Probe Design

## Purpose

Define the exact hosted Windows/MSVC/CMake proof required before Sprint 209 can
promote or re-defer the selected `qr-incompatible-ls` comparison target. Day 2
does not change generator code, manifest metadata, workflow YAML, or public
claim surfaces.

## Selected Target Contract

| Field | Value |
| --- | --- |
| Target id | `SRT-COMP-QR-INCOMPATIBLE-LS` |
| Family | `comparison` |
| Subfamily | `qr_incompatible_ls` |
| Target key | `qr-incompatible-ls` |
| Fixture | `qr_overdetermined_incompatible_4x2` |
| Operation | Incompatible QR least-squares solve with expected nonzero residual. |
| Artifact directory | `build/comparison/qr_incompatible_ls/` |
| Artifact pattern | `build/comparison/qr_incompatible_ls/study.tsv` |
| Required files | `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv` |
| Expected rows | Six selected rows: project status, baseline status, residual norm, solution norm, solution values, and project-vs-baseline max absolute delta. |
| Current platforms | Linux and macOS only. |
| Current support tier | `local_only`. |
| Windows status | Not promoted until hosted MSVC/CMake evidence, artifact inspection, generated support tier, manifest metadata, and docs align. |

## Expected Row IDs

| Row ID | Meaning |
| --- | --- |
| `comparison_qr_overdetermined_incompatible_4x2_project_status_v1` | Project solve status. |
| `comparison_qr_overdetermined_incompatible_4x2_baseline_status_v1` | Source-controlled dense baseline status. |
| `comparison_qr_overdetermined_incompatible_4x2_residual_norm_v1` | Expected nonzero incompatible least-squares residual norm. |
| `comparison_qr_overdetermined_incompatible_4x2_solution_norm_v1` | Project solution norm. |
| `comparison_qr_overdetermined_incompatible_4x2_solution_values_v1` | Project solution value vector. |
| `comparison_qr_overdetermined_incompatible_4x2_project_vs_baseline_max_abs_delta_v1` | Maximum project-vs-baseline solution delta. |

## Canonical Hosted Probe Sequence

```text
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target sparse_lu_ortho
python scripts/run_external_comparison.py --target qr-incompatible-ls --probe-build-system cmake --cmake-generator "Visual Studio 17 2022" --cmake-arch x64 --cmake-config Release --library build/Release/sparse_lu_ortho.lib
python scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target qr-incompatible-ls
```

This sequence mirrors the reviewed Windows Cholesky workflow shape while
changing only the selected target and selected artifact family. It is not a
broad Windows comparison proof.

## CMake Probe Requirements

| Requirement | Rationale |
| --- | --- |
| Use `--probe-build-system cmake` | Exercises the downstream CMake consumer path instead of direct compiler mode. |
| Use `Visual Studio 17 2022`, `x64`, and `Release` | Matches the reviewed hosted Windows workflow configuration and expected `.lib` layout. |
| Pass `--library build/Release/sparse_lu_ortho.lib` | Avoids the Unix static-library default on Windows. |
| Run target-specific freshness with `--selected-target qr-incompatible-ls` | Prevents unselected comparison rows from satisfying the proof. |
| Upload only exact files under `build/comparison/qr_incompatible_ls/` | Prevents broad Windows comparison artifact publication. |
| Keep artifact upload fail-closed with `if-no-files-found: error` | Missing selected files must block promotion. |

## Required Artifact Layout

| Artifact path | Promotion purpose |
| --- | --- |
| `build/comparison/qr_incompatible_ls/project_observations.tsv` | Confirms project status row source for the exact QR incompatible fixture. |
| `build/comparison/qr_incompatible_ls/baseline_observations.tsv` | Confirms dense reference helper status and keeps baseline failures distinct. |
| `build/comparison/qr_incompatible_ls/dependency_status.tsv` | Records helper/dependency availability and prevents dependency failure from being read as project pass evidence. |
| `build/comparison/qr_incompatible_ls/study.tsv` | Holds the six selected rows checked by freshness and manifest contracts. |
| `build/comparison/qr_incompatible_ls/summary.md` | Human-readable selected proof summary for artifact review. |
| `build/comparison/qr_incompatible_ls/manifest.tsv` | Artifact membership and generation metadata for inspection. |

## Expected Evidence

Promotion requires:

- hosted Windows configure/build/probe/freshness pass for the exact target;
- inspected artifact bundle containing only the six QR incompatible files;
- `study.tsv` row IDs matching the selected manifest contract;
- Windows-safe artifact paths accepted by the normalizer;
- dependency status separated from project build status;
- manifest metadata, generated support tier, claim scope, and docs agreeing on
  the selected Windows QR boundary.

## Failure Diagnostics To Preserve

| Diagnostic | Promotion effect |
| --- | --- |
| CMake configure/build failure | Blocks promotion until repo-owned fix or environment residual is recorded. |
| Missing `tests/qr_external_dense_reference.py` or failed baseline helper | Blocks promotion and must not be collapsed into project status. |
| Missing required artifact file | Blocks promotion and should fail workflow freshness. |
| Wrong, duplicate, stale, or incomplete selected row set | Blocks promotion with target-specific row diagnostics. |
| Windows path mismatch | Requires normalizer/path matching fix before promotion. |
| Broad artifact upload scope | Requires workflow guard fix before promotion. |
| Overbroad support tier or stale non-claim wording | Requires re-deferral or coordinated metadata/docs update. |

## Day 2 Decision

Day 2 keeps Sprint 209 in design mode. Day 3 should inspect current hosted
Windows evidence if any exists. If hosted evidence is not already available,
Day 4-Day 5 should add a bounded `qr-incompatible-ls` workflow path using the
canonical command and exact artifact contract above before any manifest
promotion decision.
