# Day 2: MSVC Probe Design

## Purpose

Day 2 defined the exact Windows/MSVC/CMake probe needed before Sprint 203 can
promote or re-defer the selected `qr-incompatible-ls` comparison target. The
day did not change generator code, manifest metadata, workflow YAML, or public
claim surfaces.

## Selected Target Contract

| Field | Value |
| --- | --- |
| Target id | `SRT-COMP-QR-INCOMPATIBLE-LS` |
| Target key | `qr-incompatible-ls` |
| Fixture | `qr_overdetermined_incompatible_4x2` |
| Operation | Incompatible QR least-squares solve with expected nonzero residual. |
| Artifact directory | `build/comparison/qr_incompatible_ls/` |
| Required files | `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, `manifest.tsv` |
| Expected rows | 6 selected rows: project status, baseline status, residual norm, solution norm, solution values, and project-vs-baseline max absolute delta. |
| Current platforms | Linux and macOS only. |
| Windows status | Not promoted until hosted MSVC/CMake evidence and metadata align. |

## Canonical Probe Sequence

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
| Render include and library paths with forward slashes | Prevents CMake string escapes such as `\a` in drive-letter paths. |
| Link `m` only under `if(NOT MSVC)` | Preserves MSVC compatibility for the generated temporary project. |
| Run target-specific freshness with `--selected-target qr-incompatible-ls` | Prevents unselected comparison rows from satisfying the proof. |

## Expected Evidence

Promotion requires:

- hosted Windows configure/build/probe/freshness pass for the exact target;
- inspected artifact bundle containing only the six QR incompatible files;
- `study.tsv` row ids matching the selected manifest contract;
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
| Wrong or incomplete selected row set | Blocks promotion with target-specific row diagnostics. |
| Stale generated rows | Blocks promotion until regenerated for current commit. |
| Windows path mismatch | Requires normalizer/path matching fix before promotion. |
| Overbroad support tier or non-claim wording | Requires re-deferral or coordinated metadata/docs update. |

## Day 2 Decision

Day 3 should run or review the canonical probe sequence above. If hosted
Windows evidence is not available, Sprint 203 must keep promotion blocked or
explicitly re-defer rather than relying on Linux/macOS or local-only output.
