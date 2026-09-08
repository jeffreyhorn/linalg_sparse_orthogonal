# Sprint 199 Day 8: Workflow Command Alignment

## Purpose

Review the Windows selected Cholesky comparison workflow against the Day 4
manifest re-deferral decision and Day 7 normalizer hardening.

## Decision

No workflow edit was made on Day 8.

The existing `.github/workflows/windows-ci.yml` selected comparison freshness
job is correctly bounded to the guarded workflow path and does not need to be
changed while the selected target manifest remains unpromoted for Windows.

## Workflow Contract Reviewed

| Field | Current value | Day 8 disposition |
| --- | --- | --- |
| Job ID | `selected-comparison-freshness` | Aligned |
| Job name | `Windows selected Cholesky comparison freshness (MSVC)` | Aligned |
| Runner | `windows-2022` | Aligned |
| CMake generator | `Visual Studio 17 2022` | Aligned |
| CMake arch | `x64` | Aligned |
| CMake config | `Release` | Aligned |
| Library path | `build/Release/sparse_lu_ortho.lib` | Aligned |
| Generator target | `cholesky-spd-tridiag-5` | Aligned |
| Freshness target | `--selected-target cholesky-spd-tridiag-5` | Aligned |
| Upload artifact | `sprint190-windows-selected-comparison-cholesky` | Aligned |

## Upload Scope

The workflow uploads only the six expected files for the selected Cholesky
artifact:

- `build/comparison/cholesky_spd_tridiag_5/project_observations.tsv`;
- `build/comparison/cholesky_spd_tridiag_5/baseline_observations.tsv`;
- `build/comparison/cholesky_spd_tridiag_5/dependency_status.tsv`;
- `build/comparison/cholesky_spd_tridiag_5/study.tsv`;
- `build/comparison/cholesky_spd_tridiag_5/summary.md`;
- `build/comparison/cholesky_spd_tridiag_5/manifest.tsv`.

No QR, partial-SVD, LU, oracle, benchmark, package, ABI, performance, release,
or broad Windows report artifact is uploaded by this job.

## Alignment With Manifest Decision

Day 4 kept `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` at
`workflow_platforms=linux;macos` until all promotion blockers are resolved.
The Day 8 workflow review preserves that decision:

- the Windows workflow remains guarded evidence, not promoted manifest
  authority;
- the workflow target and artifact names are precise enough for future
  promotion;
- the current selected target manifest remains the source-controlled authority
  for promoted platforms;
- broad Windows report freshness remains a non-claim.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Passed |
| `python3 tests/test_validate_windows_powershell.py` | Passed |
| `python3 tests/test_normalize_report_index.py` | Passed |

## Hosted Verification Checklist

When a future PR is ready for promotion, reviewers should confirm a hosted
Windows run where:

1. `Windows selected Cholesky comparison freshness (MSVC)` succeeds.
2. The generated artifact is named
   `sprint190-windows-selected-comparison-cholesky`.
3. The artifact contains the six expected selected files.
4. `study.tsv` contains the six expected Cholesky row IDs with `status=pass`.
5. Row provenance is `platform=windows-amd64`,
   `compiler=cmake-probe:Visual Studio 17 2022:Release`, clean worktree, and
   the reviewed branch or merged baseline commit.
6. Normalizer selected-target freshness diagnostics pass with
   `--selected-target cholesky-spd-tridiag-5`.

## Day 9 Handoff

Day 9 should review the PowerShell ownership guard and claim-boundary tests.
The guard should continue to treat the Windows Cholesky workflow as bounded
workflow evidence until manifest metadata and public claim wording are
promoted together.
