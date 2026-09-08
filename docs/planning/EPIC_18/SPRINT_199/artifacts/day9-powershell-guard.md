# Sprint 199 Day 9: PowerShell Ownership Guard

## Purpose

Strengthen the PowerShell ownership guard around the bounded Windows selected
Cholesky comparison freshness workflow while preserving the Day 4 manifest
re-deferral decision.

## Changed File

| File | Change |
| --- | --- |
| `tests/test_validate_windows_powershell.py` | Added direct guard tests for selected generator target drift, selected artifact-name drift, and fail-closed upload behavior. |

No production validator change was required on Day 9 because
`scripts/validate_windows_powershell.py` already enforces the required
selected Cholesky tokens.

## Guarded Workflow Contract

The validator already requires:

- job ID `selected-comparison-freshness`;
- runner `windows-2022`;
- `timeout-minutes: 20`;
- generator command for `--target cholesky-spd-tridiag-5`;
- MSVC/CMake probe options for `Visual Studio 17 2022`, `x64`, and `Release`;
- static library path `build/Release/sparse_lu_ortho.lib`;
- freshness command with `--selected-target cholesky-spd-tridiag-5`;
- `actions/upload-artifact@v4`;
- artifact name `sprint190-windows-selected-comparison-cholesky`;
- `if-no-files-found: error`;
- the six selected Cholesky upload files under
  `build/comparison/cholesky_spd_tridiag_5/`.

## New Regression Coverage

| Test | Guarded failure |
| --- | --- |
| `test_selected_cholesky_lane_generator_target_drift_fails_clearly` | Replacing `--target cholesky-spd-tridiag-5` with another target fails. |
| `test_selected_cholesky_lane_artifact_name_drift_fails_clearly` | Replacing `sprint190-windows-selected-comparison-cholesky` with another artifact name fails. |
| `test_selected_cholesky_lane_upload_must_fail_closed` | Removing `if-no-files-found: error` fails. |

## Existing Coverage Reconfirmed

| Existing guard | Status |
| --- | --- |
| Selected freshness command must include `--selected-target cholesky-spd-tridiag-5` | Covered |
| Broad `build/comparison/**` upload paths are rejected | Covered |
| Missing required selected Cholesky upload files are rejected | Covered |
| Selected report freshness commands/artifacts outside the bounded lane are rejected | Covered |
| Hosted validation must run `python scripts/validate_windows_powershell.py --require-pwsh` under `shell: cmd` | Covered |
| Local missing `pwsh` is unavailable evidence, not pass evidence | Covered |
| `--require-pwsh` fails closed when `pwsh` is missing | Covered |
| Claim-boundary markers preserve Windows non-claims | Covered |

## Retained Non-Claims

Day 9 preserves the existing non-claims for:

- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- broad Windows report freshness;
- Windows selected oracle freshness;
- Windows selected benchmark freshness;
- package-manager support;
- shared-library support;
- dynamic ABI/runtime-loader support;
- unavailable local PowerShell as pass evidence.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed |
| `make windows-powershell-validate` | Exit `2`: structural checks passed, then local `pwsh` was unavailable. This is unavailable local evidence, not pass evidence. |

## Day 10 Handoff

Day 10 should run the integrated selected freshness gates across the manifest,
normalizer, workflow guard, and report-index comparison path. The expected
state remains re-deferred for Windows in the selected manifest, with the
Windows Cholesky workflow guarded as hosted evidence only.
