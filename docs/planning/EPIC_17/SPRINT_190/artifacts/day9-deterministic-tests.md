# Sprint 190 Day 9: Deterministic Tests

## Purpose

Expand deterministic coverage for the Sprint 190 Windows selected Cholesky
freshness decision so workflow, artifact, command, and future manifest drift
fail locally without depending on hosted Windows execution.

## Manifest Test Strategy

Day 9 keeps `tests/corpus/manifests/selected_report_targets.tsv` unchanged.
The source manifest still omits `windows` until Day 10 documentation and claim
calibration can be updated together.

Instead, `tests/test_selected_report_targets_manifest.py` now includes a
simulated future allowlist helper for the exact Windows Cholesky metadata. This
lets the test suite define the intended promotion contract before the manifest
claim is widened.

The helper accepts only:

- `target_id=SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`;
- `workflow_file=.github/workflows/windows-ci.yml`;
- `workflow_job=selected-comparison-freshness`;
- `workflow_artifact=sprint190-windows-selected-comparison-cholesky`;
- `workflow_platforms=windows` in the matching metadata position;
- `expected_rows=6`;
- the standard six-file comparison artifact bundle.

It rejects:

- any non-Cholesky selected target listing `windows`;
- wrong or reused Windows artifact names;
- row-count drift;
- missing required artifact files.

## Workflow Test Expansion

`tests/test_selected_comparison_workflow.py` now validates Windows-specific
drift cases for the hosted Cholesky lane:

- missing `timeout-minutes: 20`;
- generator target drift away from `cholesky-spd-tridiag-5`;
- missing target-specific freshness guard argument;
- wrong Windows artifact name;
- broad `build/comparison/**` upload paths;
- missing required Cholesky upload files.

Existing Linux and macOS selected comparison guard coverage remains unchanged.
Windows still rejects selected oracle freshness, broad selected comparison
freshness, selected benchmark freshness, and reused Linux/macOS artifact names
outside the one selected Cholesky lane.

## PowerShell Validator Expansion

`tests/test_validate_windows_powershell.py` now mirrors the workflow drift
coverage at the standalone validator layer. The validator catches:

- missing Cholesky target-specific freshness argument;
- missing job timeout;
- broad comparison upload paths;
- missing required upload files;
- unexpected artifact uploads outside the selected Cholesky lane.

The fake-`pwsh` tests remain deterministic and do not require ambient
PowerShell unless a test explicitly models hosted `--require-pwsh` behavior.

## Current Claim Boundary

The hosted workflow path exists, but source manifest metadata and public docs
still do not claim reviewed Windows selected report freshness. Day 9 prepares
the deterministic tests needed for the Day 10 claim-calibration patch.

## Validation

Commands run:

- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `python3 tests/test_validate_windows_powershell.py`

All focused Day 9 validation commands passed.

No `.c` or `.h` files were modified, so the full `make format && make lint &&
make test` C gate is not required for Day 9.
