# Sprint 199 Day 10 Gate Integration

## Purpose

Day 10 connects the selected target manifest, report normalizer, Windows
workflow contract, and PowerShell ownership guard into one reviewable gate for
the bounded `cholesky-spd-tridiag-5` comparison freshness lane.

## Integrated Gate State

The selected Cholesky comparison freshness path is locally validated for the
current branch and remains intentionally re-deferred for Windows manifest
promotion.

`SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` still records:

- `workflow_platforms=linux;macos`
- `support_tier=local_only`
- a retained non-claim for `no Windows report freshness`

That state matches the Day 4 manifest decision. The hosted Windows artifact is
valid evidence for one bounded MSVC/CMake lane, but the source-controlled
manifest and generated support/non-claim metadata have not yet been promoted
to make Windows freshness authoritative.

## Validated Surfaces

| Surface | Command or review | Result |
| --- | --- | --- |
| Selected target manifest | `python3 tests/test_selected_report_targets_manifest.py` | Passed |
| Windows workflow selected comparison contract | `python3 tests/test_selected_comparison_workflow.py` | Passed |
| PowerShell ownership and claim-boundary tests | `python3 tests/test_validate_windows_powershell.py` | Passed |
| Normalizer selected-target/path/diagnostic tests | `python3 tests/test_normalize_report_index.py` | Passed |
| Selected Cholesky freshness check | `python3 scripts/normalize_report_index.py --family comparison --require-generated comparison --check-freshness --selected-target cholesky-spd-tridiag-5` | Passed |
| Local generated comparison freshness gate | `make report-index-comparison-freshness` | Passed |

## Pass Behavior

The selected Cholesky freshness command passes when generated comparison rows
for `cholesky-spd-tridiag-5` are present, current to `HEAD`, and matched
through the selected artifact path.

The Makefile comparison freshness target regenerates all selected local
comparison outputs, then verifies generated comparison freshness. Its terminal
message remains scoped to local evidence:

```text
report-index-comparison-freshness: passed (local-only generated comparison freshness)
```

The selected Cholesky rows reported as fresh were:

- `comparison_cholesky_spd_tridiag_5_project_status_v1`
- `comparison_cholesky_spd_tridiag_5_baseline_status_v1`
- `comparison_cholesky_spd_tridiag_5_residual_norm_v1`
- `comparison_cholesky_spd_tridiag_5_solution_norm_v1`
- `comparison_cholesky_spd_tridiag_5_solution_values_v1`
- `comparison_cholesky_spd_tridiag_5_project_vs_baseline_max_abs_delta_v1`

## Failure Behavior

Focused regression tests now prove that selected freshness does not silently
pass when the evidence is malformed or mis-targeted:

- Windows backslash, mixed-separator, and absolute suffix artifact paths match
  the selected artifact path.
- Near-match Cholesky artifact paths do not match the selected artifact path.
- Generated rows for the wrong comparison target produce a selected
  `comparison_selected_rows` row-set mismatch for
  `cholesky-spd-tridiag-5`.
- Unknown `--selected-target` values fail with selected manifest validation
  guidance.
- `--selected-target` without `--check-freshness` remains a fast-fail misuse.
- Windows workflow drift in generator target, artifact name, upload file set,
  or `if-no-files-found: error` is caught by PowerShell guard tests.

These paths identify the selected target, expected row set, observed row count,
missing rows, selected artifact, or workflow token that caused the failure.

## Claim Boundary Review

Day 10 did not promote broad Windows report freshness. The generated comparison
gate remains local-only, and the selected manifest still excludes Windows for
workflow platforms.

Still unclaimed:

- broad Windows report freshness
- Windows oracle freshness
- Windows benchmark freshness
- QR incompatible Windows comparison promotion
- Homebrew/core readiness, bottles, Linuxbrew, or general package-manager
  distribution
- shared-library package support or dynamic ABI compatibility
- performance superiority, external-library parity, or state-of-the-art status

## Residuals

1. Generated comparison rows still carry `support_tier=local_only`.
2. Generated summary/non-claim wording still includes no hosted CI proof and
   no Windows report freshness language.
3. Public and maintainer documentation have not yet been calibrated for the
   Day 4 re-deferral; Days 11 and 12 own that work.
4. Hosted Windows CI remains required for pass evidence of the bounded Windows
   workflow path. Local PowerShell unavailability remains unavailable evidence,
   not pass evidence.
