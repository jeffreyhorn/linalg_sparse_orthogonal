# Day 11: Documentation Calibration

## Purpose

Day 11 calibrated user, install, maintainer, corpus, and report-index schema
documentation so Sprint 203 cannot be read as Windows QR incompatible selected
freshness promotion.

## Documentation Updated

| Surface | Calibration |
| --- | --- |
| `README.md` | States that `qr-incompatible-ls` remains outside Windows selected freshness until hosted MSVC probe evidence, selected artifact review, selected-target manifest metadata, generated support tier, and generated non-claim wording are promoted together. |
| `INSTALL.md` | Adds a deferred Windows QR incompatible least-squares row to the support readiness matrix and repeats the deferred Windows QR boundary in the platform matrix section. |
| `docs/maintainer_guide.md` | Clarifies that the Sprint 190 Windows Cholesky workflow is guarded workflow evidence only and that QR incompatible Windows selected freshness remains deferred. |
| `tests/corpus/README.md` | Adds the QR incompatible Windows re-deferral marker to the selected comparison freshness corpus documentation. |
| `tests/corpus/schemas/report_index_fields.md` | Adds the same selected-report-index schema boundary so target manifests remain the authority for future promotion. |

## Guard Updated

`scripts/validate_windows_powershell.py` now requires the QR incompatible
Windows selected freshness deferral marker in:

- `README.md`;
- `INSTALL.md`;
- `docs/maintainer_guide.md`;
- `tests/corpus/README.md`.

## Retained Non-Claims

The calibrated docs preserve these Sprint 203 boundaries:

- no broad QR parity;
- no broad least-squares parity;
- no broad Windows report freshness;
- no selected Windows QR freshness;
- no package-manager proof;
- no shared-library ABI proof;
- no performance superiority;
- no state-of-the-art claim.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed. |
| `python3 scripts/validate_windows_powershell.py` | Structural and claim-boundary checks passed; exited `2` because local `pwsh` is unavailable. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py` | Passed. |
| `python3 tests/test_selected_performance_docs.py` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |

## Promotion Boundary

Day 11 updates documentation and guard markers only. It does not modify
selected target manifest metadata, Windows workflow upload paths, generated
support tiers, comparison generator behavior, or public support claims.
