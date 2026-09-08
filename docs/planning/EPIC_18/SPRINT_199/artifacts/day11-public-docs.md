# Sprint 199 Day 11 Public Documentation Calibration

## Purpose

Day 11 updates public documentation so the selected Windows Cholesky freshness
wording matches the Sprint 199 evidence record. The documentation now says the
hosted Windows path was reviewed and remains re-deferred for selected freshness
promotion until the selected manifest and generated metadata are promoted
together.

## Updated Public Surfaces

| Surface | Update |
| --- | --- |
| `README.md` normalized report-index section | Replaced pending review wording with the Sprint 199 disposition: hosted CI evidence was reviewed for the exact `cholesky-spd-tridiag-5` path, but Windows promotion remains re-deferred because selected-target metadata, generated support tier, and generated non-claim wording remain local-only. |
| `README.md` QR/report freshness section | Clarified that the Sprint 190 Windows hosted path remains guarded workflow evidence after Sprint 199 review, not promoted Windows selected freshness. |
| `README.md` install handoff section | Cleaned wrapped wording and pointed Windows selected Cholesky freshness promotion to manifest, generated support tier, and generated non-claim alignment. |
| `INSTALL.md` support readiness matrix | Updated the Windows selected Cholesky row from pending evidence review to reviewed-hosted-path-but-re-deferred metadata status. |
| `INSTALL.md` supported platforms table | Added the Sprint 199 reviewed/re-deferred interpretation for the Windows selected Cholesky path. |
| `tests/corpus/README.md` selected target authority section | Recorded that the source manifest still does not list `windows` after Sprint 199 review and explains the metadata blockers. |
| `tests/corpus/README.md` normalized freshness section | Calibrated the Windows hosted path interpretation to guarded workflow evidence with re-deferred selected freshness promotion. |

## Claim Boundary

The public docs now consistently distinguish three facts:

1. Sprint 190 added one guarded Windows hosted workflow path for
   `cholesky-spd-tridiag-5`.
2. Sprint 199 reviewed hosted evidence for that exact path.
3. Sprint 199 still re-deferred selected Windows freshness promotion because
   the selected manifest, generated support tier, and generated non-claim
   wording do not yet promote Windows.

## Retained Non-Claims

Day 11 keeps these non-claims explicit:

- broad Windows report freshness
- Windows selected oracle freshness
- Windows selected benchmark freshness
- QR incompatible Windows comparison promotion
- unselected Windows comparison families
- Windows Makefile parity
- Windows `pkg-config` execution parity
- package-manager support
- shared-library support
- dynamic ABI support
- runtime-loader behavior
- broad Windows parity
- performance superiority or state-of-the-art status

## Owner Surfaces

The selected target manifest remains the authority for platform promotion,
expected rows, required artifacts, support tier, claim scope, and non-claims:

```text
tests/corpus/manifests/selected_report_targets.tsv
```

The Windows workflow and PowerShell validator remain guard surfaces only. They
do not promote selected report freshness without matching selected target
metadata and generated report semantics.

## Validation Results

The Day 11 documentation edit requires docs and claim-boundary checks:

| Command | Result | Notes |
| --- | --- | --- |
| `make docs-check` | Passed | Doxygen generation and API docs coverage completed. |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Claim-boundary markers, selected workflow guards, local unavailable PowerShell behavior, and hosted-required fail-closed behavior all passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Manifest invariants remain aligned with Windows re-deferral. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Workflow guard remains selected-target scoped. |
| `git diff --check` | Passed | No whitespace errors. |

No `.c` or `.h` files were edited for Day 11.
