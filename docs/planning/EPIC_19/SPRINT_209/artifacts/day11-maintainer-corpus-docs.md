# Sprint 209 Day 11: Maintainer And Corpus Docs

## Purpose

Bring maintainer, corpus, schema, and planning status surfaces into agreement
with the Sprint 209 QR incompatible re-deferral decision.

## Documentation Changes

| File | Change |
| --- | --- |
| `docs/maintainer_guide.md` | Adds Sprint 209 to the Windows/PowerShell and selected comparison evidence-owner rows, while keeping local missing PowerShell and hosted QR collection separate from pass evidence. |
| `docs/maintainer_guide.md` | Documents the `qr-incompatible-ls` Windows lane as guarded evidence collection only; selected Windows QR freshness remains re-deferred. |
| `tests/corpus/README.md` | Updates corpus interpretation so the Sprint 209 QR lane does not imply broad report-index freshness, selected oracle/benchmark freshness, broad Windows report freshness, or local-only family support. |
| `tests/corpus/schemas/report_index_fields.md` | Records that the QR incompatible selected row stays Linux/macOS-only and `local_only` until hosted run evidence, exact artifact inspection, manifest metadata, generated support tier, and generated non-claim wording move together. |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | Updates the then-current Day 11 Sprint 209 status and keeps later sprints pending future execution. |

## Guard Changes

| Guard | Change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Requires Sprint 209 QR evidence-collection wording in maintainer guide, corpus README, and report-index schema docs. |
| `tests/test_validate_windows_powershell.py` | Adds marker-removal regressions for maintainer, corpus, and schema documentation. |

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed. |
| `make windows-powershell-guard` | Passed. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py` | Passed. |
| `make docs-check` | Passed. |
| `make support-docs-guard` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |

## Closeout

Day 11 completes Item 209.5 documentation and claim-guard calibration. The
Sprint 209 branch now presents one consistent QR decision: the Windows workflow
lane exists to collect future hosted evidence, but the selected QR manifest and
support claims remain unpromoted.
