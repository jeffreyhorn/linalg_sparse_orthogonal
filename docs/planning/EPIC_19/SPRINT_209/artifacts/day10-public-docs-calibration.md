# Sprint 209 Day 10: Public Docs Calibration

## Purpose

Update README and INSTALL so users can see the Sprint 209 QR incompatible
decision without inferring promoted selected Windows QR freshness or broader
Windows support.

## Public Documentation Changes

| File | Change |
| --- | --- |
| `README.md` | Adds Sprint 209 as a bounded Windows QR incompatible evidence-collection lane for `qr-incompatible-ls`, while stating that the selected manifest remains Linux/macOS-only and `local_only` until hosted Windows run evidence and exact QR artifact inspection support promotion. |
| `README.md` | Keeps the selected comparison section explicit that QR incompatible least-squares remains outside Windows selected freshness until hosted MSVC probe evidence, selected artifact review, selected manifest metadata, generated support tier, and generated non-claim wording move together. |
| `INSTALL.md` | Updates the support/readiness matrix QR row to `guarded-workflow deferred`, naming Sprint 209 workflow/PowerShell and manifest guards without presenting the lane as support. |
| `INSTALL.md` | Updates the Windows platform row to include the Sprint 209 evidence-collection lane while preserving broad Windows, package-manager, shared-library, ABI, runtime-loader, selected oracle/benchmark, performance, release, and state-of-the-art non-claims. |

## Guard Changes

| Guard | Change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | README and INSTALL markers now require the Sprint 209 QR evidence-collection wording. |
| `scripts/validate_windows_powershell.py` | Unsupported-claim matching rejects QR incompatible Windows selected freshness promotion wording. |
| `tests/test_validate_windows_powershell.py` | Regression coverage now fails if the public QR marker is removed or if public docs add positive QR Windows selected freshness wording. |

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed. |
| `python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py` | Passed. |
| `make windows-powershell-guard` | Passed. |
| `make docs-check` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |

## Closeout

Day 10 completes public README/INSTALL calibration for the Day 8 re-deferral
decision. Maintainer, corpus, schema, and planning-doc calibration remain
scheduled for Day 11.
