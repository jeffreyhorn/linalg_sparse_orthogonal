# Day 9: Workflow Guard Integration

## Purpose

Day 9 hardened the selected comparison workflow guard so the re-deferred
Windows QR incompatible target cannot be partially promoted by workflow drift.

## Decision

No Windows workflow lane was added for `qr-incompatible-ls` on Day 9. The
selected Windows comparison workflow remains scoped to the previously promoted
Cholesky target until hosted Windows/MSVC QR incompatible evidence exists.

## Guard Changes

| Surface | Day 9 coverage |
| --- | --- |
| `tests/test_selected_comparison_workflow.py` | Added `assert_windows_qr_incompatible_remains_redeferred()` to reject accidental Windows QR target commands, equals-form target commands, selected freshness commands, artifact names, QR subfamily tokens, and QR artifact upload paths. |
| `.github/workflows/windows-ci.yml` | Unchanged; the workflow remains Cholesky-specific for selected comparison freshness. |
| `tests/corpus/manifests/selected_report_targets.tsv` | Unchanged; QR incompatible remains Linux/macOS-only and `local_only`. |

## Negative Fixtures

| Fixture | Expected failure |
| --- | --- |
| `test_windows_qr_incompatible_target_drift_fails_clearly()` | A Windows workflow line running `run_external_comparison.py --target qr-incompatible-ls` fails clearly. |
| `test_windows_qr_incompatible_freshness_target_drift_fails_clearly()` | A Windows workflow line running selected freshness for `qr-incompatible-ls` fails clearly. |
| `test_windows_qr_incompatible_equals_target_drift_fails_clearly()` | A Windows workflow line running `run_external_comparison.py --target=qr-incompatible-ls` fails clearly. |
| `test_windows_qr_incompatible_subfamily_drift_fails_clearly()` | A Windows workflow reference to the QR incompatible subfamily path fails clearly. |
| `test_windows_qr_incompatible_artifact_upload_drift_fails_clearly()` | A Windows workflow upload path for a QR incompatible generated artifact fails clearly. |

The new tests are called from the standalone
`tests/test_selected_comparison_workflow.py` runner.

## Validation

| Command | Result |
| --- | --- |
| `python3 tests/test_selected_comparison_workflow.py` | Passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed. |
| `python3 -m py_compile tests/test_selected_comparison_workflow.py` | Passed. |
| `git diff --check -- docs/planning/EPIC_18/SPRINT_203 tests/test_normalize_report_index.py tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py` | Passed. |

## Promotion Boundary

Day 9 does not claim Windows QR incompatible report freshness. It adds a guard
against partial workflow promotion before the hosted Windows/MSVC probe,
selected artifact inspection, manifest metadata, and documentation claim
updates are available as one coherent evidence set.
