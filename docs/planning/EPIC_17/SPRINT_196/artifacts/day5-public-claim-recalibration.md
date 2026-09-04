# Sprint 196 Day 5 Artifact: Public Claim Recalibration

**Date:** 2026-09-03
**Sprint item coverage:** 196.2
**Day 5 goal:** Update public documentation so support, package, Windows,
comparison, performance, reliability, release, and state-of-the-art claims
match earned Epic 17 evidence.

## Summary

Day 5 calibrated public docs without changing code or public headers. The main
correction is that Windows selected Cholesky report work is described as a
guarded workflow path, not promoted selected freshness, because the selected
target manifest still keeps the Cholesky comparison target at `local_only` with
`linux;macos` workflow platforms.

## Changed Files

| File | Change |
| --- | --- |
| `README.md` | Tightened normalized report-index and selected comparison wording so the Windows Cholesky path is visibly guarded until hosted evidence and manifest promotion are reviewed together. |
| `README.md` | Reflowed the CMake install paragraph so Windows CMake-first support, PowerShell validation ownership, guarded selected Cholesky workflow path, and Sprint 182 residual scope are readable. |
| `INSTALL.md` | Changed the support/readiness matrix row from promoted-looking Windows selected freshness wording to `Windows selected Cholesky comparison workflow` with `guarded-workflow` status. |
| `INSTALL.md` | Reworded the Windows platform row to keep selected freshness promotion dependent on hosted evidence plus manifest metadata. |
| `docs/planning/EPIC_17/SPRINT_196/WORKING_NOTES.md` | Recorded the Day 5 public claim edits, retained non-claims, and validation plan. |

## Evidence Basis

| Evidence | Interpretation |
| --- | --- |
| `tests/corpus/manifests/selected_report_targets.tsv` row `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` | The selected Cholesky comparison target remains `local_only` and lists `linux;macos` workflow platforms. |
| `.github/workflows/windows-ci.yml` | A Windows `selected-comparison-freshness` workflow path exists for `cholesky-spd-tridiag-5`. |
| Sprint 190 closeout and retrospective | The selected Windows residual was narrowed, with hosted evidence and manifest promotion still separate. |
| Day 2 outcome ledger | Sprint 190 is complete with residual narrowed, not broad Windows report freshness completed. |
| Day 3 residual queue E17-RQ-005 | Selected Cholesky Windows freshness promotion needs hosted evidence review and selected manifest metadata promotion. |

## Retained Non-Claims

- No Homebrew or package-manager install support is claimed.
- No shared-library or dynamic ABI support is claimed.
- No broad Windows parity is claimed.
- No broad Windows report freshness is claimed.
- No Windows selected oracle or selected benchmark freshness is claimed.
- No selected Cholesky Windows freshness promotion is claimed until hosted
  evidence, selected metadata, support tier, and claim contract are reviewed
  together.
- No broad external-library parity is claimed.
- No portable performance, timing threshold, release benchmark, or release
  readiness is claimed.
- No broad allocation-failure, OS OOM, or state-of-the-art reliability claim is
  made.
- No unqualified state-of-the-art sparse linear algebra status is claimed.

## Files Intentionally Left Unchanged

| File | Reason |
| --- | --- |
| `benchmarks/README.md` | Existing selected-performance wording is already threshold-free and non-portable. |
| `docs/solver_selection.md` | Existing selected comparison caveats are accurate; any consolidation belongs with maintainer/report-doc calibration. |
| `docs/cookbook.md` | Existing support/readiness routing is accurate. |
| `docs/tutorial.md` | Existing local tutorial versus support/report boundary is accurate. |
| `examples/README.md` | Existing installed-consumer and benchmark routing is accurate. |
| `include/*.h` | No concrete header overclaim was found; avoiding header edits avoids expanding Day 5 into C/header validation. |

## Validation Results

The changed files are public docs and planning docs only. No `.c` or `.h`
files were modified, so the full C quality gate is not required for Day 5.

- `git diff --check`: passed.
- `make windows-powershell-guard`: passed.
- `bash scripts/package_manager_deferral_check.sh`: passed.
- `bash scripts/static_package_deferral_check.sh`: passed.
- `python3 tests/test_selected_report_targets_manifest.py`: passed.
- `python3 tests/test_selected_performance_docs.py`: passed.
- `python3 tests/test_normalize_report_index.py`: passed.
- `make docs-check`: passed.

Guard-driven follow-up:

- The Windows PowerShell guard requires exact README/INSTALL marker phrases for
  hosted PowerShell validation ownership, bounded selected Cholesky workflow
  wording, and the Sprint 182 deferral marker. Day 5 preserved those markers
  while clarifying that the Windows Cholesky path is guarded, not promoted
  selected freshness.
- The package-manager guard requires lowercase README non-claim markers for
  `package-manager support`, `local Homebrew formula proof`, and
  `package-manager distribution`. Day 5 preserved those markers.
