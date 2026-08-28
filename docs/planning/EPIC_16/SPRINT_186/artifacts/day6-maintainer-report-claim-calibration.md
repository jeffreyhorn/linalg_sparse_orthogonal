# Sprint 186 Day 6: Maintainer and Report Claim Calibration

## Purpose

Calibrate maintainer and report documentation so Epic 16 selected report,
Windows freshness, generated report, and comparison claims match the Day 3
reconciled evidence matrix.

## Scope

Day 6 addresses these Day 4 calibration items:

| ID | Surface | Day 6 status |
| --- | --- | --- |
| D4-CAL-004 | README and maintainer Windows sections | Maintainer report guidance calibrated; README was already claim-safe after Day 5 review. |
| D4-CAL-005 | report-index and selected report docs | Calibrated in maintainer and corpus/report schema docs. |
| D4-CAL-008 | comparison report docs and selected manifest | Calibrated by reinforcing selected-target-only interpretation. |
| D4-CAL-010 | state-of-the-art/support-tier language | Reviewed for Day 6 surfaces; no platform, package/ABI, performance, release, or state-of-the-art claim was added. |

## Documentation Changes

| File | Change |
| --- | --- |
| `docs/maintainer_guide.md` | Added Sprint 186 closeout wording that keeps the selected target manifest as positive evidence only for listed Linux/macOS selected targets and treats unavailable local PowerShell checks as environment residuals rather than Windows report freshness evidence. |
| `tests/corpus/README.md` | Added Sprint 186 closeout wording that separates positive selected target rows from Windows report freshness, unavailable PowerShell validation, optional dependency skips, and absent generated local reports. |
| `tests/corpus/schemas/report_index_fields.md` | Added schema guidance that formal deferrals and environment residuals belong in planning/closeout evidence, not fake selected rows or widened `workflow_platforms`. |

## Earned Claims Preserved

| Claim family | Day 6 result |
| --- | --- |
| Selected target manifest authority | Preserved as positive authority for selected oracle, comparison, and benchmark target metadata. |
| Selected comparison freshness | Preserved for manifest-selected QR, partial-SVD, LU, and Cholesky comparison families only. |
| Cholesky selected comparison | Preserved as `cholesky_spd_tridiag_5` fixture-local evidence only. |
| Windows reviewed CMake support | Preserved as CMake/MSVC configure, build, `ctest`, and static-first install/downstream validation. |

## Non-Claims Preserved

Day 6 preserves these non-claims:

- selected target rows do not widen report-family claims;
- missing local generated rows do not count as pass evidence;
- optional dependency skips and defers do not count as pass evidence;
- unavailable local PowerShell checks remain environment residuals;
- Windows selected report freshness remains formally deferred;
- no Windows report-generation platform was added to selected target rows;
- no broad report-index freshness, unselected report-family freshness,
  package/ABI support, performance, release-readiness, external-library
  parity, or state-of-the-art claim was added.

## Residuals Carried Forward

| Residual | Day 6 handling |
| --- | --- |
| R186-WIN-PWSH | Remains active. Local `pwsh` is unavailable, so PowerShell parse/workflow checks need a suitable environment or hosted validation ownership. |
| R186-WIN-REPORT-FRESHNESS | Remains active. Windows selected report freshness is still a formal product deferral. |
| R186-BROAD-COMPARISON | Remains active. Comparison evidence remains selected-fixture-only, including the Cholesky target added in Sprint 183. |

## Validation

Day 6 changed documentation files only. No `.c` or `.h` files were modified, so
the full C quality gate is not required.

Required focused validation:

```sh
python3 scripts/validate_corpus_schema.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_normalize_report_index.py
git diff --check
```
