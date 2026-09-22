# Sprint 208 Day 11 Public Documentation Calibration

## Scope

Day 11 calibrated user-facing README and INSTALL wording for the Sprint 208
Windows selected Cholesky decision. The public documentation now points users
at the current bounded hosted evidence and guard surfaces while preserving the
re-deferred selected Windows freshness boundary.

## Changed Files

- `README.md`
- `INSTALL.md`

## Public Wording Updates

`README.md` now states that Sprint 208:

- reviewed the current bounded Windows Cholesky hosted path for
  `cholesky-spd-tridiag-5`;
- strengthened manifest, workflow/PowerShell, and normalizer guards;
- kept selected Windows freshness re-deferred;
- keeps selected target metadata, workflow upload names, claim scopes, and
  non-claims in `tests/corpus/manifests/selected_report_targets.tsv`;
- links users to `INSTALL.md#support-readiness-matrix` and Sprint 208 working
  notes for support interpretation.

The selected comparison section now says the Sprint 190 Windows Cholesky
artifact is evidence for that exact path and re-deferred selected Windows
freshness only.

`INSTALL.md` now updates the support/readiness matrix and platform table so the
Windows selected Cholesky row and Windows platform notes reference Sprint 208
instead of the older Sprint 199 review. The retained non-claims still say there
is no promoted Windows selected freshness until selected metadata, generated
support tier, and generated non-claim wording are promoted together.

## Retained Non-Claims

Day 11 did not claim:

- broad Windows report freshness;
- Windows selected oracle freshness;
- Windows selected benchmark freshness;
- Windows QR incompatible selected freshness;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support;
- shared-library or dynamic ABI support;
- runtime-loader support;
- performance, release, external-library parity, or state-of-the-art evidence.

## Validation

Commands run:

```sh
python3 tests/test_validate_windows_powershell.py
python3 tests/test_selected_comparison_workflow.py
python3 tests/test_selected_report_targets_manifest.py
rg -n "Windows selected (Cholesky|comparison|report) freshness is promoted|Windows report freshness is supported|PowerShell validation proves Windows report freshness" README.md INSTALL.md docs/maintainer_guide.md tests/corpus/README.md tests/corpus/schemas/report_index_fields.md
```

Results:

- `test-validate-windows-powershell: ok`
- `test-selected-comparison-workflow: ok`
- `test-selected-report-targets-manifest: ok`
- The forbidden broad-claim search returned no matches.

## Completion Notes

Item 208.5 now has public-facing documentation aligned with the current
Sprint 208 evidence and guard state. Maintainer/corpus/schema documentation
alignment remains owned by Day 12.
