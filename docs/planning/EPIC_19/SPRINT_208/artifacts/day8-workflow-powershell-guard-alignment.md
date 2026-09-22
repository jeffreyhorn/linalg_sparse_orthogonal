# Sprint 208 Day 8 Workflow And PowerShell Guard Alignment

## Scope

Day 8 aligned the Windows workflow and PowerShell validation guard surface with
the Sprint 208 Day 5 re-deferral decision. The Windows hosted workflow may keep
one bounded selected Cholesky comparison freshness lane, but the selected target
manifest must not list Windows while generated metadata and claim surfaces still
retain re-deferral wording.

## Implementation

Changed files:

- `scripts/validate_windows_powershell.py`
- `tests/test_validate_windows_powershell.py`

Guard changes:

- added an exact selected Cholesky manifest re-deferral contract to the
  Windows PowerShell validator;
- required `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` to retain `support_tier=local_only`;
- required Cholesky workflow metadata to remain the current Linux/macOS pair:
  `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`,
  `generated-report-freshness`, `selected-comparison-freshness`,
  `sprint175-linux-selected-comparison-freshness`,
  `sprint175-macos-selected-comparison-freshness`, and `linux;macos`;
- required the Cholesky non-claim set to retain `no Windows report freshness`
  and the existing package, ABI, performance, and state-of-the-art non-claims;
- added unsupported-claim detection for positive wording such as
  `Windows selected Cholesky freshness is promoted`.

Regression coverage added:

- Cholesky manifest row cannot append a Windows platform while deferral is
  active;
- Cholesky manifest row cannot replace the macOS selected artifact with the
  Windows selected artifact;
- Cholesky manifest row cannot remove `no Windows report freshness`;
- public docs cannot introduce direct Windows selected Cholesky promotion
  wording while the deferral contract is active.

## Validation

Commands run:

```sh
python3 tests/test_validate_windows_powershell.py
python3 tests/test_selected_comparison_workflow.py
python3 -m py_compile scripts/validate_windows_powershell.py tests/test_validate_windows_powershell.py
```

Results:

- `test-validate-windows-powershell: ok`
- `test-selected-comparison-workflow: ok`
- Python compilation completed without diagnostics.

## Completion Notes

Day 8 closes the workflow/PowerShell guard portion of Item 208.3. The selected
Windows Cholesky lane remains bounded to hosted workflow evidence and cannot be
reinterpreted as source-controlled selected Windows manifest freshness without
coherent manifest and claim-boundary edits.
