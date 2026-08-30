# Sprint 189 Day 11: Windows Claim Calibration

## Purpose

Align user-facing and report-facing documentation with the new Windows
PowerShell validation owner while keeping Windows report freshness and broad
Windows parity out of supported claims.

## Implementation Summary

| Surface | Change |
| --- | --- |
| `README.md` | Added the hosted PowerShell validation ownership job to the CI support summary and clarified it is workflow validation only. |
| `INSTALL.md` | Added hosted PowerShell validation ownership to the Windows platform row while retaining Windows non-claims. |
| `tests/corpus/README.md` | Added report-facing guidance that the hosted Windows PowerShell lane owns snippet parsing and structural guards only. |
| `scripts/validate_windows_powershell.py` | Extended claim-boundary markers to require the new README, INSTALL, and corpus README wording. |
| `docs/planning/EPIC_17/SPRINT_189/WORKING_NOTES.md` | Recorded Day 11 user-facing and report-facing claim calibration. |

## Calibrated Support Wording

The public support summary now says Windows has:

- reviewed CMake build/test coverage;
- reviewed CMake install/downstream validation for the static-first package
  surface;
- hosted PowerShell validation ownership for selected Windows workflow
  snippets.

The same surfaces still say Windows does not claim:

- Makefile parity;
- `pkg-config` execution parity;
- report freshness;
- package-manager support;
- shared-library support;
- dynamic ABI support;
- runtime-loader behavior;
- broad Windows parity.

## Report Documentation Boundary

The corpus README now states that selected target rows remain the positive
authority for selected oracle, comparison, and performance freshness, and that
the hosted Windows PowerShell validation lane is not selected report freshness
or selected artifact publication evidence.

## Revisit Criteria

Windows report freshness remains deferred until one reviewed change adds all
of the following:

- a Windows-safe generated report path;
- exact selected upload scope;
- selected-target manifest metadata including `windows`;
- support-tier, claim-scope, and non-claim fields;
- workflow and validator guard updates for the promoted lane.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | User-facing and report-facing claim markers, hosted wiring, fake PowerShell, and unavailable wording guards pass. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/docs/hosted checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Stable maintainer entry point reports unavailable evidence locally. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted/fail-closed mode rejects missing local `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest keeps Windows report freshness deferred. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing workflow guard still rejects selected Windows report freshness commands/uploads. |

## Day 12 Handoff

Day 12 should run the integrated Windows-adjacent validation set and record
the final pass/unavailable/fail-closed evidence for the changed workflow,
validator, report docs, and claim-boundary surfaces.

## Validation Scope

Day 11 changed user-facing documentation, report-facing documentation, a
Python script, and planning documentation. No `.c` or `.h` files were
modified, so `make format && make lint && make test` is not required.
