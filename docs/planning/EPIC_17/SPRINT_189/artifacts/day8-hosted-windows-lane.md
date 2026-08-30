# Sprint 189 Day 8: Hosted Windows CI Wiring

## Purpose

Wire the owned PowerShell validation command into hosted Windows CI while
keeping Windows report freshness, selected report artifact publication, and
broad Windows parity claims formally deferred.

## Implementation Summary

| Surface | Change |
| --- | --- |
| `.github/workflows/windows-ci.yml` | Added a dedicated `powershell-validation` job on `windows-2022`. |
| `.github/workflows/windows-ci.yml` | The hosted job runs `python scripts/validate_windows_powershell.py --require-pwsh` under `shell: cmd`. |
| `scripts/validate_windows_powershell.py` | Added structural validation for the hosted Windows validation job. |
| `tests/test_validate_windows_powershell.py` | Added drift tests for missing `--require-pwsh`, wrong runner, and accidental `shell: pwsh` on the hosted validation step. |
| `docs/planning/EPIC_17/SPRINT_189/WORKING_NOTES.md` | Recorded hosted CI wiring and retained non-claims. |

## Hosted Job Contract

The new workflow job is:

| Field | Value |
| --- | --- |
| Job id | `powershell-validation` |
| Runner | `windows-2022` |
| Step | `Validate owned Windows PowerShell workflow material` |
| Command | `python scripts/validate_windows_powershell.py --require-pwsh` |
| Shell | `cmd` |

The job is intentionally separate from the existing CMake build/test and
install/downstream jobs. It validates PowerShell ownership and snippet parsing;
it does not generate reports, upload selected report artifacts, or promote
Windows report freshness.

## Fail-Closed Behavior

Hosted Windows uses `--require-pwsh`, so the validation command fails if:

- `pwsh` is missing on the hosted runner;
- selected workflow snippets fail PowerShell parse validation;
- the hosted validation job loses `windows-2022`;
- the hosted validation command loses `--require-pwsh`;
- the hosted validation step changes from `shell: cmd` to `shell: pwsh`;
- selected report freshness commands or selected artifact names appear in the
  Windows workflow while Sprint 182 deferral remains active.

## Why `shell: cmd`

The hosted validation step runs under `cmd` so it can invoke Python without
becoming another selected `shell: pwsh` workflow snippet. The validator then
uses the hosted `pwsh` executable directly to parse the five owned PowerShell
snippets. This keeps the ownership set explicit and prevents the validation
step from validating itself as unowned PowerShell material.

## Expected CI Evidence

A passing hosted run should include:

```text
windows-powershell-validate: hosted Windows PowerShell validation wiring ok
windows-powershell-validate: PowerShell parse validation (5 snippets) ok
windows-powershell-validate: passed (5 selected PowerShell snippets)
```

Local machines without `pwsh` still return exit `2` in default mode. That
local unavailable result remains non-pass evidence and is not a substitute for
the hosted Windows job.

## Retained Non-Claims

Day 8 does not claim:

- Windows report freshness;
- selected Windows report artifact publication;
- Windows report generator execution;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support;
- shared-library, dynamic ABI, DLL/import-library, or runtime-loader support;
- broad Windows parity beyond the reviewed CMake/install/downstream and
  PowerShell-validation lanes.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Hosted wiring and local/fake PowerShell paths are guarded. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural and hosted wiring checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Stable maintainer entry point reports unavailable evidence locally. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted/fail-closed mode rejects missing local `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest keeps Windows report freshness deferred. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing workflow guard still rejects selected Windows report freshness commands/uploads. |

## Day 9 Handoff

Day 9 should deepen ownership guard coverage and add claim-drift checks around
the new hosted validation lane so documentation cannot imply Windows report
freshness from PowerShell validation ownership alone.

## Validation Scope

Day 8 changed a workflow file, Python tests, a Python script, and planning
documentation. No `.c` or `.h` files were modified, so
`make format && make lint && make test` is not required.
