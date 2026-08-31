# Sprint 189 Day 4: Local Validation Command Scaffold

## Purpose

Implement the first owned PowerShell validation command and stable local
maintainer entry point from the Day 3 command contract.

## Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Added the owned validation script for selected Windows CI PowerShell material. |
| `Makefile` | Added `windows-powershell-validate` as the stable maintainer entry point. |
| `docs/planning/EPIC_17/SPRINT_189/WORKING_NOTES.md` | Recorded scaffold behavior, validation evidence, and Day 5 handoff. |

## Command Behavior

| Invocation | Local result | Interpretation |
| --- | --- | --- |
| `python3 scripts/validate_windows_powershell.py` | Exit `2` | Structural checks passed, but local `pwsh` is unavailable. This is unavailable evidence, not pass evidence. |
| `make windows-powershell-validate` | Exit `2` | Make exposes the same local unavailable state from the script. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Exit `1` | Hosted/fail-closed mode rejects missing `pwsh`. |

## Structural Checks Added

The script validates these inputs without requiring local `pwsh`:

- `.github/workflows/windows-ci.yml` exists;
- the workflow retains the Sprint 182 Windows report freshness deferral
  comment;
- `build-and-test` and `install-and-downstream` jobs exist;
- both selected Windows jobs run on `windows-2022`;
- selected Windows workflow steps still declare `shell: pwsh`;
- selected PowerShell steps contain their expected command and evidence
  anchors;
- Windows CI remains free of selected report freshness commands and selected
  report upload artifact names;
- the Sprint 182 deferral artifact retains the formal deferral marker;
- `tests/corpus/manifests/selected_report_targets.tsv` contains no `windows`
  workflow platform while the deferral remains active.

## Selected PowerShell Snippets

The Day 4 scaffold owns five selected snippets:

| Job | Step |
| --- | --- |
| `build-and-test` | `Run enforced reviewed CMake configure path (MSVC, x64)` |
| `build-and-test` | `Run enforced reviewed CMake build path (Release)` |
| `build-and-test` | `Inspect enforced Windows reviewed consumer CTest surface (ctest -N)` |
| `build-and-test` | `Run enforced reviewed CMake execution path (ctest)` |
| `install-and-downstream` | `Run reviewed CMake install/downstream validation proof` |

## PowerShell Parse Path

When `pwsh` is available, the script writes each selected workflow `run` block
to a temporary snippet and asks PowerShell to parse it with
`[scriptblock]::Create(...)`. This validates syntax without executing CMake,
CTest, report generators, upload steps, or generated artifact commands.

## Retained Non-Claims

Day 4 does not promote:

- Windows report freshness;
- selected Windows report artifact publication;
- broad Windows platform parity;
- Windows Makefile parity;
- Windows `pkg-config` execution parity;
- package-manager support;
- shared-library package support;
- dynamic ABI or runtime-loader behavior.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Local `pwsh` is unavailable after structural checks pass. |
| `make windows-powershell-validate` | Expected exit `2` | Make target forwards the local unavailable state. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted/fail-closed mode rejects missing `pwsh`. |

## Day 5 Handoff

Day 5 should expand workflow snippet validation by tightening command
reference checks, adding focused drift tests where useful, and confirming the
selected `shell: pwsh` ownership remains stable across Windows workflow
changes.

## Validation Scope

Day 4 changed a Python script, the Makefile, and planning documentation. No
`.c` or `.h` files were modified, so `make format && make lint && make test`
is not required.
