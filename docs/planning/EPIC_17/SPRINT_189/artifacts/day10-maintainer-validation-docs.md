# Sprint 189 Day 10: Maintainer Validation Docs

## Purpose

Document the owned Windows PowerShell validation command, local unavailable
semantics, hosted fail-closed behavior, and retained Windows report freshness
non-claims in the maintainer-facing policy surface.

## Implementation Summary

| Surface | Change |
| --- | --- |
| `docs/maintainer_guide.md` | Added a `Windows PowerShell Validation Ownership` section. |
| `scripts/validate_windows_powershell.py` | Extended maintainer-guide claim-boundary markers to require the new command and hosted invocation guidance. |
| `docs/planning/EPIC_17/SPRINT_189/WORKING_NOTES.md` | Recorded Day 10 documentation ownership and validation expectations. |

## Maintainer Command Contract

Maintainers now have one documented local entry point:

```sh
make windows-powershell-validate
```

The guide documents that the Make target runs
`python3 scripts/validate_windows_powershell.py` and validates selected
Windows PowerShell workflow snippets, selected report manifest references,
Windows report freshness deferral markers, hosted validation wiring, and
Windows/PowerShell claim-boundary anchors.

## Local Exit Semantics

The maintainer guide now states:

| Exit | Meaning |
| ---: | --- |
| `0` | `pwsh` was available locally and selected snippets parsed after structural checks passed. |
| `2` | Structural checks passed but local `pwsh` was unavailable; this is environment blocker evidence, not pass evidence. |
| `1` | Structural, claim-boundary, hosted wiring, missing-`pwsh` fail-closed, or PowerShell parse failure. |

## Hosted Evidence Contract

The guide now documents the hosted Windows lane:

```sh
python scripts/validate_windows_powershell.py --require-pwsh
```

The hosted job runs on `windows-2022`; `--require-pwsh` makes missing
PowerShell fail closed. A passing hosted job proves validation ownership and
parseability for selected PowerShell snippets only.

## Retained Non-Claims

The maintainer guide keeps these boundaries explicit:

- the hosted validation lane does not upload selected report artifacts;
- it does not run selected report generation commands;
- it does not add `windows` to selected target `workflow_platforms`;
- it does not claim Windows report freshness;
- a future promotion must add a Windows-safe generator path, exact selected
  upload scope, selected-target metadata, and guard updates together.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Maintainer command markers, claim boundaries, hosted wiring, fake PowerShell, and unavailable wording guards pass. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural/report/docs/hosted checks pass; local `pwsh` remains unavailable evidence. |
| `make windows-powershell-validate` | Expected exit `2` | Stable maintainer entry point reports unavailable evidence locally. |
| `python3 scripts/validate_windows_powershell.py --require-pwsh` | Expected exit `1` | Hosted/fail-closed mode rejects missing local `pwsh`. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest keeps Windows report freshness deferred. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing workflow guard still rejects selected Windows report freshness commands/uploads. |

## Day 11 Handoff

Day 11 should align user-facing and report-facing docs with the new validation
owner only if they need outward-facing mention. The retained boundary remains:
PowerShell validation ownership is not Windows report freshness.

## Validation Scope

Day 10 changed maintainer documentation, a Python script, and planning
documentation. No `.c` or `.h` files were modified, so
`make format && make lint && make test` is not required.
