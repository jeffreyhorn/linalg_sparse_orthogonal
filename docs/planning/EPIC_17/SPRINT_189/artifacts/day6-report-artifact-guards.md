# Sprint 189 Day 6: Report Artifact and Manifest Validation

## Purpose

Deepen the PowerShell validation owner so selected report artifact names,
manifest references, and Windows report freshness deferral assumptions are
validated from source-controlled report metadata.

## Implementation Summary

| Surface | Change |
| --- | --- |
| `scripts/validate_windows_powershell.py` | Added selected report manifest reference validation. |
| `scripts/validate_windows_powershell.py` | Added manifest-derived selected report freshness tokens so Windows CI blockers come from `selected_report_targets.tsv`. |
| `tests/test_validate_windows_powershell.py` | Added tests for manifest-derived artifact blockers and missing workflow-file references. |
| `docs/planning/EPIC_17/SPRINT_189/WORKING_NOTES.md` | Recorded Day 6 artifact-name and manifest validation coverage. |

## Manifest Reference Coverage

The validator now checks every row in
`tests/corpus/manifests/selected_report_targets.tsv` for:

- selected family limited to `oracle`, `comparison`, or `benchmark`;
- non-empty `workflow_file`, `workflow_job`, `workflow_artifact`, and
  `workflow_platforms` metadata;
- workflow artifact cardinality of either one shared artifact name or one
  artifact name per workflow platform;
- referenced workflow files existing in the repository;
- sprint-scoped selected artifact names;
- no `windows` entry in `workflow_platforms` while the Sprint 182 deferral is
  active.

## Manifest-Derived Windows Blockers

Windows CI selected report freshness blockers now include:

- fixed selected freshness commands and artifact names already guarded by
  `tests/test_selected_comparison_workflow.py`;
- every manifest `generator_command`;
- every manifest `workflow_job`;
- every manifest `workflow_artifact`.

This means new selected report artifacts added to the manifest are
automatically forbidden in Windows CI while Windows report freshness remains
deferred, without requiring a separate hard-coded update in the PowerShell
validator.

## Artifact Inventory

Current selected workflow artifacts remain Linux/macOS scoped:

| Artifact | Source |
| --- | --- |
| `sprint159-oracle-freshness` | Selected oracle row. |
| `sprint175-linux-selected-comparison-freshness` | Selected comparison rows on Linux. |
| `sprint175-macos-selected-comparison-freshness` | Selected comparison rows on macOS. |
| `sprint168-selected-performance-freshness` | Selected benchmark row on Linux. |

No selected Windows report artifact is introduced on Day 6.

## Report Family Boundary

The selected manifest currently carries only selected oracle, comparison, and
benchmark rows. Package, coverage, dead-code, guardrail, documentation, API,
and Windows report publication surfaces remain outside Sprint 189 report
freshness promotion.

## Drift Coverage

| Drift scenario | Expected failure |
| --- | --- |
| Windows CI mentions a manifest-owned selected artifact name. | `windows workflow must not run or upload selected report freshness ...` |
| A manifest row references a missing workflow file. | `references missing workflow_file ...` |
| A manifest row has mismatched artifact/platform cardinality. | `workflow_artifact must contain one shared artifact or one artifact per workflow platform` |
| A manifest row lists `windows` in `workflow_platforms`. | `selected_report_targets.tsv must not list windows ...` |
| A manifest row uses an unexpected selected family. | `unexpected selected family ...` |

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Manifest-derived artifact blockers and workflow-reference checks behave as expected. |
| `python3 scripts/validate_windows_powershell.py` | Expected exit `2` | Structural, report, and manifest checks pass; local `pwsh` remains unavailable evidence. |
| `python3 scripts/validate_corpus_schema.py` | Passed | Corpus schema baseline remains valid. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Selected manifest keeps Windows report freshness deferred. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Existing workflow guard still rejects selected Windows report freshness commands/uploads. |

## Day 7 Handoff

Day 7 should focus on local `pwsh` available/unavailable behavior. Since this
local environment does not provide `pwsh`, Day 7 should verify the unavailable
path remains explicit and add any practical self-test coverage that can run
without requiring PowerShell.

## Validation Scope

Day 6 changed Python tests, a Python script, and planning documentation. No
`.c` or `.h` files were modified, so `make format && make lint && make test`
is not required.
