# Sprint 182 Day 9: Manifest And Support-Tier Alignment

## Purpose

Align selected target manifest semantics and report-index support-tier wording
with the Sprint 182 Windows report freshness decision.

## Alignment Decision

Windows report freshness remains a formal deferral, not a selected target.
`tests/corpus/manifests/selected_report_targets.tsv` therefore stays unchanged:
it remains positive selected-target authority for Linux and macOS selected
freshness lanes and does not gain a fake Windows deferral row.

The deferral is represented by
`docs/planning/EPIC_16/SPRINT_182/artifacts/windows-report-freshness-deferral-decision.md`
and guarded by the selected workflow and manifest regression tests added in
Days 7-8.

## Manifest State

| Manifest | Day 9 action | Reason |
| --- | --- | --- |
| `tests/corpus/manifests/selected_report_targets.tsv` | No row change | The selected-target manifest records selected workflow platforms. Since Windows is deferred, adding `windows` would falsely promote selected freshness. |
| `tests/corpus/manifests/report_families.tsv` | No row change | Broad report-family support-tier and freshness-policy vocabulary already separates source-controlled, generated-local, hosted-ci, optional-data, and deferred-governance semantics. |
| `docs/planning/EPIC_16/SPRINT_182/artifacts/windows-report-freshness-deferral-decision.md` | Existing record retained | The record is the explicit Windows status authority while deferral is active. |

## Support-Tier Semantics

Windows CMake build/test and static install/downstream checks are reviewed
Windows evidence. They do not imply generated report freshness. While the
deferral is active:

- selected target rows must not list `windows` in `workflow_platforms`;
- hosted selected freshness claims remain limited to the Linux/macOS platforms
  listed in the selected-target manifest;
- Windows report freshness remains a non-claim for generated report artifacts;
- a future Windows promotion must add exact workflow file, job, artifact,
  platform, support-tier, claim-scope, and non-claim metadata.

## Documentation Updated

- `tests/corpus/README.md` now states that the selected-target manifest is
  positive selected-target authority, not a deferral registry.
- `tests/corpus/README.md` now links Windows report freshness deferral to the
  Sprint 182 decision record and states that Windows CMake/install validation
  does not imply generated report freshness.
- `tests/corpus/schemas/report_index_fields.md` now states that deferrals are
  not represented as fake selected rows and that future Windows promotion
  requires exact workflow metadata.

## Regression Coverage

Existing Day 7-8 tests now cover the Day 9 alignment:

- Windows workflow drift toward selected report freshness commands fails
  clearly.
- Windows workflow drift toward selected upload artifact names fails clearly.
- Selected target rows that list `windows` while the deferral is active fail
  clearly and name the offending row.
- The formal deferral record must preserve the required decision and blocker
  wording used by the guards.

## Day 10 Handoff

Day 10 should continue documentation alignment by checking the user-facing
support/installation surfaces against this manifest decision, especially where
Windows CMake/install claims appear near report freshness or generated report
claims.

## Validation

Planned Day 9 validation:

- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `git diff --check`

## Completion Criteria

- Manifest state matches the chosen Windows deferral decision.
- Support-tier wording is consistent across corpus and report-index docs.
- No accidental Windows selected row can pass silently while the deferral is
  active.
