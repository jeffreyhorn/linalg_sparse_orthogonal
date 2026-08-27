# Sprint 182 Day 8: Implementation Batch 2

## Purpose

Day 8 completes the remaining implementation slice for the formal Windows
report freshness deferral path. Day 7 added the decision record and workflow
guard coverage; Day 8 adds selected-target manifest regression coverage and
records the residual implementation state for Day 9 alignment work.

## Implemented Changes

| Area | Change |
| --- | --- |
| Manifest regression coverage | Updated `tests/test_selected_report_targets_manifest.py` to assert selected target rows do not list `windows` while the Sprint 182 deferral record is active. |
| Drift diagnostics | Added a focused manifest drift regression that appends `windows` to a selected comparison row and requires a clear failure message naming the row. |
| Deferral record linkage | Manifest tests now read `windows-report-freshness-deferral-decision.md` and require the formal deferral statement before enforcing the Windows non-selection invariant. |
| Linux/macOS preservation | Existing selected manifest validation and workflow guard checks remain unchanged for current Linux/macOS lanes. |

## Completed Deferral Implementation State

The first implementation phase now has:

- a formal source-controlled Windows report freshness deferral record;
- workflow guard coverage for accidental Windows selected freshness commands
  and selected artifact names;
- workflow guard coverage that selected target rows do not list `windows`;
- selected-target manifest regression coverage for the same Windows
  non-selection invariant;
- focused drift diagnostics for accidental selected Windows platform metadata.

No Windows selected report freshness lane was added.

## Manifest Behavior

The selected target manifest remains positive selected-target authority. Day 8
does not add a fake Windows selected row or turn the manifest into a general
deferral registry.

The active invariant is:

- selected rows may list current selected platforms such as `linux` and
  `macos`;
- selected rows must not list `windows` while the formal Sprint 182 deferral
  record is active;
- a future Windows promotion must add exact workflow file, job, artifact,
  platform, support-tier, claim-scope, and non-claim metadata instead of
  drifting a current row silently.

## Preserved Behavior

The existing Linux/macOS selected report behavior remains unchanged:

- Linux selected oracle freshness is still manifest-backed.
- Linux selected comparison freshness is still manifest-backed.
- Linux selected benchmark freshness is still manifest-backed.
- macOS selected comparison freshness is still manifest-backed.
- Upload block checks still require exact artifacts and
  `if-no-files-found: error`.

## Deferred To Day 9

Day 9 should align manifest/support-tier documentation and decide whether any
additional schema or docs wording is needed for the Windows deferral. The
implementation should preserve the Day 8 invariant unless Day 9 introduces a
more explicit non-selected Windows status representation.

## Validation

Day 8 changed planning artifacts and one Python manifest test. Validation:

```sh
python3 -m py_compile tests/test_selected_report_targets_manifest.py tests/test_selected_comparison_workflow.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
git diff --check
```

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Implementation no longer has known incomplete branches. | Complete | Deferral record, workflow guard, and manifest regression coverage are in place. |
| Tests cover expected success or deferral failure behavior. | Complete | Manifest success and drift tests cover Windows non-selection while deferral is active. |
| Existing selected report lanes remain unchanged unless explicitly justified. | Complete | Day 8 does not alter selected manifest rows or Linux/macOS workflow behavior. |
