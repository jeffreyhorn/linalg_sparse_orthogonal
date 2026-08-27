# Sprint 182 Day 7: Implementation Batch 1

## Purpose

Day 7 implements the first slice of the formal Windows report freshness
deferral path selected on Day 5 and designed on Day 6.

This batch adds the source-controlled deferral decision record and strengthens
workflow guard coverage so unsupported Windows report freshness interpretations
fail clearly.

## Implemented Changes

| Area | Change |
| --- | --- |
| Deferral record | Added `windows-report-freshness-deferral-decision.md` as the formal Sprint 182 Windows report freshness deferral record. |
| Workflow guard | Updated `tests/test_selected_comparison_workflow.py` to verify the deferral record and assert no selected manifest row lists `windows` as a selected freshness platform. |
| Drift tests | Added focused Windows drift tests for accidental selected freshness command and artifact strings. |
| Blocker diagnostics | Added a guard assertion that the formal deferral record names exact blocker text. |

## Guard Behavior

The updated guard keeps Windows fail-closed for:

- selected report freshness commands in `.github/workflows/windows-ci.yml`;
- selected report freshness artifact names in `.github/workflows/windows-ci.yml`;
- selected target manifest rows that list `windows` in `workflow_platforms`;
- missing blocker wording in the formal deferral record.

The guard still preserves exact Linux and macOS selected freshness lane checks.
It does not add a Windows selected freshness lane or allow a broad workflow
scan to stand in for manifest-backed evidence.

## Deferral Record Scope

The formal record states:

- Windows report freshness remains formally deferred.
- Windows CMake/MSVC build/test and static install/downstream claims remain
  supported.
- Selected oracle, comparison, benchmark, and broad generated report
  freshness remain unsupported on Windows.
- Future promotion requires Windows-safe CMake/MSVC probe support, exact
  Python executable proof, `.lib`/`.exe` handling, exact artifact uploads,
  manifest metadata, guard allowlist, and documentation alignment.

## Deferred To Day 8

Day 8 should complete the remaining implementation slice:

- decide whether additional manifest tests should assert no Windows selected
  platform outside the workflow guard;
- add or refine any formal deferral diagnostics needed by docs/report-index
  surfaces;
- preserve Linux/macOS selected report behavior while adding any remaining
  Windows deferral checks;
- update the Day 8 artifact with exact blocker diagnostics and residual
  implementation state.

## Validation

Day 7 changed planning artifacts and one Python guard test. Validation:

```sh
python3 -m py_compile tests/test_selected_comparison_workflow.py
python3 tests/test_selected_report_targets_manifest.py
python3 tests/test_selected_comparison_workflow.py
git diff --check
```

## Completion Criteria Review

| Criterion | Status | Evidence |
| --- | --- | --- |
| Selected path has executable guard coverage. | Complete | Workflow guard test now reads the deferral record and verifies no Windows selected freshness platform. |
| Workflow assertions remain scoped to exact jobs and upload blocks. | Complete | Existing Linux/macOS job and upload block assertions remain unchanged; Windows additions check forbidden selected strings. |
| Unsupported Windows interpretations fail clearly. | Complete | New drift tests cover accidental selected command, selected artifact, and missing blocker wording. |
