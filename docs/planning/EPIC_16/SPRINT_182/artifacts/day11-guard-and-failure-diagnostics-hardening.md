# Sprint 182 Day 11: Guard And Failure Diagnostics Hardening

## Purpose

Harden guard failure messages for the formal Windows report freshness deferral
path so maintainers can identify the workflow block, manifest row, deferral
record, artifact name, or required file that must be fixed.

## Hardened Diagnostics

| Failure mode | Day 11 coverage |
| --- | --- |
| Missing Windows workflow job | `tests/test_selected_comparison_workflow.py` now requires the reviewed `build-and-test` and `install-and-downstream` jobs and fails with the missing job name. |
| Missing Windows deferral wording | The Windows workflow contract now requires `Sprint 182 formally defers Windows report freshness` in `.github/workflows/windows-ci.yml`. |
| Accidental Windows selected command promotion | Existing drift coverage continues to fail on selected report freshness command names in the Windows workflow. |
| Accidental Windows selected upload artifact | Existing drift coverage continues to fail on selected report freshness artifact names in the Windows workflow. |
| Missing deferral blocker | Existing drift coverage continues to require blocker text in the formal deferral record. |
| Wrong selected upload artifact name | Existing drift coverage continues to name the missing selected artifact. |
| Missing `if-no-files-found: error` | Existing drift coverage continues to require fail-closed upload behavior. |
| Broad selected upload path | Existing drift coverage continues to reject broad selected comparison upload paths. |
| Missing required upload file | Day 11 adds drift coverage that removes `build/comparison/qr_minnorm/project_observations.tsv` and requires a diagnostic naming that exact path. |

## Stale Or Missing Generated Rows

Sprint 182 selected the formal deferral path rather than a Windows report
freshness promotion. Windows-specific stale or missing generated-row
diagnostics are therefore not introduced on Day 11. The existing Linux/macOS
selected gates still own stale and missing generated-row diagnostics through
their selected report freshness commands and normalized report-index checks.

## Claim Boundary

The hardened guards preserve these boundaries:

- Windows CI remains CMake build/test and static install/downstream scoped.
- Windows selected report generation commands remain absent.
- Windows selected report upload artifact names remain absent.
- Selected target manifest rows remain free of `windows` while the formal
  deferral is active.
- Documentation and workflow comments must keep the formal deferral visible.

## Validation

Planned Day 11 validation:

- `python3 -m py_compile tests/test_selected_comparison_workflow.py`
- `python3 scripts/validate_corpus_schema.py`
- `python3 tests/test_selected_report_targets_manifest.py`
- `python3 tests/test_selected_comparison_workflow.py`
- `git diff --check`

## Completion Criteria

- Failure modes direct maintainers to the exact file, job, artifact, or path to
  fix.
- Accidental Windows selected report freshness claims fail in guards.
- Missing generated-row diagnostics remain owned by existing selected
  Linux/macOS freshness paths because Windows freshness is formally deferred.
