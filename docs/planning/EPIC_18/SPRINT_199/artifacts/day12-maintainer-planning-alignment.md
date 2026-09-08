# Sprint 199 Day 12 Maintainer and Planning Alignment

## Purpose

Day 12 aligns maintainer guidance and Epic 18 planning status with the Sprint
199 selected Windows Cholesky disposition. The sprint has reviewed hosted
Windows evidence for `cholesky-spd-tridiag-5`, hardened selected-target gates,
and calibrated public docs, but selected Windows freshness promotion remains
re-deferred.

## Maintainer Guidance Updates

`docs/maintainer_guide.md` now records that:

- Sprint 190 created one bounded Windows hosted workflow path for
  `cholesky-spd-tridiag-5`.
- Sprint 199 reviewed hosted evidence for that exact path.
- the path remains guarded workflow evidence, not promoted selected Windows
  freshness.
- promotion remains blocked until selected metadata, generated support tier,
  generated non-claim wording, and the claim contract are promoted together.
- local unavailable PowerShell remains environment residual evidence, not pass
  evidence.

The selected target manifest remains the authority for selected target
platforms, expected rows, artifacts, support tiers, freshness policies, claim
scopes, and non-claims.

## Planning Status Update

`docs/planning/EPIC_18/PROJECT_PLAN.md` no longer lists Sprint 199 as pending
future execution. The status row now marks Sprint 199 as in progress with
Windows promotion re-deferred and cites Day 1-Day 12 artifacts.

The status row does not close Sprint 199 and does not promote Windows selected
freshness. Day 13 still owns integrated validation, and Day 14 still owns
closeout review.

## Sprint Item Status

| Item | Day 12 status |
| --- | --- |
| 199.1 Hosted Evidence Review | Complete for evidence semantics: hosted run `34269219871` proves one exact selected Windows MSVC/CMake Cholesky workflow path. |
| 199.2 Manifest Promotion Decision | Re-deferred: `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` still omits `windows` and remains `local_only`. |
| 199.3 Normalizer Hardening | Complete for selected-target path and diagnostic hardening. |
| 199.4 Workflow Guard Update | Complete for current guarded workflow ownership. |
| 199.5 Documentation Calibration | Complete for re-deferred disposition: README, INSTALL, corpus docs, maintainer guide, and Epic planning status now match the re-deferred disposition. |
| 199.6 Validation | In progress: Day 10-Day 12 focused gates pass; final integrated validation remains for Day 13. |

## Residual Queue

| Residual | Owner | Required future action |
| --- | --- | --- |
| Selected Windows Cholesky freshness promotion | selected target manifest, generator metadata, normalizer, docs | Promote `workflow_platforms`, support tier, generated summary/non-claim wording, and claim contract together after final evidence review. |
| Broad Windows report freshness | future Windows report freshness sprint | Add separate Windows-safe generation paths, selected upload scopes, manifest metadata, and guards; do not infer from Cholesky path. |
| QR incompatible Windows comparison freshness | future target-specific comparison work | Prove its MSVC project probe and promote selected-target metadata separately. |
| Windows selected oracle freshness | future oracle workflow work | Add selected oracle generation/upload/freshness path and manifest metadata. |
| Windows selected benchmark freshness | future benchmark workflow work | Add selected benchmark methodology, hosted lane, artifact, and manifest metadata. |
| Local PowerShell unavailable checks | maintainer environment | Record exit `2` as unavailable local evidence; hosted `--require-pwsh` remains pass/fail ownership. |

## Validation Results

Day 12 maintainer/planning edits were validated with:

| Command | Result | Notes |
| --- | --- | --- |
| `make docs-check` | Passed | Doxygen generation and API docs coverage completed. |
| `python3 tests/test_validate_windows_powershell.py` | Passed | Claim-boundary markers, selected workflow guards, local unavailable PowerShell behavior, and hosted-required fail-closed behavior all passed. |
| `python3 tests/test_selected_report_targets_manifest.py` | Passed | Manifest invariants still agree with Windows re-deferral. |
| `python3 tests/test_selected_comparison_workflow.py` | Passed | Workflow contract remains selected-target scoped. |
| `git diff --check` | Passed | No whitespace errors. |

No `.c` or `.h` files were edited for Day 12.
