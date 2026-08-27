# Sprint 182 Day 13: Decision Reconciliation

## Purpose

Reconcile Sprint 182 implementation, documentation, manifests, guards,
validation records, and residual Windows risks before closeout.

## Final Decision State

Windows report freshness remains formally deferred for Sprint 182. The
repository now has a single consistent interpretation:

- Windows CI proves the reviewed CMake/MSVC build/test path and the
  static-first CMake install/downstream path.
- Windows CI does not prove selected oracle, comparison, benchmark, or broad
  generated report freshness.
- `tests/corpus/manifests/selected_report_targets.tsv` remains positive
  selected-target authority and does not contain a Windows selected freshness
  platform.
- The formal deferral record owns the Windows report freshness decision until a
  future manifest-backed promotion changes it.

## Project-Plan Reconciliation

| Item | Status | Evidence |
| --- | --- | --- |
| 182.1 Windows Report Audit | Complete | Days 1-4 artifacts audit inherited selected-target authority, Windows workflow/toolchain constraints, report command compatibility, and artifact/data semantics. |
| 182.2 Candidate Selection | Complete | Days 5-6 artifacts select formal deferral and design the decision-record, guard, documentation, and future-promotion contract. |
| 182.3 CI or Deferral Implementation | Complete | Days 7-8 add the formal deferral record and guard coverage for Windows workflow and selected manifest drift. |
| 182.4 Manifest Integration | Complete | Day 9 documents that selected manifests remain positive authority and must not gain fake Windows deferral rows. |
| 182.5 Documentation Alignment | Complete | Day 10 updates README, maintainer guide, and Windows workflow comments; Day 11 hardens failure diagnostics; Day 13 reconciles the resulting surfaces. |
| 182.6 Validation | In progress | Day 12 runs the feasible local sweep. Day 14 should repeat final checks for closeout. |

## Surface Reconciliation

| Surface | Reconciled state |
| --- | --- |
| Formal decision record | `windows-report-freshness-deferral-decision.md` states Windows report freshness remains formally deferred and lists exact blockers. |
| Selected target manifest | No selected row lists `windows` in `workflow_platforms`; no fake deferral row was added. |
| Windows workflow | `.github/workflows/windows-ci.yml` remains CMake/install scoped and explicitly excludes generated report freshness. |
| README | User-facing CI/report-index sections cite the Sprint 182 deferral and future promotion prerequisites. |
| Maintainer guide | Maintainer guidance keeps Windows out of selected `workflow_platforms` while deferral is active and requires generation, upload, manifest, and guard changes for promotion. |
| Corpus docs | `tests/corpus/README.md` and report-index schema docs describe selected manifests as positive selected-target authority, not a deferral registry. |
| Guard tests | Workflow and manifest tests reject accidental Windows selected commands, selected upload artifacts, and selected manifest platform promotion. |
| Validation records | Day 12 records feasible local validation and the local `pwsh` availability gap. |

## Retained Non-Claims

Sprint 182 intentionally retains these unsupported interpretations:

- selected Windows oracle freshness;
- selected Windows comparison freshness;
- selected Windows benchmark freshness;
- broad Windows generated report freshness;
- Windows Makefile parity;
- Windows Bash/POSIX report generation support;
- Windows `pkg-config` execution parity;
- package-manager support;
- shared-library, dynamic ABI, DLL/import-library, and runtime-loader support;
- broad Windows platform parity;
- portable performance, performance superiority, release readiness, or
  state-of-the-art status from generated reports.

## Residual Risks

| Risk | Current mitigation | Handoff |
| --- | --- | --- |
| Windows-safe selected comparison generation is still unimplemented. | Formal deferral plus workflow and manifest guards. | Build a CMake/MSVC project probe path before any Windows selected comparison promotion. |
| Local PowerShell parsing is unavailable in this environment. | Workflow text guards validate job and deferral contract locally. | Rely on GitHub Actions `windows-2022` for hosted PowerShell execution until local `pwsh` is available. |
| Oracle artifacts are stale locally against current `HEAD`. | Non-required freshness diagnostics remain warnings in Day 12 validation. | Required oracle freshness remains owned by existing Linux selected freshness commands. |
| Future docs could imply Windows report freshness without manifest promotion. | README, maintainer guide, corpus docs, workflow comments, and guards all cite the deferral. | Keep future promotion changes atomic across generator, workflow, manifest, tests, and docs. |

## Retrospective Inputs

- The strongest Sprint 182 outcome is a clear product decision: defer Windows
  report freshness while guarding against accidental promotion.
- The selected comparison path remains the best future promotion candidate
  because its artifact shape is favorable, but it needs Windows-safe build/link
  probe work.
- Treat selected manifests as positive authority only; do not represent
  deferrals as fake selected rows.
- Guard diagnostics should continue naming exact workflow jobs, artifact names,
  required files, manifest rows, and deferral blocker text.

## Sprint 183 Handoff

If Sprint 183 revisits Windows report freshness, start with selected comparison
and require all of these before promotion:

- Windows-safe direct generator invocation that avoids Makefile and Bash
  assumptions;
- CMake/MSVC project probe build/link support with `.lib` handling and no Unix
  `-lm`;
- `.exe`-aware temporary probe execution;
- exact Python executable proof on `windows-2022`;
- exact selected upload scope with `if-no-files-found: error`;
- selected target manifest metadata for workflow file, job, artifact, platform,
  support tier, claim scope, and non-claims;
- guard allowlist updates and public/maintainer documentation updates.

## Day 14 Closeout Inputs

Day 14 should repeat the focused validation suite, confirm no C files were
modified unless the final diff changes, verify `pwsh` remains the only local
validation gap if still unavailable, and prepare the sprint closeout artifact.

## Completion Criteria

- Windows report freshness decision is internally consistent.
- Docs, guards, and manifests agree on support tier and claim scope.
- Remaining risks are explicit enough for Sprint 183 planning.
