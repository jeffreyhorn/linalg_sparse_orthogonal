# Day 4: Project Plan Status Design

**Sprint:** 206 - Epic 18 Final Validation, Claim Calibration & Closeout  
**Theme:** Design the final Epic 18 project-plan status update and evidence
index before changing the project plan.  
**Time estimate:** 12 hours  
**Branch:** `sprint-206`  
**Base commit:** `4d819093`

## Scope

Day 4 designs the final project-plan and current-status update. It does not
edit `PROJECT_PLAN.md`, `EPIC_18_RETROSPECTIVE.md`, or
`EPIC_18_RESIDUAL_QUEUE.md` yet. Implementation belongs to Day 7, Day 11, and
Day 12 after the public and maintainer claim recalibration passes.

## Final Status Vocabulary

| Status phrase | Use when | Do not use when |
| --- | --- | --- |
| Historical final-validation evidence | Describing `SPRINT_197` artifacts that implemented the requested final-validation path under a numbering caveat. | Describing the explicit `SPRINT_206` branch after current closeout artifacts exist. |
| Closed selected scope | A sprint completed the exact selected closure and retained broader non-claims. | The result proves broad package, platform, ABI, performance, release, or state-of-the-art support. |
| Closed re-deferral | A sprint completed review, local proof, guards, or docs, but deliberately did not promote support because required evidence was missing. | The target metadata or public docs are actually promoted. |
| Closed local-only policy | A policy decision was made and guarded as local-only output. | Hosted publication, retained artifacts, or committed generated output are claimed. |
| Closed documentation consolidation | Documentation routing, vocabulary, and guards were simplified without support promotion. | New platform/package/performance/API support was added. |
| In progress explicit Sprint 206 closeout | Sprint 206 artifacts exist but final retrospective, residual queue, validation, and closeout are not complete yet. | Day 14 validation and closeout are complete. |
| Residualized | Remaining work has owner surfaces, closure target, evidence expectation, validation commands, and non-claims. | The work is merely vaguely desirable or already fully closed. |
| Superseded | A later artifact deliberately replaces an earlier current-status artifact. | Historical sprint artifacts should remain as immutable evidence. |
| Unearned broad claim | A claim remains unsupported beyond selected evidence. | There is exact source-controlled or hosted evidence supporting that broad claim. |

## Proposed Sprint Status Rows

| Sprint | Final current-status wording | Evidence placement |
| --- | --- | --- |
| 197 | Historical final-validation branch evidence with numbering caveat. | Keep compact link to `SPRINT_197/RETROSPECTIVE.md` and Day 8/Day 14 artifacts; explain it is historical evidence, not current Sprint 206 closeout. |
| 198 | Closed with developer-mode local Homebrew static source proof. | Project plan gets compact artifact list; residual queue records broader package-manager distribution remains unclaimed. |
| 199 | Closed with selected Windows Cholesky promotion re-deferred. | Project plan keeps guarded workflow/re-deferral evidence; residual queue records promotion conditions. |
| 200 | Closed with selected symbolic LU allocation-failure proof. | Project plan and retrospective state selected owner only; residual queue can keep future owners as broader residual. |
| 201 | Closed for selected SVD helper review-surface reduction. | Project plan and retrospective state selected cluster only; residual queue can keep future clusters as broader residual. |
| 202 | Closed with macOS hosted selected benchmark freshness evidence. | Project plan keeps PR run/job/artifact/digest summary; benchmark docs remain detailed owner. |
| 203 | Closed with Windows QR incompatible promotion re-deferred. | Project plan and residual queue keep hosted Windows/MSVC proof as missing promotion condition. |
| 204 | Closed with stronger local-only generated API policy. | Project plan and retrospective state local-only guard closure; residual queue keeps future hosted/artifact/committed publication as optional future decision. |
| 205 | Closed with support matrix and adoption quick-reference consolidation. | Project plan and retrospective must both mark closed and link `SPRINT_205/RETROSPECTIVE.md`. |
| 206 | In progress until Day 14, then closed with final validation, claim calibration, retrospective, and residual queue. | Project plan row should move from `SPRINT_197`-only evidence to explicit `SPRINT_206` artifacts as they exist. |

## Sprint 206 Item Status Design

| Item | Current Day 4 target status | Evidence owner |
| --- | --- | --- |
| 206.1 Evidence Reconciliation | Complete after Day 2; keep open only for Day 13 consistency hardening. | `SPRINT_206/artifacts/day2-outcome-reconciliation.md` and Day 13 artifact. |
| 206.2 Claim Recalibration | Audit complete after Day 3; implementation pending Days 5-6. | Day 3 audit, Day 5 public claim update, Day 6 maintainer claim update. |
| 206.3 Project Plan Status | Design complete after Day 4; implementation pending Day 7. | Day 4 design and Day 7 status implementation artifact. |
| 206.4 Integrated Validation | Pending until Day 8-Day 10 validation records exist. | Day 8 validation scope, Day 9 focused validation, Day 10 broad gates. |
| 206.5 Epic Retrospective | Pending until Day 11 draft and Day 13-Day 14 review. | `EPIC_18_RETROSPECTIVE.md`, Day 11 artifact, Day 14 closeout. |
| 206.6 Residual Queue | Pending until Day 12 draft and Day 13-Day 14 review. | `EPIC_18_RESIDUAL_QUEUE.md`, Day 12 artifact, Day 14 closeout. |

## Evidence Placement Rules

| Surface | Evidence to include | Evidence to avoid |
| --- | --- | --- |
| `PROJECT_PLAN.md` current-status snapshot | One compact row per sprint with links to plan, working notes, retrospective, daily artifacts, and the narrow outcome/non-claim boundary. | Full validation logs, long residual rationale, or repeated public support caveats. |
| `EPIC_18_RETROSPECTIVE.md` | Outcome narrative, project-plan metrics, validation summary, changed-surface summary, earned claims, non-claims, residual summary, state-of-the-art assessment, handoff. | Exhaustive day-by-day artifact lists already indexed in the project plan. |
| `EPIC_18_RESIDUAL_QUEUE.md` | Future work with exact closure target, owner surfaces, expected evidence, validation commands, and retained non-claims. | Treating selected closures as still unstarted or treating residuals as support claims. |
| `SPRINT_206/WORKING_NOTES.md` | Day-by-day implementation notes, validation log pointers, open questions, and current branch-specific decisions. | Replacing current Epic-level docs as the final public handoff. |
| `SPRINT_197` artifacts | Historical requested-branch evidence and prior final-validation context. | Rewriting as if they were produced after Sprints 198-205 were merged. |

## Current-Status Edit Map

| File | Required later edit | Owner day |
| --- | --- | --- |
| `PROJECT_PLAN.md` | Rename or supersede the "Sprint 197 Day 8 Interim Status Snapshot" framing with a current Epic 18 closeout snapshot after Sprint 206 evidence exists; update Sprint 206 row to point to `SPRINT_206` plan, working notes, artifacts, retrospective inputs, final validation, Epic retrospective, and residual queue. | Day 7 |
| `PROJECT_PLAN.md` | Keep Sprint 197 as historical/numbering-caveat evidence, not as the current completed Sprint 206 branch. | Day 7 |
| `EPIC_18_RETROSPECTIVE.md` | Change Sprint 205 from pending to closed and update metrics so selected Sprint 198-205 closures and Sprint 206 in-progress/completed state are internally consistent. | Day 11 |
| `EPIC_18_RETROSPECTIVE.md` | Update non-claims so selected Sprint 200-205 closures are not described as absent, while broad residual claims remain unearned. | Day 11 |
| `EPIC_18_RESIDUAL_QUEUE.md` | Update preamble from Sprint 197-seeded closeout to current Sprint 206 explicit closeout. | Day 12 |
| `EPIC_18_RESIDUAL_QUEUE.md` | Update E18-RQ-001 and E18-RQ-002 to selected closure/re-deferral plus broader residual wording. | Day 12 |
| `EPIC_18_RESIDUAL_QUEUE.md` | Audit E18-RQ-003 through E18-RQ-008 for selected closure versus broader residual wording. | Day 12 |

## Consistency Rules

1. A selected closure may be `complete` only for the exact selected scope named
   by its sprint evidence.
2. A re-deferral may be `closed` only if it records why promotion was withheld
   and what evidence is required for future promotion.
3. A residual queue row must include owner surfaces, closure target, expected
   evidence, validation commands, and claim boundary.
4. Historical artifacts should not be edited to change their branch-time
   meaning; current aggregate docs should add newer context instead.
5. Sprint 206 should not claim final closeout complete until retrospective,
   residual queue, validation records, and Day 14 closeout are present.
6. Status rows must distinguish selected proof from broad package, platform,
   ABI, performance, release, hosted API, and state-of-the-art support.

## Validation And Hygiene

| Check | Day 4 result |
| --- | --- |
| `git diff --check` | Planned after Day 4 artifact creation. |
| C/header quality gate | Not required for Day 4; no `.c` or `.h` edits. |
| Generated-output status | No generated output intentionally created. |
| Current-status docs | Not edited on Day 4; this artifact is design only. |
| Guard scripts/workflows/manifests | Not edited on Day 4. |

## Completion Criteria Review

| Criterion | Result |
| --- | --- |
| Item 206.3 has an implementation-ready project-plan update plan. | Met. This artifact defines status wording, evidence placement, file-level edit map, and Sprint 206 item status design. |
| Status wording cannot overstate support beyond available evidence. | Met. The vocabulary separates selected closure, re-deferral, local-only policy, residualized work, and unearned broad claims. |
| Planning docs have one source of truth for current Epic 18 closeout status. | Met as a design: `PROJECT_PLAN.md` remains the compact status index, while retrospective and residual queue retain narrative and future-work ownership. |

## Day 4 Disposition

Day 4 is complete. Day 5 should begin public claim recalibration with the Day 3
audit and this status design as constraints, leaving project-plan status
implementation for Day 7.
