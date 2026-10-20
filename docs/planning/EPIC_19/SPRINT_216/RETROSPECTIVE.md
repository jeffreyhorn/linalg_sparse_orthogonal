# Sprint 216 Retrospective

**Sprint:** 216 - Epic 19 Final Validation, Claim Calibration & Closeout

**Duration:** 14 planned days on branch `sprint-216`

**Status:** Six selected deliverables closed locally; review and merge pending

## Source And Definition Of Done

The [project plan](../PROJECT_PLAN.md) defines six items totaling 166
estimated hours. The [sprint plan](PLAN.md), [working notes](WORKING_NOTES.md),
[Day 14 closeout](artifacts/day14-integrated-validation-closeout.md),
[Epic 19 retrospective](../EPIC_19_RETROSPECTIVE.md), and 14 daily artifacts
are the evidence trail. Estimates are not measured hours. `complete` means
the selected audit, documentation, status, validation, retrospective, or
queue deliverable, not promotion of the capability it considered.

- [x] Reconciled all nine merged Sprint 207-215 decisions, validation
      records, review chronology, and residuals; kept historical and
      current-tree evidence separate.
- [x] Calibrated README, INSTALL, maintainer, benchmark, API, corpus, schema,
      and planning claims; added focused external-claim mutation coverage.
- [x] Assigned exactly one evidence-backed selected-deliverable disposition
      to each of the 54 Sprint 207-215 items and all six Sprint 216 items.
- [x] Ran focused changed-surface local checks, including selected report,
      benchmark, API, corpus, pilot-synthetic, static-install, and docs gates;
      recorded the conditional C gate decision and unrun hosted lanes.
- [x] Published the Epic 19 retrospective with an explicit no-state-of-the-art
      assessment and final non-claim list.
- [x] Published 21 prioritized residual targets with owner role, dependency
      or revisit trigger, acceptance evidence, stop condition, and claim gate.
- [ ] Established a current hosted Windows promotion, accepted external
      two-sided result, controlled comparative timing, shared ABI, release,
      or state-of-the-art result. None was a selected Sprint 216 deliverable.

## What Went Well

1. **The item ledger distinguished work from capability.** Days 1-7 mapped
   the nine prior sprints to 54 selected-item dispositions. Re-deferred
   Windows freshness, threshold-free benchmark policy, local-only generated
   API, static-first shared-library deferral, and the blocked external pilot
   are complete *decisions or fallbacks*, not promoted product support.
2. **The claim audit reached user-facing surfaces.** Days 3-5 narrowed
   unqualified performance wording, corrected the Sprint 213 local-only API
   policy owner, made the blocked external pilot visible in INSTALL, and
   separated source-controlled comparison rows from accepted independent
   evidence. The focused guard tests cover these boundaries.
3. **Current-tree checks were tied to changed surfaces.** Day 10 passed 14
   local focused commands, Day 11 passed serial static/package checks, and
   Day 14 reran affected claim, API, corpus, manifest, comparison, benchmark,
   pilot-synthetic, and docs checks. The closeout records exact local scope
   instead of claiming a hosted rerun.
4. **The state-of-the-art answer is testable.** Days 9 and 13 recorded zero
   accepted external cells, no solver execution or timing, and no publication
   approval. The queue orders v1 validation, one two-sided correctness cell,
   and publication review before any bounded external statement; broad
   assessment has a separate representative evidence bar.
5. **Residuals remained actionable.** Days 8-9 deduplicated old handoffs
   into 21 acceptance targets without deleting a missing capability. Day 14
   kept those targets and closed the selected sprint items only after local
   validation and link/status checks.

## What Didn't Go Well

1. **Hosted evidence did not become current.** The Sprint 208 Windows
   Cholesky artifact is historical and still has a support-tier mismatch;
   Sprint 209's Windows QR lane lacks an inspected hosted result for
   promotion. Sprint 216 did not re-query exact hosted run/job/artifact
   currency. Selected Windows freshness remains re-deferred.
2. **The external pilot remains blocked upstream.** The v1 schema and
   validator are still absent. The Sprint 215 preflight did not run either
   solver; Day 14's 13 pilot tests were synthetic. A pinned CHOLMOD
   discovery identity is not runtime-loaded-library provenance or numerical
   correctness, and fixture/module publication rights remain unapproved.
3. **Local freshness is easy to overread.** Day 14 regenerated 40 passing
   comparison rows, but every row uses a source-controlled dense helper and
   `local_only`; optional NumPy/SciPy dependencies defer. The canonical
   benchmark still has one repeat and `threshold=n/a`. Neither result is
   independent-package parity, portable speed, or a timing gate.
4. **Build-state cleanup invalidated generated snapshots.** Day 11's Make
   install test invoked `make clean` and removed Day 10 ignored reports.
   Day 14 regenerated them before freshness inspection. Future integrated
   runs should continue to order install checks and generated-report checks
   explicitly rather than reusing stale digests.
5. **Some guards are expensive to rerun.** The full local API-docs freshness
   fixture suite took several minutes on Day 14. Its result was retained,
   while subsequent planning-only edits received the affected claim/docs and
   link reruns. This is a workflow cost, not a reason to omit the gate or
   reinterpret it as hosted publication.

## Final Metrics And Claim

| Metric | Sprint 216 close state |
| --- | --- |
| Project-plan selected items | 54/54 prior items and 6/6 Sprint 216 items have exactly one `complete` disposition; no capability promotion inferred |
| Planned effort | 166 h; actual hours not tracked |
| Daily evidence artifacts | 14 day-numbered Markdown artifacts |
| Prioritized residuals | 21 unique entries with five required closure fields |
| Day 10 focused checks | 14 commands exited 0 locally |
| Day 11 Make/CMake install consumers | 26/26 and 30/30 passed locally |
| Day 14 generated comparison | 40/40 `pass` rows across six local bundles, all source-controlled-helper and `local_only`; NumPy/SciPy defer six each |
| Day 14 canonical benchmark | Selected freshness passed; `--repeat 1`, `threshold=n/a`, no variance |
| Day 14 pilot synthetic tests | 13 passed; actual external solver runs and accepted v1 cells: 0 |
| Day 14 generated API coverage | 18 public headers, 18 reference pages, 18 source pages; HTML local-only |
| C/header files changed | 0; conditional `make format && make lint && make test` not triggered |
| Current hosted external/platform or release evidence accepted | 0 in this closeout |

The earned result is an Epic 19 selected-scope evidence and claim closeout
on this branch. It is **not** public package distribution, promoted selected
Windows freshness, broad allocation-failure reliability, a timing threshold,
hosted generated HTML, shared-library ABI compatibility, release readiness,
independent external-library parity, or state-of-the-art performance. The
[Epic retrospective](../EPIC_19_RETROSPECTIVE.md) explains the capability
boundaries; the [residual queue](RESIDUAL_QUEUE.md) owns future gates.

## Daily Evidence

- [Day 1 source intake](artifacts/day1-epic-evidence-intake.md)
- [Day 2 outcome reconciliation](artifacts/day2-outcome-validation-reconciliation.md)
- [Day 3 claim inventory](artifacts/day3-claim-inventory.md)
- [Day 4 user/maintainer calibration](artifacts/day4-user-maintainer-claims.md)
- [Day 5 benchmark/API/corpus matrix](artifacts/day5-benchmark-api-corpus-claims.md)
- [Day 6 prior-item status, 207-211](artifacts/day6-project-plan-status-207-211.md)
- [Day 7 prior-item status, 212-215](artifacts/day7-project-plan-status-212-215.md)
- [Day 8 residual crosswalk](artifacts/day8-residual-queue-intake.md)
- [Day 9 queue decision and validation matrix](artifacts/day9-residual-decision-validation-matrix.md)
- [Day 10 focused validation](artifacts/day10-focused-validation.md)
- [Day 11 static/package gates and outline](artifacts/day11-quality-gate-and-retro-outline.md)
- [Day 12 Epic outcomes draft](artifacts/day12-epic-outcomes-draft.md)
- [Day 13 state-of-the-art audit](artifacts/day13-retrospective-review.md)
- [Day 14 integrated closeout](artifacts/day14-integrated-validation-closeout.md)

## Residual Handoff

The highest-priority next steps are RQ-01 (implement and negatively test the
external v1 validator), RQ-02 (record one actual, provenance-complete
two-sided correctness cell), and RQ-03 (approve exact attribution, rights,
retention, claim and rollback). One accepted cell would justify only a
bounded statement about its exact inputs, builds and platform. RQ-04/05 keep
selected Windows promotion separate; RQ-06/13 distinguish a static source
candidate from shared ABI work. RQ-11/12/17/18 define the wider evidence
program required even to reconsider state-of-the-art wording. None of these
targets is closed by this retrospective.
