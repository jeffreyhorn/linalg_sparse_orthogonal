# Sprint 216 Day 1: Epic Evidence Intake

## Purpose

Establish the merged evidence sources for Epic 19 before recalibrating claims
or assigning final item statuses. This is an intake, not a fresh validation or
a state-of-the-art decision. The [working-notes ledger](../WORKING_NOTES.md)
maps each item 207.1-216.6 to an owner role, evidence pointer, and provisional
acceptance check. The [sprint plan](../PLAN.md) schedules reconciliation and
final validation for later days.

## Source Inventory

The [project plan](../../PROJECT_PLAN.md),
[Epic review](../../reviews/review-codex-2026-09-20.md), and
[Epic todo](../../reviews/todo-codex-2026-09-20.md) define the intended
closure tracks. Each Sprint 207-215 directory has a plan, working notes,
retrospective, daily artifacts, decision record, validation record, and
closeout review. Git merge history on `master` records PRs #230-#238 in
sprint order, ending with #238 at merge commit `c5f30490`. This establishes
that the sprint branches landed, not that their hosted or optional workflows
were re-executed for Day 1.

| Sprint | Merged evidence and narrow observed outcome | What it does not prove |
| --- | --- | --- |
| 207 | [Retrospective](../../SPRINT_207/RETROSPECTIVE.md), [decision](../../SPRINT_207/artifacts/day5-provider-decision.md), [validation](../../SPRINT_207/artifacts/day12-integrated-validation.md), [closeout](../../SPRINT_207/artifacts/day14-closeout-review.md), PR #230: continued provider deferral with local developer-mode Homebrew static proof and guards. | Public tap, Homebrew/core, bottles or broad package distribution. |
| 208 | [Retrospective](../../SPRINT_208/RETROSPECTIVE.md), [decision](../../SPRINT_208/artifacts/day5-promotion-decision.md), [validation](../../SPRINT_208/artifacts/day13-integrated-validation.md), [closeout](../../SPRINT_208/artifacts/day14-closeout-review.md), PR #231: hosted Windows Cholesky artifact inspected; selected freshness re-deferred because generated metadata conflicts. | Selected Windows Cholesky freshness or broad Windows parity. |
| 209 | [Retrospective](../../SPRINT_209/RETROSPECTIVE.md), [decision](../../SPRINT_209/artifacts/day8-manifest-decision.md), [validation](../../SPRINT_209/artifacts/day13-integrated-validation.md), [closeout](../../SPRINT_209/artifacts/day14-closeout-review.md), PR #232: bounded Windows/MSVC QR collection lane; selected freshness re-deferred without inspected hosted QR artifact. | A hosted QR result or selected Windows QR freshness. |
| 210 | [Retrospective](../../SPRINT_210/RETROSPECTIVE.md), [selection](../../SPRINT_210/artifacts/day2-owner-ranking.md), [validation](../../SPRINT_210/artifacts/day12-integrated-validation.md), [closeout](../../SPRINT_210/artifacts/day14-closeout-review.md), PR #233: selected no-reorder linked-list LDLT deterministic allocation-failure proof. | Broad allocator, solver-family or platform reliability. |
| 211 | [Retrospective](../../SPRINT_211/RETROSPECTIVE.md), [selection](../../SPRINT_211/artifacts/day2-candidate-ranking.md), [validation](../../SPRINT_211/artifacts/day12-integrated-validation.md), [closeout](../../SPRINT_211/artifacts/day14-closeout-review.md), PR #234: selected LDLT CSC native-parity test helper extraction with ownership guards. | New solver behavior or product capability. |
| 212 | [Retrospective](../../SPRINT_212/RETROSPECTIVE.md), [decision](../../SPRINT_212/artifacts/day5-product-policy-decision.md), [validation](../../SPRINT_212/artifacts/day12-integrated-validation.md), [closeout](../../SPRINT_212/artifacts/day14-closeout-review.md), PR #235: selected canonical benchmark remains threshold-free with stronger freshness guards. | Hosted timing threshold, speed superiority or platform timing parity. |
| 213 | [Retrospective](../../SPRINT_213/RETROSPECTIVE.md), [decision](../../SPRINT_213/artifacts/day5-product-policy-decision.md), [validation](../../SPRINT_213/artifacts/day13-integrated-validation.md), [closeout](../../SPRINT_213/artifacts/day14-closeout-review.md), PR #236: generated API HTML remains local-only with stronger workflow/routing guards. | Hosted, retained, or committed generated API publication. |
| 214 | [Retrospective](../../SPRINT_214/RETROSPECTIVE.md), [decision](../../SPRINT_214/artifacts/day2-abi-release-scope-decision.md), [validation](../../SPRINT_214/artifacts/day13-integrated-validation.md), [closeout](../../SPRINT_214/artifacts/day14-closeout-review.md), [residual queue](../../SPRINT_214/RESIDUAL_QUEUE.md), PR #237: static-first deferral and ABI/release design with local installed static-consumer proof. | Shared artifact, dynamic ABI, hosted candidate or release approval. |
| 215 | [Retrospective](../../SPRINT_215/RETROSPECTIVE.md), [blocked pilot](../../SPRINT_215/artifacts/day10-selected-pilot-outcome.md), [validation](../../SPRINT_215/artifacts/day13-integrated-validation.md), [closeout](../../SPRINT_215/artifacts/day14-closeout-review.md), [residual queue](../../SPRINT_215/RESIDUAL_QUEUE.md), PR #238: pinned external blueprint and fail-closed pilot preflight. | Accepted external numerical result, timing, platform parity or state-of-the-art status. |

The source index is a map, not an endorsement of every historical check as
current. Later days must inspect raw logs, changed surfaces, and hosted run
identities before closing new claims.

## Intake Gaps And Contradictions

1. The [project-plan status snapshot](../../PROJECT_PLAN.md) still states
   that Sprints 215-216 are pending and has a combined `215-216` pending row.
   Sprint 215 is merged as PR #238; Day 6-7 must correct this stale planning
   statement with item-level evidence. Sprint 216 remains in progress.
2. Sprint 215 designed, but did not implement, the isolated external v1
   schema/validator. Its opt-in helper compiled, but the selected pilot ran
   neither solver and produced no accepted result. Treat this as a residual,
   not a successful external comparison.
3. Sprint 208 has a hosted Windows Cholesky artifact but not internally
   consistent promotion metadata. Sprint 209 registered a Windows QR lane
   without an inspected hosted QR artifact at closeout. Both selected
   freshness promotions remain re-deferred.
4. Sprint 214's ABI policy and release checklist are designs. Local installed
   static consumers do not establish a shared-library build, dynamic ABI,
   candidate-hosted result, or approved release.
5. The review/todo and nine sprint retrospectives contain overlapping
   residuals. Day 8 must de-duplicate them with exact closure targets.
6. No fresh hosted evidence, external solver result, or end-to-end quality
   gate was run for this Day 1 documentation intake. Historical pass counts
   must not be reported as new Sprint 216 validation.

## Claim Boundaries

- **Local versus hosted:** A local test or proof may support the named local
  path only. A workflow definition is not evidence that its hosted run passed.
- **Selected versus broad:** Cholesky/QR selected targets, one LDLT owner, one
  benchmark row, and one helper cluster do not imply family-wide support.
- **Design versus execution:** ABI policy, release checklist, and external
  baseline blueprint define future acceptance but are not executed product
  proofs.
- **Blocked versus pass:** The Sprint 215 schema gate blocked the pilot before
  either solver ran; the provisional attempt is not an accepted v1 record.
- **Publication:** Generated Doxygen HTML is local-only; package and external
  results have separate provenance/publication gates.

## Day 1 Outcome And Handoff

All nine sprint evidence packages and their merge commits are located. The
item ledger covers all 54 Sprint 207-215 items and six Sprint 216 items with
owner roles and provisional checks. The project-plan snapshot discrepancy and
unearned claim boundaries are recorded, not silently corrected today.

Day 2 should compare each closeout statement with its decision and validation
record, then make a discrepancy table with exact evidence. Day 3 should freeze
the reconciled map and inventory claim wording across all named public and
planning surfaces.

## Day 1 Validation

- `make docs-check`: passed locally; Doxygen generated local ignored HTML and
  the API docs coverage check found 18 public headers and matching pages.
- Local Markdown link existence check for the two new Day 1 records: passed.
- Trailing-whitespace check for the Sprint 216 plan and Day 1 records: passed.
- Ledger count: 60 item rows, covering 207.1-216.6.
- No `.c` or `.h` file changed, so the full C quality gate was not required.
  These documentation checks do not constitute product or hosted validation.
