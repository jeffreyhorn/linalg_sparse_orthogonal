# Sprint 216 Day 8: Residual Queue Intake

## Intake Rule

The [Day 8 draft queue](../RESIDUAL_QUEUE.md) has 21 ranked closure targets.
The [Day 6](day6-project-plan-status-207-211.md) and
[Day 7](day7-project-plan-status-212-215.md) ledgers mark selected sprint
deliverables complete, not their larger product capabilities. This intake
deduplicates by **acceptance gate**. It does not erase an unearned claim merely
because a decision, checklist, workflow, or blueprint was delivered. Epic 19
review/todo themes are mapped below; Day 9 must finalize this first order.

## Sprint Handoff Crosswalk

| Source | What closed in Sprint 207-215 | What remains and draft queue target |
| --- | --- | --- |
| [207 retrospective](../../SPRINT_207/RETROSPECTIVE.md#residuals) | Provider tier decision and local developer-mode static Homebrew proof/guards. | Public source provider **RQ-07**; Homebrew/core, bottles/Linuxbrew, other recipes and binary packages **RQ-16**; shared package ABI **RQ-13**. Public distribution is not implied by the local proof. |
| [208 retrospective](../../SPRINT_208/RETROSPECTIVE.md#residuals) | Six-row hosted Windows Cholesky evidence reviewed, with selected freshness re-deferred. | Generated support-tier/manifest alignment **RQ-04**; broader Windows freshness/Make/`pkg-config`/benchmark **RQ-15**; QR handoff merged with 209 at **RQ-05**. |
| [209 retrospective](../../SPRINT_209/RETROSPECTIVE.md#residuals) | Windows/MSVC QR collection workflow and guards, without an inspectable hosted QR artifact. | Exact hosted QR proof/claim alignment **RQ-05**; broad Windows **RQ-15**; broad QR/least-squares parity needs external cells/oracles **RQ-11/RQ-18**. |
| [210 retrospective](../../SPRINT_210/RETROSPECTIVE.md#residuals) | No-reorder linked-list LDLT 25-site deterministic owner proof. | One next CSC/reordered/Cholesky/matrix/QR/SVD/eigensolver allocation owner **RQ-09**. Broad reliability remains a series of owner proofs, not one inherited claim. |
| [211 retrospective](../../SPRINT_211/RETROSPECTIVE.md#residuals) | One LDLT CSC native-parity test extraction and guards. | Next explicit large-test/implementation cluster **RQ-10**; shared comment-stripping/parser maintenance **RQ-19**. Local full CMake CTest execution later passed 60/60 in [Sprint 214 Day 13](../../SPRINT_214/artifacts/day13-integrated-validation.md), so that *local* check is not requeued as absent; candidate-hosted CMake evidence belongs to **RQ-06**. |
| [212 retrospective](../../SPRINT_212/RETROSPECTIVE.md#residuals) | Selected threshold-free freshness policy and docs guards. | Timing gate **RQ-08**; portable/platform speed and optional timing **RQ-12/RQ-17**; Windows benchmark **RQ-15**; broad publication/release benchmark **RQ-21**. `--repeat 1` freshness cannot satisfy any timing target. |
| [213 retrospective](../../SPRINT_213/RETROSPECTIVE.md#residuals) | Stronger local-only generated API policy, routing and staging guards. | Hosted, retained-artifact or committed HTML choice **RQ-14**; structured workflow parser **RQ-19**; broader API coverage/first-use route **RQ-20**. The three publication modes share one policy gate and are not three simultaneous commitments. |
| [214 residual queue](../../SPRINT_214/RESIDUAL_QUEUE.md) and [retrospective](../../SPRINT_214/RETROSPECTIVE.md) | ABI/release design, static-first guards and local installed static consumers. | Candidate-specific source release **RQ-06**; shared ABI/export/loader/consumer **RQ-13**; branch/candidate hosted proof attached to each claimed lane; public provider/API publication remain **RQ-07/RQ-14/RQ-16** rather than duplicated here. |
| [215 residual queue](../../SPRINT_215/RESIDUAL_QUEUE.md) and [retrospective](../../SPRINT_215/RETROSPECTIVE.md) | Pinned blueprint, optional helper and fail-closed `schema_validation_unavailable` attempt. | v1 validator **RQ-01**; two-sided correctness and runtime provenance **RQ-02**; fixture/module/license/publication review **RQ-03**; matrices/second solver **RQ-11**; hosted/optional timing/claim review **RQ-12**. Neither solver ran in the blocked attempt. |

The [Sprint 214 queue](../../SPRINT_214/RESIDUAL_QUEUE.md) overlaps the
package, API and hosted-evidence terms of earlier sprints. It is retained as
evidence for RQ-06/RQ-13, not copied into a second generic distribution task.
The [Sprint 215 queue](../../SPRINT_215/RESIDUAL_QUEUE.md) is split in
dependency order so the absent v1 validator, actual solver result and public
approval cannot be promoted together by a single green synthetic test.

## Review And Todo Crosswalk

| [Review](../../reviews/review-codex-2026-09-20.md) gap / [todo](../../reviews/todo-codex-2026-09-20.md) track | Queue placement or completed narrow outcome |
| --- | --- |
| State-of-the-art and external baseline / track 9 | Blueprint closed; validation, one cell, approval, breadth, hosted/timing and claim review **RQ-01/02/03/11/12**. Broad numerical conformance **RQ-18**. |
| Package distribution / track 1 | Provider decision closed as local-only; public source tier **RQ-07**, other/binary providers **RQ-16**, candidate provenance **RQ-06**. |
| ABI/release / track 8 | Design/checklist closed; candidate release **RQ-06**, shared ABI/loader/compatibility **RQ-13**. |
| Windows selected Cholesky and QR / tracks 2-3 | Re-deferral decisions closed; selected promotions **RQ-04/05**, broad Windows/benchmark/Make/`pkg-config` **RQ-15**. |
| Allocation reliability / track 4 | One selected owner closed; next owner **RQ-09**; no broad reliability inference. |
| Review surface / track 5 | One cluster closed; next cluster **RQ-10**, parser/guard upkeep **RQ-19**. |
| Benchmark methodology / track 6 | Threshold-free policy closed; one threshold **RQ-08**, public/release reports **RQ-21**, comparative measurement **RQ-12/17**. |
| Generated API / track 7 | Local-only policy closed; selected publication **RQ-14**, API completeness and first-use **RQ-20**. |
| Storage/performance architecture, usability, documentation, oracle quality / review gaps 8-11 | Compressed-first measured kernel **RQ-17**; independent conformance/oracles **RQ-18**; guard tooling **RQ-19**; first-use/API coverage **RQ-20**. |
| Epic closeout / track 10 | Sprint 216 items 216.1-216.6 remain provisional in the [project plan](../../PROJECT_PLAN.md); Days 9-14 validation/retrospective/claim decision are **current-sprint work**, not silently requeued as completed next-epic work. |

## First Priority Judgment

1. **P0 RQ-01 to RQ-03:** correctness and publication integrity outrank a
   benchmark or support claim. RQ-01 is a hard pre-solve gate; RQ-02 then
   proves one exact cell; RQ-03 determines whether that cell may be public.
2. **P1 RQ-04 to RQ-10:** bounded fixtures and decisions already exist. A
   selected Windows result or static candidate has user impact, but requires
   exact hosted/claim alignment. Benchmark, allocation and review work remain
   one-row/one-owner/one-cluster candidates to control blast radius.
3. **P2 RQ-11 to RQ-21:** broader comparisons, shared ABI, public generated
   docs, packages, cross-platform behavior, compressed-kernel performance,
   conformance, and publication need new scope or dependencies. Their queue
   entries name revisit triggers, evidence and stop conditions; none is a
   standing promise of future support.

## Open Questions For Day 9

- Who approves the external v1 object and fixture/module publication terms?
  Until roles are assigned, RQ-01/03 are product/report decisions, not gates
  that an optional pilot can waive.
- Which first release candidate or public provider, if any, gets product
  approval? RQ-06/07 remain separate from the Sprint 214 checklist design.
- Should a future benchmark threshold and report publication be pursued in
  the same candidate? RQ-08 and RQ-21 remain separate because measurement
  policy does not itself approve retention or public routing.
- Which exact changed-surface checks and hosted artifacts must be rerun for
  the final Sprint 216 closeout? Day 9 owns the validation matrix; historical
  sprint passes and workflow registrations cannot answer it.

## Day 8 Validation

Day 8 edited Markdown only. The 21-ID uniqueness audit and per-entry
owner/dependency/acceptance/stop/claim-field audit passed. Local links,
`make docs-check`, `make support-docs-guard`, trailing-whitespace check, and
`git diff --check` passed locally. This validates the queue's structure and
documentation; it does not execute any queued capability gate.
