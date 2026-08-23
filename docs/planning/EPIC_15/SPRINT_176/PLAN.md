# Sprint 176 Plan: Allocation-Failure Evidence, Claim Recalibration & Epic Closeout

**Sprint Duration:** 14 days
**Goal:** Add one targeted allocation-failure proof, reconcile all Epic 15
claims, and close the epic with evidence-bound documentation. This sprint
implements the Sprint 176 section of
`docs/planning/EPIC_15/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references
`docs/planning/EPIC_12/PROJECT_PLAN.md`, but the active merged Sprint 176
project-plan section lives in `docs/planning/EPIC_15/PROJECT_PLAN.md` and has
the title "Sprint 176: Allocation-Failure Evidence, Claim Recalibration &
Epic Closeout".

**Starting Point:** Sprint 176 begins from:

- Epic 15 evidence ledger updates across Sprints 167-175;
- completed or explicitly deferred performance, ABI, package-manager, public
  header, generated API, comparison, platform freshness, and report-publication
  decisions;
- current allocator and cleanup-path behavior in solver and shared subsystems;
- retained non-claims for broad state-of-the-art status, broad package-manager
  distribution, shared-library ABI support, portable performance superiority,
  and broad platform parity;
- Sprint 175 cross-platform report freshness promotion and associated
  fail-closed workflow guards.

The sprint must:

- select exactly one allocation-heavy subsystem for deterministic
  allocation-failure proof;
- add or extend a failure harness that exercises cleanup and ownership paths;
- document cleanup invariants for the selected subsystem;
- update README, maintainer guidance, report indexes, evidence ledgers, and
  non-claim tables to match the final Epic 15 state;
- create the Epic 15 retrospective with earned claims, retained non-claims,
  residuals, and validation evidence;
- run required quality gates and record final validation status.

**End State:** Sprint 176 leaves behind:

- deterministic allocation-failure evidence for one selected subsystem;
- cleanup and ownership invariant documentation for that subsystem;
- recalibrated public claims and non-claims for all Epic 15 work;
- Epic 15 retrospective and final residual queue;
- Sprint 176 working notes, daily artifacts, and final validation record.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 176 project-plan estimate.

---

## Day 1: Sprint Intake And Closeout Scope

**Title:** Closeout Intake
**Theme:** Establish Sprint 176 scope, source references, artifact layout, and
final Epic 15 closeout boundaries
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 176 section of
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.
2. Record the prompt path/source-artifact mismatch in working notes.
3. Create Sprint 176 working notes and artifact directory structure.
4. Inventory Sprint 167-175 retrospectives and closeout artifacts.
5. Define closeout categories: allocation failure, claim recalibration,
   evidence ledger, documentation, validation, and residual queue.
6. Write the Day 1 closeout-intake artifact.

### Deliverables

- Sprint 176 working-notes baseline
- artifact directory structure
- source artifact note
- Epic 15 closeout category map
- Day 1 closeout-intake artifact

### Completion Criteria

- Sprint 176 scope is tied to the active Epic 15 project plan
- all closeout evidence categories are identified
- retained non-claims are visible before implementation begins

---

## Day 2: Allocation Surface Inventory

**Title:** Allocation Inventory
**Theme:** Inventory allocation-heavy solver and shared subsystems before
selecting one failure-proof target
**Time estimate:** 12 hours

### Tasks

1. Locate allocation, reallocation, workspace, and cleanup paths in solver and
   shared internal subsystems.
2. Inventory existing allocator wrappers, fault-injection hooks, and
   allocation-related tests.
3. Identify subsystems with meaningful cleanup risk and feasible deterministic
   failure points.
4. Classify candidates by user impact, claim risk, testability, and blast
   radius.
5. Record candidate exclusions and dependencies.
6. Write the Day 2 allocation-surface inventory artifact.

### Deliverables

- allocation-heavy subsystem inventory
- existing failure-test and allocator-hook map
- candidate risk/value matrix
- excluded-candidate rationale
- Day 2 allocation-inventory artifact

### Completion Criteria

- candidate subsystems are comparable
- existing test and injection capabilities are understood
- no subsystem is selected before failure-path feasibility is assessed

---

## Day 3: Subsystem Selection Decision

**Title:** Subsystem Selection
**Theme:** Choose the single subsystem for deterministic allocation-failure
proof and define success criteria
**Time estimate:** 12 hours

### Tasks

1. Review the Day 2 allocation candidate matrix.
2. Select one allocation-heavy subsystem for Sprint 176 proof.
3. Define exact APIs, setup paths, allocation points, cleanup paths, and
   expected error behavior in scope.
4. Define out-of-scope allocation paths and retained non-claims.
5. Draft pass/fail criteria for deterministic allocation-failure evidence.
6. Write the Day 3 subsystem-selection decision artifact.

### Deliverables

- selected subsystem decision
- selected API and cleanup-path scope
- deterministic failure-point target list
- out-of-scope and non-claim notes
- Day 3 subsystem-selection artifact

### Completion Criteria

- one subsystem is selected and justified
- expected failure behavior is testable
- scope does not imply broad allocation-failure coverage

---

## Day 4: Failure Harness Design

**Title:** Harness Design
**Theme:** Design deterministic allocation-failure tests and cleanup
observability for the selected subsystem
**Time estimate:** 12 hours

### Tasks

1. Review existing test conventions for the selected subsystem.
2. Identify how allocation failures can be injected deterministically.
3. Define setup and teardown routines that detect leaked or double-freed
   ownership state.
4. Design assertions for return codes, partial initialization cleanup, and
   caller-visible state.
5. Decide whether new helper APIs, internal hooks, or test-only fixtures are
   required.
6. Write the Day 4 failure-harness design artifact.

### Deliverables

- allocation-failure harness design
- failure-point enumeration
- cleanup-observability strategy
- test assertion checklist
- Day 4 harness-design artifact

### Completion Criteria

- failure injection is deterministic
- cleanup checks are explicit enough to implement
- design does not require unsupported production behavior

---

## Day 5: Failure Harness Implementation

**Title:** Harness Implementation
**Theme:** Implement the deterministic allocation-failure harness for the
selected subsystem
**Time estimate:** 12 hours

### Tasks

1. Add or extend test fixtures for the selected subsystem.
2. Wire deterministic allocation-failure controls into the test path.
3. Exercise selected setup, workspace, and partial-initialization failure
   points.
4. Assert expected errors and caller-visible state after each injected failure.
5. Keep implementation scoped to testability and selected subsystem behavior.
6. Write the Day 5 harness-implementation artifact.

### Deliverables

- implemented allocation-failure test harness
- selected failure-point coverage
- expected-error assertions
- implementation notes
- Day 5 harness-implementation artifact

### Completion Criteria

- tests compile or run under the local development path
- injected allocation failures are deterministic
- implementation remains limited to the selected subsystem

---

## Day 6: Cleanup Invariant Implementation

**Title:** Cleanup Invariants
**Theme:** Strengthen and verify ownership cleanup behavior exposed by the
failure harness
**Time estimate:** 12 hours

### Tasks

1. Run the Day 5 harness and inspect failing cleanup or ownership cases.
2. Fix selected-subsystem cleanup behavior if defects are found.
3. Add assertions for repeated cleanup, null cleanup, or partially initialized
   state when applicable.
4. Confirm successful paths are not changed by failure-path fixes.
5. Document any defects found and the exact cleanup invariant now enforced.
6. Write the Day 6 cleanup-invariants artifact.

### Deliverables

- cleanup-path fixes if required
- invariant assertions
- success-path regression notes
- defect and ownership summary
- Day 6 cleanup-invariants artifact

### Completion Criteria

- failure-path cleanup is verified for selected cases
- success-path behavior remains stable
- ownership invariants are ready for documentation

---

## Day 7: Allocation Failure Regression Gate

**Title:** Regression Gate
**Theme:** Integrate the selected allocation-failure proof into maintained
test or validation surfaces
**Time estimate:** 12 hours

### Tasks

1. Decide the maintained test target or script that should own the new proof.
2. Wire the new allocation-failure test into Make and CMake surfaces as
   appropriate.
3. Update any test-count or platform inventory guards affected by the new
   coverage.
4. Run focused validation for the selected subsystem and test-registration
   surface.
5. Record platform limitations or staged exclusions if any remain.
6. Write the Day 7 regression-gate artifact.

### Deliverables

- maintained regression-gate integration
- test registration updates
- focused validation output
- platform limitation notes
- Day 7 regression-gate artifact

### Completion Criteria

- allocation-failure proof is reachable from maintained validation
- platform/test inventory remains coherent
- exclusions are explicit rather than accidental

---

## Day 8: Cleanup Invariant Documentation

**Title:** Invariant Docs
**Theme:** Document selected-subsystem ownership and cleanup invariants without
overstating allocation-failure coverage
**Time estimate:** 12 hours

### Tasks

1. Draft selected-subsystem cleanup and ownership invariant documentation.
2. Identify where the invariant belongs: header comments, maintainer guide,
   corpus notes, or planning artifact.
3. Update documentation with exact supported behavior and failure-scope
   limits.
4. Add non-claims for untested allocation paths and other subsystems.
5. Cross-reference the regression test or validation command.
6. Write the Day 8 invariant-documentation artifact.

### Deliverables

- cleanup invariant documentation
- selected-subsystem failure-scope statement
- allocation-failure non-claim language
- validation command reference
- Day 8 invariant-docs artifact

### Completion Criteria

- documentation matches implemented evidence
- unsupported allocation-failure coverage is not implied
- maintainers can locate the proof and invariant

---

## Day 9: Claim Surface Inventory

**Title:** Claim Inventory
**Theme:** Inventory all public and planning claim surfaces before final Epic
15 recalibration
**Time estimate:** 12 hours

### Tasks

1. Search README, maintainer guide, benchmark docs, corpus docs, package docs,
   and planning docs for Epic 15 claim language.
2. Map claims to evidence categories from Sprints 167-176.
3. Identify stale, unsupported, ambiguous, or overbroad wording.
4. Separate earned claims from local-only, hosted-only, advisory, supplemental,
   and unsupported claims.
5. Build a claim-recalibration checklist for Day 10.
6. Write the Day 9 claim-surface inventory artifact.

### Deliverables

- public claim surface inventory
- evidence-to-claim map
- stale and unsupported wording list
- recalibration checklist
- Day 9 claim-inventory artifact

### Completion Criteria

- every major public claim surface has been reviewed
- unsupported claims are listed before editing
- local-only and hosted evidence are not conflated

---

## Day 10: Claim Recalibration Implementation

**Title:** Claim Recalibration
**Theme:** Update public documentation, indexes, non-claim tables, and evidence
ledger to match completed Epic 15 evidence
**Time estimate:** 12 hours

### Tasks

1. Apply claim wording updates identified on Day 9.
2. Update evidence ledger entries for allocation-failure proof and all Epic 15
   completed decisions.
3. Update README and maintainer-facing docs with precise support tiers.
4. Preserve non-claims for broad package-manager, ABI, performance, platform,
   external-parity, release, and state-of-the-art assertions.
5. Add or update freshness/claim guard tests if documentation changes require
   mechanical enforcement.
6. Write the Day 10 claim-recalibration artifact.

### Deliverables

- updated public claim documentation
- updated evidence ledger
- updated non-claim tables or wording
- guard updates if required
- Day 10 claim-recalibration artifact

### Completion Criteria

- public claims match available evidence
- retained non-claims remain explicit
- claim changes are mechanically guarded where feasible

---

## Day 11: Epic 15 Retrospective Draft

**Title:** Retrospective Draft
**Theme:** Draft the Epic 15 retrospective from sprint artifacts, evidence,
earned claims, retained non-claims, and residuals
**Time estimate:** 12 hours

### Tasks

1. Review Sprint 167-176 working notes, artifacts, and retrospectives.
2. Summarize completed Epic 15 objectives and evidence links.
3. Summarize narrowed or deferred objectives with reasons.
4. Draft earned claims and retained non-claims.
5. Draft residual queue and next-epic candidates.
6. Write the Day 11 retrospective-draft artifact.

### Deliverables

- Epic 15 retrospective draft structure
- completed objective summary
- earned-claim and non-claim draft
- residual queue draft
- Day 11 retrospective-draft artifact

### Completion Criteria

- retrospective draft is source-backed by sprint artifacts
- residuals are explicit and actionable
- earned claims are separated from aspirations

---

## Day 12: Integrated Validation

**Title:** Final Validation
**Theme:** Run required quality gates, focused allocation-failure checks, and
documentation sanity checks
**Time estimate:** 12 hours

### Tasks

1. Run required quality gates for any modified C or header files.
2. Run focused allocation-failure tests and selected subsystem regression
   tests.
3. Run documentation, report, and claim-guard tests affected by claim
   recalibration.
4. Run install/package/report checks touched by Epic 15 closeout updates.
5. Record skipped or infeasible checks with reasons.
6. Write the Day 12 integrated-validation artifact.

### Deliverables

- final validation command log
- allocation-failure proof validation
- documentation and claim-guard validation
- skipped-check rationale
- Day 12 validation artifact

### Completion Criteria

- required checks pass before closeout
- validation scope matches touched files
- skipped checks are justified and visible

---

## Day 13: Epic 15 Retrospective Finalization

**Title:** Retrospective Final
**Theme:** Finalize Epic 15 retrospective, residual queue, and closeout
evidence links
**Time estimate:** 12 hours

### Tasks

1. Reconcile the Day 11 draft against Day 12 validation results.
2. Finalize `docs/planning/EPIC_15/EPIC_15_RETROSPECTIVE.md`.
3. Verify that all Sprint 167-176 deliverables are represented as complete,
   narrowed, or residualized.
4. Add final residual queue and next-epic handoff candidates.
5. Cross-check links to validation artifacts, evidence ledger entries, and
   claim documentation.
6. Write the Day 13 retrospective-finalization artifact.

### Deliverables

- finalized Epic 15 retrospective
- final residual queue
- closeout evidence link map
- next-epic handoff candidates
- Day 13 retrospective-final artifact

### Completion Criteria

- Epic 15 retrospective is complete and internally consistent
- every deferred item has a reason and next step
- evidence links support all earned claims

---

## Day 14: Sprint And Epic Closeout

**Title:** Closeout
**Theme:** Finalize Sprint 176 records, verify repository state, and prepare
the Epic 15 closeout handoff
**Time estimate:** 12 hours

### Tasks

1. Review all Sprint 176 artifacts for consistency and completeness.
2. Update Sprint 176 working notes with final validation status and decisions.
3. Confirm Epic 15 retrospective, claim docs, and allocation-failure evidence
   agree.
4. Run final lightweight sanity checks appropriate for the final touched
   surface.
5. Prepare the Sprint 176 retrospective handoff.
6. Write the Day 14 sprint-closeout artifact.

### Deliverables

- completed Sprint 176 working notes
- Day 14 sprint-closeout artifact
- final validation and repository status notes
- Sprint 176 retrospective handoff
- Epic 15 closeout handoff

### Completion Criteria

- Sprint 176 has a complete day-by-day evidence trail
- Epic 15 closeout state is ready for review
- remaining residuals are explicit and do not weaken public claims
