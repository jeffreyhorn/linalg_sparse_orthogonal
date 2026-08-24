# Sprint 178 Plan: Allocation-Failure Proof Batch 2

**Sprint Duration:** 14 days
**Goal:** Add deterministic allocation-failure cleanup evidence for one
additional high-risk subsystem beyond iterative repeated-run handles. This
sprint implements the Sprint 178 section of
`docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Source Artifact Note:** This plan lives under
`docs/planning/EPIC_16/SPRINT_178/PLAN.md` and implements the Sprint 178
section of `docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Starting Point:** Sprint 178 begins from:

- Sprint 176 private allocation-failure hook semantics and iterative
  repeated-run handle proof;
- Sprint 177 Gate 1 acceptance requirements for Allocation-Failure Proof
  Batch 2;
- Sprint 177 Day 12 handoff for Sprint 178;
- current allocation helper surfaces in `src/sparse_alloc_internal.*`;
- current candidate subsystem surfaces in matrix construction/conversion,
  direct solver setup, decomposition workspace, and public matrix/solver
  APIs;
- retained non-claims for broad allocation-failure safety across all solvers,
  constructors, package/install flows, generated tooling, and unrelated
  allocation paths.

The sprint must:

- select exactly one additional allocation-heavy subsystem;
- document selected entry points, ownership graph, failure sites, cleanup
  invariants, retry semantics, and unsupported breadth;
- extend or reuse deterministic allocation-failure injection only as far as
  needed for the selected subsystem;
- add regression tests for failed construction, factorization, conversion, or
  workspace growth as applicable;
- prove cleanup, no stale public state publication, and successful retry after
  reset;
- add a focused Make/CTest gate or label for the selected proof;
- update README and maintainer wording with scoped allocation-failure claims;
- run required C/header quality gates if implementation files change.

**End State:** Sprint 178 leaves behind:

- a selected-subsystem decision artifact;
- cleanup invariant documentation for the selected subsystem;
- deterministic allocation-failure regression coverage;
- a focused validation target and CTest label or registration proof;
- scoped public and maintainer documentation;
- Sprint 178 working notes, daily artifacts, and validation records.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 178 project-plan estimate.

---

## Day 1: Sprint Intake And Gate Baseline

**Title:** Intake And Gate Baseline
**Theme:** Establish Sprint 178 scope, artifact layout, acceptance gate, and
allocation-failure boundaries
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 178 section of
   `docs/planning/EPIC_16/PROJECT_PLAN.md`.
2. Review Sprint 177 Gate 1 and Day 12 Sprint 178 handoff.
3. Create Sprint 178 working notes and artifact directory structure.
4. Record current allocation-failure proof status from Sprint 176.
5. Record protected non-claims for broad allocation-failure coverage.
6. Write the Day 1 sprint-intake artifact.

### Deliverables

- Sprint 178 working-notes baseline
- artifact directory structure
- acceptance-gate summary
- current proof and non-claim baseline
- Day 1 sprint-intake artifact

### Completion Criteria

- Sprint 178 scope is tied to the Epic 16 project plan
- Gate 1 pass/fail requirements are visible before selection begins
- broad allocation-failure claims remain rejected

---

## Day 2: Allocation Surface Inventory

**Title:** Allocation Surface Inventory
**Theme:** Inventory candidate allocation-heavy subsystems and compare closure
fitness
**Time estimate:** 12 hours

### Tasks

1. Inspect `src/sparse_alloc_internal.*` and current fail-injection tests.
2. Inventory matrix construction and conversion allocation paths.
3. Inventory direct solver setup and factorization allocation paths.
4. Inventory decomposition workspace ownership paths.
5. Compare candidates by public state exposure, cleanup observability, retry
   feasibility, and implementation risk.
6. Write the Day 2 allocation-surface inventory artifact.

### Deliverables

- allocation-heavy subsystem inventory
- candidate failure-site list
- cleanup observability notes
- retry-feasibility notes
- Day 2 allocation-surface artifact

### Completion Criteria

- at least three candidate subsystems are comparable
- failure and cleanup paths are mapped to concrete files
- no subsystem is selected before feasibility is assessed

---

## Day 3: Subsystem Selection Detail

**Title:** Subsystem Selection Detail
**Theme:** Select one subsystem and freeze the exact entry points, ownership
paths, and failure sites in scope
**Time estimate:** 12 hours

### Tasks

1. Review Day 2 candidate inventory.
2. Select one subsystem for deterministic allocation-failure proof.
3. Define exact public or internal entry points in scope.
4. Define allocation sites and ownership cleanup paths in scope.
5. Define out-of-scope adjacent allocation paths and retained non-claims.
6. Write the Day 3 subsystem-selection decision artifact.

### Deliverables

- selected subsystem decision
- in-scope entry-point list
- selected failure-site list
- out-of-scope and non-claim list
- Day 3 subsystem-selection artifact

### Completion Criteria

- exactly one subsystem is selected
- selected failure sites are deterministic and testable
- scope does not imply broad allocation-failure coverage

---

## Day 4: Cleanup Invariant Record

**Title:** Cleanup Invariant Record
**Theme:** Document cleanup behavior, no-publication rules, retry semantics,
and unsupported breadth before code changes
**Time estimate:** 12 hours

### Tasks

1. Trace selected subsystem ownership from allocation through cleanup.
2. Define what public state must remain unchanged on injected failure.
3. Define what partially allocated internal state must be released.
4. Define successful retry expectations after fail-injection reset.
5. Define the exact unsupported breadth that must remain documented.
6. Write the Day 4 cleanup-invariant artifact.

### Deliverables

- cleanup invariant record
- no-stale-publication rules
- retry semantics
- unsupported-breadth notes
- Day 4 cleanup-invariant artifact

### Completion Criteria

- cleanup assertions can be converted directly into tests
- retry behavior is observable without changing public API
- broad allocation-failure non-claims are preserved

---

## Day 5: Harness Design

**Title:** Harness Design
**Theme:** Design deterministic allocation-failure injection and observation
for the selected subsystem
**Time estimate:** 12 hours

### Tasks

1. Compare existing Sprint 176 allocation-failure hook semantics against the
   selected subsystem's allocation sites.
2. Decide whether the existing fail-at-count hook is sufficient.
3. Design any subsystem-local harness helpers needed for deterministic
   failure placement.
4. Define test helper APIs and reset behavior.
5. Define how tests will detect cleanup, no stale state, and retry success.
6. Write the Day 5 harness-design artifact.

### Deliverables

- harness design
- helper API plan
- reset and countdown semantics
- cleanup observation strategy
- Day 5 harness-design artifact

### Completion Criteria

- failure injection remains deterministic
- hook semantics stay private/internal
- no public product API is added for test injection

---

## Day 6: Harness Implementation

**Title:** Harness Implementation
**Theme:** Extend or reuse allocation-failure injection for the selected
subsystem
**Time estimate:** 12 hours

### Tasks

1. Implement the minimal harness changes needed for the selected subsystem.
2. Preserve existing Sprint 176 hook countdown semantics.
3. Add or update test helper declarations if needed.
4. Add focused compile coverage for the new helper path.
5. Run the smallest relevant build or compile check for touched files.
6. Write the Day 6 harness-implementation artifact.

### Deliverables

- minimal harness implementation
- helper declarations or test helpers
- focused compile/build note
- Day 6 harness-implementation artifact

### Completion Criteria

- existing allocation-failure tests still compile
- new harness can trigger the selected failure site deterministically
- no broad hook redesign is introduced

---

## Day 7: First Failure Regression

**Title:** First Failure Regression
**Theme:** Add the first deterministic injected-failure regression for the
selected subsystem
**Time estimate:** 12 hours

### Tasks

1. Add the first selected-subsystem failure test.
2. Assert the expected error contract.
3. Assert cleanup or no stale public state for the first ownership path.
4. Assert successful retry after fail-injection reset.
5. Run the focused test locally.
6. Write the Day 7 first-regression artifact.

### Deliverables

- first deterministic failure regression
- cleanup and retry assertions
- focused test output note
- Day 7 first-regression artifact

### Completion Criteria

- first failure case fails before cleanup is proven or passes with the fix
- retry after reset is covered
- the test does not rely on nondeterministic allocation ordering

---

## Day 8: Failure Coverage Expansion

**Title:** Failure Coverage Expansion
**Theme:** Add remaining selected ownership-path regressions
**Time estimate:** 12 hours

### Tasks

1. Add failure tests for remaining selected ownership paths.
2. Cover failed construction, factorization, conversion, or workspace growth
   depending on the selected subsystem.
3. Add no-stale-state assertions for each path.
4. Add retry assertions for each path.
5. Run focused selected-subsystem tests.
6. Write the Day 8 coverage-expansion artifact.

### Deliverables

- expanded selected-subsystem regression tests
- per-path cleanup assertions
- per-path retry assertions
- Day 8 coverage-expansion artifact

### Completion Criteria

- each selected failure site has a regression test
- cleanup behavior is asserted for each selected ownership path
- broad adjacent allocation paths remain out of scope

---

## Day 9: Cleanup Fixes And Error-Contract Alignment

**Title:** Cleanup And Error Contracts
**Theme:** Fix selected cleanup defects and preserve public error-ordering
contracts
**Time estimate:** 12 hours

### Tasks

1. Fix selected subsystem cleanup defects exposed by Day 7-8 tests.
2. Ensure no partial public state is published on injected failure.
3. Preserve documented NULL-handle and bad-argument precedence.
4. Confirm retry after reset succeeds without stale internal state.
5. Run focused tests and existing related tests.
6. Write the Day 9 cleanup-fix artifact.

### Deliverables

- cleanup fixes for selected subsystem
- error-contract alignment note
- focused regression results
- Day 9 cleanup-fix artifact

### Completion Criteria

- all selected failure regressions pass
- no stale public state is observable after injected failure
- public error precedence remains consistent

---

## Day 10: Focused Gate Registration

**Title:** Focused Gate Registration
**Theme:** Add maintained Make and CTest coverage for the selected proof
**Time estimate:** 12 hours

### Tasks

1. Add a focused Make target for the selected allocation-failure proof.
2. Add or update CMake/CTest registration or labels as needed.
3. Ensure the focused gate runs only the selected proof and needed support
   tests.
4. Add registration-drift checks if new targets or labels are introduced.
5. Run the focused Make and CTest commands locally where available.
6. Write the Day 10 focused-gate artifact.

### Deliverables

- focused Make target
- CTest registration or label
- registration validation note
- Day 10 focused-gate artifact

### Completion Criteria

- maintainers have a single focused command for the selected proof
- CMake/CTest registration matches Makefile registration
- focused gate does not imply broad allocation-failure coverage

---

## Day 11: Public And Maintainer Documentation

**Title:** Scoped Claim Documentation
**Theme:** Update README and maintainer guidance with the exact selected proof
and protected non-claims
**Time estimate:** 12 hours

### Tasks

1. Update README wording for the selected allocation-failure proof.
2. Update maintainer guidance with the selected subsystem, command, and
   support boundary.
3. Keep "allocation-failure" terminology consistent.
4. Preserve non-claims for unselected solver families, constructors,
   package/install flows, generated tooling, and unrelated allocation paths.
5. Update Sprint 178 artifacts with documentation decisions.
6. Write the Day 11 documentation artifact.

### Deliverables

- README scoped claim update
- maintainer allocation-failure guidance update
- protected non-claim wording
- Day 11 documentation artifact

### Completion Criteria

- positive claim names exactly one selected subsystem
- docs name the focused validation command
- broad allocation-failure wording is absent

---

## Day 12: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run required C/header, focused, and registration validation for all
touched surfaces
**Time estimate:** 12 hours

### Tasks

1. Run the focused selected-subsystem allocation-failure gate.
2. Run related focused tests for the selected subsystem.
3. Run CMake/CTest registration validation if registration changed.
4. Run `make format && make lint && make test` if C or header files changed.
5. Run documentation hygiene checks.
6. Write the Day 12 validation artifact.

### Deliverables

- focused gate validation record
- full C quality-gate record if required
- registration validation record
- documentation hygiene note
- Day 12 validation artifact

### Completion Criteria

- all required quality gates pass
- any unrun checks are explicitly justified
- validation matches the changed file surfaces

---

## Day 13: Claim Recalibration And Residuals

**Title:** Claim Recalibration
**Theme:** Reconcile evidence, public claims, non-claims, and residual gaps
before closeout
**Time estimate:** 12 hours

### Tasks

1. Compare final evidence against Sprint 177 Gate 1.
2. Confirm documentation wording matches the selected proof.
3. Record retained non-claims for broad allocation-failure coverage.
4. Record any residuals or blockers found during implementation.
5. Prepare retrospective inputs.
6. Write the Day 13 claim-recalibration artifact.

### Deliverables

- Gate 1 reconciliation
- retained non-claim list
- residual queue update
- retrospective input notes
- Day 13 claim-recalibration artifact

### Completion Criteria

- earned claim is no broader than evidence
- remaining gaps are explicit residuals
- sprint can close without ambiguous allocation-failure wording

---

## Day 14: Sprint Closeout

**Title:** Sprint Closeout
**Theme:** Finalize Sprint 178 records and leave a clean handoff for Sprint
179
**Time estimate:** 12 hours

### Tasks

1. Finalize Sprint 178 working notes.
2. Finalize selected-subsystem, cleanup-invariant, harness, test, gate,
   documentation, validation, and residual artifacts.
3. Confirm generated files and artifacts are placed under
   `docs/planning/EPIC_16/SPRINT_178/`.
4. Run `git diff --check`.
5. Prepare Sprint 178 retrospective inputs.
6. Write the Day 14 sprint-closeout artifact.

### Deliverables

- finalized Sprint 178 working notes
- finalized artifact inventory
- Sprint 179 handoff confirmation
- validation note
- Day 14 closeout artifact

### Completion Criteria

- Sprint 178 is ready for retrospective creation
- selected allocation-failure proof has focused evidence or a documented
  blocker
- broad allocation-failure non-claims remain protected
