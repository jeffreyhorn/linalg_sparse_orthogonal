# Sprint 110 Plan: Residual Matrix I/O, Behavior Owners & Proof-Owner Follow-Through

**Sprint Duration:** 14 days
**Goal:** Convert Sprint 109 residual deferred debt into dependency-ordered
implementation and proof-owner follow-through without duplicating dense Jacobi
extraction, Matrix Market owner selection, QR exact-RHS cleanup, or completed
validation/drift work. This sprint implements the Sprint 110 section of
`docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 110 begins from:
- the Sprint 109 retrospective and residual-debt closeout artifacts
- merged dense Jacobi extraction with source-list parity and focused validation
- Sprint 109 eigensolver behavior-sensitive no-go conditions
- Sprint 109 Matrix Market future-owner contract
- merged QR exact-RHS cleanup, which is excluded from duplicate work

The sprint must:
- re-read Sprint 109 residual deferred debt and create a duplicate-work fence
- decide Matrix builder ownership before any Matrix Market source movement
- either move Matrix Market load/save toward a private I/O owner or publish a
  no-split deferral artifact
- validate at most one behavior-sensitive eigensolver owner beyond dense Jacobi
- perform one bounded proof-owner cleanup batch across QR, LDLT CSC, or
  iterative tests without hiding proof values
- review SVD proof-loop boundaries and extract only one safe setup helper family
- close with validation, metrics, and residual handoff for downstream sprints

**End State:** Sprint 110 leaves behind:
- a Sprint 109 residual-debt intake and duplicate-work exclusion artifact
- a Matrix builder ownership decision
- an optional Matrix Market source split with build/source-list parity, or a
  documented no-split deferral
- focused matrix and solver-smoke validation for any Matrix Market movement
- a behavior-sensitive eigensolver owner validation or no-move contract
- one bounded proof-owner cleanup batch, if the boundary remains safe
- an SVD proof-loop boundary artifact and optional safe helper cleanup
- validation evidence proving no accidental public API, install-header,
  helper-target, source-list, or reviewed CTest drift

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 110 project-plan estimate.

---

## Day 1: Sprint 110 Scope & Duplicate-Work Fence

**Title:** Residual Intake
**Theme:** Convert Sprint 109 residual deferred debt into a dependency-safe
Sprint 110 work package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 110 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 109 `WORKING_NOTES.md`, artifacts, and retrospective,
   emphasizing residual deferred debt.
3. Inventory candidate owners:
   - `src/sparse_matrix.c`
   - Matrix Market load/save logic
   - `SparseBuildEntry` and `sparse_matrix_build_from_entries`
   - eigensolver behavior owners beyond dense Jacobi
   - `tests/test_qr.c`
   - `tests/test_ldlt_csc.c`
   - `tests/test_iterative.c`
   - `tests/test_svd.c`
4. Mark completed Sprint 109 work as explicit exclusions:
   - dense Jacobi source extraction
   - Matrix Market future-owner selection
   - QR exact-RHS helper cleanup
   - validation and drift closeout already completed
5. Create Sprint 110 working notes and artifacts directory.
6. Write the residual-debt intake and duplicate-work exclusion artifact.

### Deliverables
- Sprint 110 workstream inventory
- duplicate-work exclusion list
- dependency-ordered work queue
- working-notes baseline
- Day 1 residual-debt intake artifact

### Completion Criteria
- every Sprint 110 project-plan item has day-level ownership
- no Sprint 110 item depends on work scheduled later in the sprint
- completed Sprint 109 work is explicitly excluded
- validation expectations are visible before code movement begins

---

## Day 2: Matrix Builder Ownership Audit

**Title:** Builder Audit
**Theme:** Determine whether Matrix builder helpers can leave the central matrix
shell safely
**Time estimate:** 12 hours

### Tasks
1. Locate `SparseBuildEntry`, `sparse_matrix_build_from_entries`, and all direct
   callers.
2. Map builder use across Matrix Market load, CSR/CSC constructors, copy,
   transpose, tests, and private helpers.
3. Identify public behavior coupled to builder ordering, duplicate handling,
   zero handling, allocation failure, and error propagation.
4. Check whether the builder needs a private header boundary, a private source
   owner, or central `src/sparse_matrix.c` ownership.
5. Record risks for source-list parity, install headers, and downstream
   consumers.
6. Draft Matrix builder ownership decision criteria.

### Deliverables
- Matrix builder dependency map
- public behavior coupling notes
- private-header and source-owner options
- extraction risk checklist
- Day 2 builder audit artifact

### Completion Criteria
- all builder callers are known
- Matrix Market movement prerequisites are explicit
- public behavior risks are documented before a decision
- no Matrix Market source split begins before the builder boundary is decided

---

## Day 3: Matrix Builder Ownership Decision

**Title:** Builder Decision
**Theme:** Choose the Matrix builder owner and publish the prerequisite contract
for Matrix Market work
**Time estimate:** 12 hours

### Tasks
1. Compare the private builder source option against central matrix-shell
   ownership.
2. Decide whether `SparseBuildEntry` and `sparse_matrix_build_from_entries`
   should move, stay central, or become a documented future split.
3. Define the owner contract for copy, transpose, load, duplicate-entry, and
   allocation-failure behavior.
4. Define required focused matrix and solver-smoke validation for any later
   Matrix Market movement.
5. Update working notes with the go/no-go decision.
6. Write the Matrix builder ownership decision artifact.

### Deliverables
- Matrix builder ownership decision
- go/no-go rationale
- Matrix Market prerequisite checklist
- focused validation checklist
- Day 3 decision artifact

### Completion Criteria
- the decision is explicit enough to unblock or defer Matrix Market movement
- no public API or install-header change is implied accidentally
- Matrix Market work has a concrete validation gate
- downstream days can proceed without revisiting builder ownership

---

## Day 4: Matrix Market Boundary Plan

**Title:** Matrix Market Plan
**Theme:** Prepare Matrix Market source movement or a deliberate no-split
deferral from the builder decision
**Time estimate:** 12 hours

### Tasks
1. Locate Matrix Market load/save logic and helper dependencies.
2. Map source-list, Makefile, CMake, manifest, and package implications for a
   future `src/sparse_matrix_io.c` owner.
3. Identify focused matrix tests that cover roundtrip, duplicate entries,
   pattern/symmetric handling, bad input, and errno behavior.
4. Identify solver-smoke fixture lanes that prove loaded matrices still work
   through direct and iterative paths.
5. If Day 3 selected no split, write the no-split deferral artifact and close
   Matrix Market implementation work for this sprint.
6. If Day 3 selected a split, write the implementation checklist for Day 5.

### Deliverables
- Matrix Market dependency map
- build/source-list update checklist
- focused matrix validation plan
- solver-smoke validation plan
- implementation checklist or no-split deferral artifact

### Completion Criteria
- Matrix Market movement is either safely planned or explicitly deferred
- every required build-system touchpoint is listed
- validation covers file I/O behavior and solver use after load
- no code movement has occurred without a go decision

---

## Day 5: Matrix Market Source Split Follow-Through

**Title:** Matrix Market Follow-Through
**Theme:** Execute the approved Matrix Market movement or close the no-split
path cleanly
**Time estimate:** 12 hours

### Tasks
1. If the split is approved, move only the planned Matrix Market load/save
   behavior toward `src/sparse_matrix_io.c`.
2. Update Makefile, CMake, package manifest, and source-list parity as needed.
3. Keep public headers, install headers, and CTest registration unchanged unless
   explicitly justified.
4. If the split is not approved, ensure the deferral artifact captures the
   blocker and future prerequisites.
5. Run focused formatting/build checks required by touched files.
6. Update working notes with implementation status and residuals.

### Deliverables
- Matrix Market source movement or no-split deferral
- source-list and build-system parity updates, if applicable
- public API/install-header no-drift notes
- focused implementation notes
- Day 5 follow-through artifact

### Completion Criteria
- implementation or deferral matches the Day 4 plan
- build/source-list parity is preserved for any new source owner
- public API and install headers remain stable
- follow-on validation needs are documented for Day 6

---

## Day 6: Matrix Market Focused Validation

**Title:** Matrix I/O Validation
**Theme:** Prove Matrix Market behavior and solver-smoke lanes after movement
or deferral
**Time estimate:** 12 hours

### Tasks
1. Run focused Matrix Market tests identified on Day 4.
2. Run solver-smoke fixtures that consume loaded matrices.
3. Verify no reviewed CTest registration drift occurred.
4. Verify source-list, Makefile, CMake, and package manifest parity if code
   moved.
5. Capture any platform or install/export implications for downstream sprints.
6. Write the Matrix Market validation artifact.

### Deliverables
- focused Matrix Market validation results
- solver-smoke validation results
- CTest no-drift evidence
- source-list/build parity evidence
- Day 6 validation artifact

### Completion Criteria
- Matrix Market behavior is proven or explicitly deferred with evidence
- solver-smoke lanes pass for any changed file I/O path
- no accidental public API, install-header, or CTest drift remains
- Matrix I/O work is closed before eigensolver behavior-owner validation begins

---

## Day 7: Eigensolver Behavior Owner Selection

**Title:** Eigs Owner Selection
**Theme:** Choose one behavior-sensitive eigensolver owner beyond dense Jacobi
for bounded validation
**Time estimate:** 12 hours

### Tasks
1. Re-read Sprint 109 eigensolver behavior-sensitive no-go artifacts.
2. Inventory candidate owners:
   - defaults
   - backend dispatch
   - workspace preparation and growth
   - refinement defaults and budgets
   - shift-invert setup
   - shared Lanczos kernels
3. Exclude dense Jacobi work already completed in Sprint 109.
4. Select at most one behavior owner for validation this sprint.
5. Define direct tests, source boundaries, and no-public-header-drift gates for
   the selected owner.
6. Write the behavior-owner selection artifact.

### Deliverables
- eigensolver behavior-owner candidate list
- selected owner or no-move rationale
- test and no-drift checklist
- Day 7 behavior-owner selection artifact

### Completion Criteria
- exactly one behavior owner is selected, or all candidates are explicitly
  deferred
- dense Jacobi work is not duplicated
- behavior preservation tests are identified before code movement
- public header drift is fenced off

---

## Day 8: Eigensolver Behavior Owner Validation

**Title:** Eigs Validation
**Theme:** Validate the selected eigensolver behavior owner or publish a no-move
contract
**Time estimate:** 12 hours

### Tasks
1. If Day 7 selected a low-risk owner, make only the narrow approved movement or
   internal boundary cleanup.
2. Add or strengthen direct tests for defaults, dispatch, workspace,
   refinement, or shift-invert behavior as applicable.
3. If Day 7 selected no movement, write the no-move contract and required future
   proof.
4. Run focused eigensolver validation:
   - `test_eigs`
   - `test_eigs_thick_restart`
   - `test_eigs_lobpcg`
   - `test_sprint29_integration`
5. Verify public headers, install headers, source lists, helper targets, and
   CTest registrations do not drift unexpectedly.
6. Update working notes with the validation result.

### Deliverables
- behavior-owner movement or no-move contract
- direct behavior-preservation tests, if applicable
- focused eigensolver validation evidence
- public-header/source-list/no-drift notes
- Day 8 validation artifact

### Completion Criteria
- selected eigensolver behavior remains externally unchanged
- focused eigensolver gates pass
- no public API or install-header drift occurs
- unsafe behavior-sensitive movement is explicitly deferred

---

## Day 9: Direct & Iterative Proof-Owner Boundary Selection

**Title:** Proof Boundary
**Theme:** Select one bounded proof-owner cleanup batch without hiding proof
values
**Time estimate:** 12 hours

### Tasks
1. Review remaining QR sequential RHS, LDLT CSC oracle, and iterative exact-RHS
   cleanup candidates.
2. Exclude the merged Sprint 109 QR exact-RHS helper cleanup from the worklist.
3. For each candidate, identify proof values that must remain visible at call
   sites:
   - least-squares residuals
   - refinement results
   - dense oracle comparisons
   - convergence status
   - residual norms
4. Select one bounded cleanup family for Day 10.
5. Define focused test gates and no-helper-target-drift checks.
6. Write the proof-owner boundary artifact.

### Deliverables
- QR/LDLT CSC/iterative proof-owner candidate map
- selected bounded cleanup family
- proof-value preservation checklist
- focused validation checklist
- Day 9 boundary artifact

### Completion Criteria
- one cleanup family is selected, or all are explicitly deferred
- proof assertions remain visible and localized
- no new compiled helper target is required
- validation gates are known before edits begin

---

## Day 10: Direct & Iterative Proof-Owner Cleanup

**Title:** Proof Cleanup
**Theme:** Execute one bounded QR, LDLT CSC, or iterative proof-owner cleanup
without weakening tests
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 9 cleanup to only the selected test owner or helper family.
2. Preserve visible proof values at call sites.
3. Avoid broad helper abstractions that mix solver families.
4. Avoid new compiled helper targets unless already approved by the boundary
   artifact.
5. Run the selected focused test binary and any adjacent proof-owner tests.
6. Update working notes with changed assertions and residual risks.

### Deliverables
- bounded proof-owner cleanup change or explicit deferral
- focused test results
- proof-value preservation notes
- no-helper-target-drift notes
- Day 10 cleanup artifact

### Completion Criteria
- cleanup stays within the selected proof-owner family
- test failures remain localized and meaningful
- focused tests pass
- hidden proof-value regressions are not introduced

---

## Day 11: SVD Proof-Loop Boundary Review

**Title:** SVD Boundary
**Theme:** Identify the one safe SVD setup helper family, or defer cleanup if
proof visibility would weaken
**Time estimate:** 12 hours

### Tasks
1. Review `tests/test_svd.c` storage-layout, stride, rank, orthogonality, and
   reconstruction proof loops.
2. Map proof values that must remain visible at call sites.
3. Identify repeated setup that can be extracted without hiding rank,
   orthogonality, or reconstruction claims.
4. Select at most one safe setup helper family for Day 12.
5. Define focused SVD validation commands.
6. Write the SVD proof-loop boundary artifact.

### Deliverables
- SVD proof-loop map
- proof-value preservation checklist
- selected setup helper family or no-cleanup deferral
- focused SVD validation checklist
- Day 11 boundary artifact

### Completion Criteria
- SVD proof obligations are visible before edits
- only one helper family is selected
- rank, orthogonality, stride, and reconstruction claims remain inspectable
- no SVD cleanup begins without a boundary artifact

---

## Day 12: SVD Proof-Loop Cleanup

**Title:** SVD Cleanup
**Theme:** Extract one safe SVD setup helper family or publish the deferral
evidence
**Time estimate:** 12 hours

### Tasks
1. If Day 11 selected a safe helper family, apply only that extraction.
2. Keep storage-layout, stride, rank, orthogonality, and reconstruction proof
   values visible at call sites.
3. If no safe family was selected, write the explicit SVD deferral artifact.
4. Run focused SVD tests and any adjacent integration tests required by the
   boundary artifact.
5. Check no public API, helper-target, or CTest registration drift occurred.
6. Update working notes with the SVD closeout status.

### Deliverables
- one SVD setup helper cleanup or deferral artifact
- focused SVD validation results
- proof-value preservation notes
- drift-check notes
- Day 12 cleanup artifact

### Completion Criteria
- SVD cleanup does not hide claim-critical assertions
- focused SVD validation passes
- no reviewed test registration or helper-target drift remains
- residual SVD work is explicit for downstream planning

---

## Day 13: Integrated Validation & Metrics

**Title:** Validation Gate
**Theme:** Run the required quality gates and capture maintainability, source,
and proof-owner metrics
**Time estimate:** 12 hours

### Tasks
1. Review all files touched during Sprint 110.
2. Run the required quality checks for touched code, test, build, and docs
   surfaces.
3. If any `*.c` or `*.h` files changed, run:
   - `make format`
   - `make lint`
   - `make test`
4. Verify no accidental public API, install-header, source-list, helper-target,
   or reviewed CTest drift remains.
5. Capture before/after metrics for touched source and proof-owner files.
6. Write the integrated validation and metrics artifact.

### Deliverables
- required quality-check results
- public API/install-header no-drift evidence
- source-list/helper-target/CTest no-drift evidence
- maintainability metrics
- Day 13 validation artifact

### Completion Criteria
- all required checks pass
- any code/test/build change has matching validation evidence
- drift status is explicit and reviewed
- metrics are ready for Sprint 110 closeout

---

## Day 14: Sprint Closeout & Residual Handoff

**Title:** Closeout
**Theme:** Close Sprint 110 with artifacts, residuals, and downstream handoff
for shifted Epic 10 sprints
**Time estimate:** 12 hours

### Tasks
1. Reconcile all Sprint 110 artifacts against the project-plan items.
2. Update `WORKING_NOTES.md` with final decisions, validations, and residuals.
3. Confirm duplicate-work exclusions stayed honored through the sprint.
4. Identify residual deferred debt for Sprints 111 and beyond.
5. Write the Sprint 110 closeout and residual handoff artifact.
6. Prepare the retrospective input list from working notes and artifacts.

### Deliverables
- completed Sprint 110 artifact index
- final duplicate-work exclusion confirmation
- residual deferred debt list
- downstream handoff notes
- Day 14 closeout artifact

### Completion Criteria
- every Sprint 110 project-plan item has a final disposition
- validation evidence is linked from closeout notes
- residual debt is dependency-ordered for downstream planning
- Sprint 110 is ready for retrospective creation
