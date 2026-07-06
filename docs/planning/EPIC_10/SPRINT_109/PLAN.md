# Sprint 109 Plan: Residual Source Boundary & Proof-Owner Debt Closeout

**Sprint Duration:** 14 days
**Goal:** Convert Sprint 108's residual deferred debt into one bounded
implementation/source-boundary pass without duplicating completed Sprint 108
helpers, broadening public support surfaces, or hiding proof logic in the
remaining giant tests. This sprint implements the Sprint 109 section of
`docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 109 begins from:
- the Sprint 108 retrospective and residual-debt closeout artifacts
- merged Sprint 108 LDLT CSC, QR, iterative, and SVD helper follow-through
- the Sprint 108 eigensolver source-feasibility artifacts
- the Sprint 108 matrix-shell public-behavior review
- current reviewed constraints around public API, install headers, source-list
  parity, helper targets, and CTest registration counts

The sprint must:
- re-read and order Sprint 108 residual deferred debt
- explicitly exclude already-completed Sprint 108 helpers
- revalidate the `s21_dense_sym_jacobi` source-boundary candidate
- either extract only that dense Jacobi helper or publish a no-split deferral
- audit behavior-sensitive eigensolver boundaries without moving unsafe code
- define one future `src/sparse_matrix.c` public-behavior owner contract
- perform one bounded giant-test proof-owner cleanup pass
- close with validation, metrics, and residuals for shifted downstream sprints

**End State:** Sprint 109 leaves behind:
- a residual-debt intake and dependency-ordering artifact
- an eigensolver dense Jacobi source-boundary artifact
- an optional private dense spectral helper extraction with source-list parity,
  or an explicit no-split deferral artifact
- an eigensolver behavior-sensitive boundary audit
- a matrix-shell public-behavior source-boundary contract
- one bounded giant-test proof-owner cleanup pass if safe
- validation evidence proving no accidental public API, install-header,
  source-list, helper-target, or CTest drift

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 109 project-plan estimate.

---

## Day 1: Sprint 109 Scope & Residual Intake

**Title:** Residual Intake
**Theme:** Convert Sprint 108 residual deferred debt into a dependency-safe
Sprint 109 work package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 109 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 108 `WORKING_NOTES.md`, artifacts, and retrospective,
   emphasizing residual deferred debt.
3. Inventory Sprint 109 owners:
   - `src/sparse_eigs.c`
   - eigensolver private/internal headers and source lists
   - `src/sparse_matrix.c`
   - `tests/test_ldlt_csc.c`
   - `tests/test_qr.c`
   - `tests/test_iterative.c`
   - `tests/test_svd.c`
4. Mark completed Sprint 108 helpers as explicit exclusions.
5. Create Sprint 109 working notes and artifacts directory.
6. Write the residual-debt intake and dependency-ordering artifact.

### Deliverables
- Sprint 109 workstream inventory
- duplicate-work exclusion list
- dependency-ordered work queue
- working-notes baseline
- Day 1 residual-debt intake artifact

### Completion Criteria
- every Sprint 109 project-plan item has day-level ownership
- no Sprint 109 item depends on work scheduled later in the sprint
- completed Sprint 108 helper work is explicitly excluded
- validation expectations are visible before code movement begins

---

## Day 2: Dense Jacobi Boundary Revalidation

**Title:** Jacobi Boundary
**Theme:** Revalidate `s21_dense_sym_jacobi` as the only candidate for an
eigensolver source move
**Time estimate:** 12 hours

### Tasks
1. Locate `s21_dense_sym_jacobi` and all direct callers.
2. Map private data dependencies, static helper dependencies, and header needs.
3. Check whether extraction can avoid public header, install-header, and API
   changes.
4. Define the proposed private source owner and internal declaration point.
5. Identify Makefile, CMake, package manifest, and source-list parity updates.
6. Write the dense Jacobi source-boundary artifact with go/no-go criteria.

### Deliverables
- dense Jacobi dependency map
- source-owner and declaration proposal
- build-system/source-list update checklist
- focused validation command list
- go/no-go boundary artifact

### Completion Criteria
- the candidate boundary is narrower than surrounding eigensolver behavior
- all source-list parity changes are known before edits begin
- no public API or install-header change is required
- no code is moved before the boundary is approved

---

## Day 3: Source-List Parity & Validation Harness Prep

**Title:** Source Parity Prep
**Theme:** Prepare the private helper extraction path without changing behavior
**Time estimate:** 12 hours

### Tasks
1. Inspect Makefile, CMake, pkg-config, install/export, and manifest handling
   for internal source additions.
2. Record the current reviewed CTest registration surface.
3. Confirm focused eigensolver gates:
   - `test_eigs`
   - `test_eigs_thick_restart`
   - `test_eigs_lobpcg`
   - `test_sprint29_integration`
4. Define exact no-drift checks for public headers, install headers, helper
   targets, and CTest counts.
5. Create an extraction checklist that can be followed atomically on Day 4.

### Deliverables
- source-list parity checklist
- reviewed CTest baseline notes
- focused eigensolver validation plan
- no-drift checklist
- extraction execution checklist

### Completion Criteria
- every build-system touch point is identified
- focused validation commands are known and bounded
- public support surfaces are protected before implementation starts
- Day 4 can proceed without discovering new source-list owners

---

## Day 4: Dense Jacobi Extraction or Deferral

**Title:** Jacobi Follow-Through
**Theme:** Move only the approved dense Jacobi helper or publish an explicit
no-split deferral
**Time estimate:** 12 hours

### Tasks
1. If Days 2-3 approved extraction, move only `s21_dense_sym_jacobi` into a
   private dense spectral helper source.
2. Add only the required private declaration and source-list updates.
3. Preserve existing behavior, call signatures, diagnostics, and ownership.
4. If extraction is not safe, write the no-split deferral artifact instead of
   moving code.
5. Run focused eigensolver build and test checks for touched surfaces.
6. Record before/after source-size and source-list metrics.

### Deliverables
- private dense spectral helper extraction or no-split deferral artifact
- source-list parity updates if extraction occurs
- focused validation notes
- before/after metrics

### Completion Criteria
- no behavior-sensitive eigensolver code is moved accidentally
- no public header, install-header, helper-target, or CTest drift occurs
- focused eigensolver checks pass if code changed
- deferral, if chosen, is evidence-backed and specific

---

## Day 5: Dense Jacobi Cross-Lane Validation

**Title:** Jacobi Validation
**Theme:** Prove the dense Jacobi decision across all focused eigensolver lanes
**Time estimate:** 12 hours

### Tasks
1. Run the focused eigensolver tests from Day 3.
2. Inspect any changed compile commands or source-list generated outputs.
3. Verify Makefile and CMake parity for the dense helper source or deferral.
4. Check that reviewed CTest counts and names did not drift.
5. Update working notes with validation outputs and residual risks.
6. Write the dense Jacobi validation artifact.

### Deliverables
- focused eigensolver validation results
- source-list parity evidence
- CTest no-drift evidence
- dense Jacobi validation artifact
- residual risk notes

### Completion Criteria
- all focused eigensolver validation passes
- build-system parity is documented
- no reviewed test-count or target drift is present
- the dense Jacobi workstream is closed or explicitly deferred

---

## Day 6: Grow-M, Refinement & Shared-Kernel Audit

**Title:** Eigensolver Audit 1
**Theme:** Audit behavior-sensitive eigensolver boundaries that should not move
without stronger evidence
**Time estimate:** 12 hours

### Tasks
1. Inspect grow-m refinement code paths and convergence/refinement helpers.
2. Inspect shared Lanczos and dense spectral kernel dependencies.
3. Identify behavior-sensitive state, workspace, and tolerance assumptions.
4. Classify each candidate as safe-to-split-later, no-go, or needs more proof.
5. Add focused evidence for why no additional movement happens in Sprint 109.
6. Write the Day 6 eigensolver behavior audit artifact.

### Deliverables
- grow-m/refinement dependency map
- shared-kernel boundary classification
- no-go and future-proof notes
- Day 6 behavior audit artifact

### Completion Criteria
- unsafe movement is blocked by evidence, not instinct
- future extraction candidates have explicit prerequisites
- no code movement occurs outside the approved dense Jacobi boundary

---

## Day 7: Dispatch, Handle & Shift-Invert Audit

**Title:** Eigensolver Audit 2
**Theme:** Finish eigensolver behavior-sensitive boundary review around public
workflow glue
**Time estimate:** 12 hours

### Tasks
1. Audit dispatch/default behavior for grow-m, thick-restart, and LOBPCG lanes.
2. Audit public handle and workspace glue for ownership and reuse assumptions.
3. Audit shift-invert paths and direct-solver interactions.
4. Check whether any future source owner would require public contract wording.
5. Capture no-go conditions for movement that would hide behavior proofs.
6. Write the dispatch/handle/shift-invert audit artifact.

### Deliverables
- dispatch/default behavior map
- handle/workspace ownership notes
- shift-invert source-boundary notes
- future extraction no-go list
- Day 7 behavior audit artifact

### Completion Criteria
- behavior-sensitive eigensolver seams are documented with evidence
- no future extraction depends on an undocumented public behavior assumption
- the eigensolver audit can support downstream planning without code churn

---

## Day 8: Matrix Shell Candidate Boundary Contract

**Title:** Matrix Boundary
**Theme:** Choose one future `src/sparse_matrix.c` public-behavior owner without
moving matrix-shell code prematurely
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 108 matrix-shell public-behavior review.
2. Inventory current `src/sparse_matrix.c` responsibility clusters.
3. Choose one future public-behavior owner candidate for a later sprint.
4. Document private-header dependencies and source-list requirements.
5. Define focused public behavior tests and solver-smoke gates.
6. Write the matrix-shell candidate boundary contract.

### Deliverables
- `src/sparse_matrix.c` responsibility cluster map
- selected future source-owner candidate
- private-header dependency notes
- focused behavior-test and solver-smoke gate list
- matrix-shell boundary contract

### Completion Criteria
- one future matrix-shell owner is selected and bounded
- no matrix-shell code moves unless independently proven low risk
- public behavior and solver-smoke requirements are explicit
- the contract is usable by a later implementation sprint

---

## Day 9: Matrix Shell Validation & No-Move Decision

**Title:** Matrix Contract Proof
**Theme:** Validate the matrix-shell contract against public behavior and solver
workflow expectations
**Time estimate:** 12 hours

### Tasks
1. Run or identify the focused matrix public behavior tests selected on Day 8.
2. Inspect solver smoke coverage that depends on matrix construction, mutation,
   norms, transpose, copy, and factored-state behavior.
3. Verify the selected future owner does not require public API or install
   header changes.
4. Record source-list and private-header requirements for a later split.
5. Publish a no-move decision for Sprint 109 unless all low-risk conditions are
   independently satisfied.

### Deliverables
- matrix-shell validation notes
- solver-smoke dependency map
- public API/install-header no-drift notes
- Sprint 109 move/no-move decision

### Completion Criteria
- the matrix-shell contract is evidence-backed
- public behavior remains the proof owner
- no matrix-shell movement occurs without explicit low-risk evidence
- downstream Sprint 110+ planning can consume the contract

---

## Day 10: Giant-Test Cleanup Candidate Selection

**Title:** Cleanup Boundary
**Theme:** Select one bounded proof-owner cleanup family across the remaining
large tests
**Time estimate:** 12 hours

### Tasks
1. Re-inventory residual cleanup candidates in:
   - `tests/test_ldlt_csc.c`
   - `tests/test_qr.c`
   - `tests/test_iterative.c`
   - `tests/test_svd.c`
2. Exclude Sprint 108 helper families and already-cleaned patterns.
3. Rank candidates by proof clarity, review size, validation cost, and failure
   localization.
4. Select one bounded cleanup batch or explicitly defer all candidates.
5. Define call-site proof-visibility rules before edits.
6. Write the giant-test cleanup boundary artifact.

### Deliverables
- residual giant-test candidate inventory
- duplicate-work exclusion list
- selected cleanup batch or deferral
- proof-visibility rules
- focused validation plan

### Completion Criteria
- the selected batch fits one reviewable cleanup family
- no new compiled helper target is needed
- proof assertions remain visible at call sites
- validation scope is known before edits begin

---

## Day 11: Giant-Test Cleanup Follow-Through

**Title:** Cleanup Follow-Through
**Theme:** Implement the approved proof-owner cleanup while preserving proof
intent
**Time estimate:** 12 hours

### Tasks
1. Implement only the Day 10 approved cleanup batch.
2. Keep assertion specificity and failure-localization intact.
3. Avoid moving proof logic into broad or generic helpers.
4. Update nearby comments or helper names only when they clarify ownership.
5. Run focused validation for touched tests.
6. Capture before/after file, helper, and call-site metrics.

### Deliverables
- bounded giant-test cleanup change or explicit no-change deferral
- focused validation notes
- before/after maintainability metrics
- remaining cleanup residuals

### Completion Criteria
- focused tests pass if code changed
- proof intent remains readable at updated call sites
- no helper-target or CTest registration drift occurs
- remaining cleanup is queued or rejected explicitly

---

## Day 12: Focused Integration & Drift Check

**Title:** Focused Integration
**Theme:** Validate all Sprint 109 code and build-system changes before final
closeout
**Time estimate:** 12 hours

### Tasks
1. Run focused validation for every touched code, test, and build-system
   surface.
2. Verify public API and install-header no-drift.
3. Verify source-list parity between Makefile and CMake if source files changed.
4. Verify helper-target and reviewed CTest registration no-drift.
5. Re-run targeted solver smoke lanes touched by eigensolver, matrix-shell, or
   giant-test work.
6. Write the focused integration and drift-check artifact.

### Deliverables
- focused integration validation results
- public API/install-header no-drift evidence
- source-list parity evidence
- CTest/helper-target no-drift evidence
- Day 12 validation artifact

### Completion Criteria
- all focused checks pass
- every touched surface has an explicit validation result
- no accidental public support-surface change is present
- unresolved validation gaps are known before full closeout

---

## Day 13: Full Quality Gate & Metrics

**Title:** Full Validation
**Theme:** Run required quality checks and capture maintainability metrics for
Sprint 109
**Time estimate:** 12 hours

### Tasks
1. Run `make format` if code or formatted sources changed.
2. Run `make lint` if code or build-system source lists changed.
3. Run `make test` if any `.c` or `.h` files changed.
4. Run docs-only checks when only planning or artifact files changed.
5. Capture size, helper, source-list, and validation metrics.
6. Update Sprint 109 working notes with full quality-gate results.

### Deliverables
- full quality-gate output summary
- maintainability metrics
- validation gap list, if any
- updated working notes

### Completion Criteria
- all required checks pass before closeout begins
- metrics cover every changed owner
- no quality failure is deferred silently
- validation evidence is ready for the retrospective

---

## Day 14: Sprint 109 Closeout & Residual Queue

**Title:** Residual Closeout
**Theme:** Publish Sprint 109 outcomes, residuals, and downstream handoff for
shifted Epic 10 sprints
**Time estimate:** 12 hours

### Tasks
1. Reconcile Sprint 109 outcomes against all seven project-plan items.
2. Confirm dense Jacobi extraction or deferral status.
3. Confirm eigensolver behavior-sensitive no-go conditions.
4. Confirm matrix-shell future owner contract status.
5. Confirm giant-test cleanup outcome and remaining proof-owner debt.
6. Publish residuals for shifted downstream sprints without duplicating
   completed work.
7. Prepare Sprint 109 closeout notes for the retrospective.

### Deliverables
- Sprint 109 item-by-item closeout
- residual queue for downstream sprints
- no-duplicate completed-work list
- validation and metrics closeout artifact
- retrospective-ready notes

### Completion Criteria
- every Sprint 109 deliverable has a completed, deferred, or rejected status
- downstream residuals are dependency-ordered
- no sprint-exit claim exceeds validation evidence
- Sprint 109 is ready for retrospective creation
