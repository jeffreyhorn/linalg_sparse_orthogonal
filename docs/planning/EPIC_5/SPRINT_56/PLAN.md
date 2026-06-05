# Sprint 56 Plan: Large-Source Decomposition Phase 2

**Sprint Duration:** 14 days  
**Goal:** Continue large-source hotspot reduction across the remaining CSC
direct-solver and dense-algorithm implementation files without reopening the
validated public contracts preserved through Sprints 50-55.
This sprint implements the Sprint 56 section of
`docs/planning/EPIC_5/PROJECT_PLAN.md`.

**Starting Point:** Sprint 55 closed with the first bounded decomposition
phase complete:
- `src/sparse_eigs.c` reduced through LOBPCG and thick-restart extraction
- `src/sparse_iterative.c` reduced through MINRES extraction
- public one-shot and repeated-run solver contracts unchanged
- reviewed validation parity still exact

The strongest remaining maintainability hotspots now shift to the large CSC
direct-solver production files plus the SVD implementation:
- `src/sparse_ldlt_csc.c`
- `src/sparse_chol_csc.c`
- `src/sparse_svd.c`

**End State:** Sprint 56 leaves behind smaller and clearer CSC direct-solver
and SVD ownership, with bounded helper/module extractions landed, touched
comments and coupled wording normalized, and the maintained validation
baseline still exact.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
148 hours, matching the Sprint 56 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 56 Scope Audit & Residual Hotspot Baseline

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 56 project-plan items plus the Sprint 55 close state
into a bounded CSC/SVD decomposition map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 56 section of `docs/planning/EPIC_5/PROJECT_PLAN.md`,
   the Sprint 55 closeout, and the Epic 5 review/todo notes related to
   remaining large production-file ownership.
2. Reconfirm the preserved Sprint 56 constraints:
   - no public API redesign
   - no reopening the direct lifecycle support boundary
   - decomposition-first, not feature-first
   - preserve reviewed validation and truthfulness anchors
3. Define the Sprint 56 workstreams explicitly:
   - `sparse_ldlt_csc.c` residual audit
   - LDLT CSC decomposition batch
   - `sparse_chol_csc.c` residual audit
   - Cholesky CSC decomposition batch
   - `sparse_svd.c` maintainability batch
   - touched-doc and comment reconciliation
   - validation and closeout
4. Record the highest-risk seams for the sprint:
   - extracting code that still carries dense shared state implicitly
   - splitting files mechanically without sharpening ownership
   - creating mismatched Makefile/CMake surfaces
   - behavior drift in CSC-native versus wrapper entry points
5. Open Sprint 56 working notes and record the initial landing order.

### Deliverables
- Sprint 56 scope inventory
- CSC/SVD baseline notes
- Working-notes starting assumptions

### Completion Criteria
- Sprint 56 starts from the Sprint 55 validated decomposition state rather
  than reopening public design questions
- Preserved compatibility and scope fences are explicit before seam audits or
  code patches land
- The remaining large-source workstreams are named before implementation begins

---

## Day 2: Validation Baseline & Touched-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed local baseline and the exact rerun set Sprint
56 code days must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the maintained reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity
   - current truthfulness-anchor counts
2. Reconfirm the mandatory gate for later `*.c` / `*.h` decomposition batches:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial ownership batches:
   - `make quality-review-full`
4. Refresh the targeted Sprint 56 follow-on binaries most likely to be needed:
   - `./build/test_chol_csc`
   - `./build/test_ldlt_csc`
   - `./build/test_cholesky`
   - `./build/test_ldlt`
   - `./build/test_etree`
   - `./build/test_svd`
   - `./build/test_integration`
   - `./build/bench_refactor_csc`
   - `./build/example_analysis`
5. Record the authoritative validation boundary for docs-only audit/design days
   versus implementation landing days.

### Deliverables
- Refreshed validation/truthfulness notes
- Sprint 56 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 56 uses the same baseline wording and parity anchors as the live repo
- The authoritative CSC/SVD rerun set is explicit before decomposition work
  begins
- No validation ambiguity remains around implementation-extraction days

---

## Day 3: `sparse_ldlt_csc.c` Residual Ownership Audit

**Title:** LDLT CSC Audit  
**Theme:** Reduce `src/sparse_ldlt_csc.c` to concrete extraction seams before
code movement begins  
**Time estimate:** 10 hours

### Tasks
1. Audit the live ownership inside `src/sparse_ldlt_csc.c` and separate:
   - native storage/container management
   - symmetric swap and pivot-selection machinery
   - scalar elimination and solve paths
   - supernodal helper clusters
   - wrapper/validation seams
2. Identify which seams can be extracted without changing the public LDLT CSC
   behavior or dispatch contract.
3. Rank the candidate seams by:
   - ownership clarity
   - behavioral risk
   - line-count reduction value
   - proof cost across CSC-specific tests
4. Reject seams that only move code mechanically without clarifying long-term
   ownership.
5. Write the seam audit artifact and the ranked landing order.

### Deliverables
- `sparse_ldlt_csc.c` seam audit
- Ranked LDLT CSC extraction targets
- Proposed first LDLT CSC extraction boundary

### Completion Criteria
- The LDLT CSC decomposition problem is reduced to named ownership seams
- The first extraction target is justified by maintainability, not only line
  count
- Sprint 56 can start LDLT CSC implementation work from a concrete map

---

## Day 4: LDLT CSC Decomposition Design

**Title:** LDLT CSC Design  
**Theme:** Freeze the first bounded LDLT CSC extraction boundary before editing
permanent implementation files  
**Time estimate:** 12 hours

### Tasks
1. Select the Day 3 highest-value LDLT CSC seam for the first landing.
2. Define the exact source-file ownership split:
   - what remains in `src/sparse_ldlt_csc.c`
   - what moves into the new helper/module file
   - what private declarations stay in the current internal header surface
3. Define the invariants the extraction must preserve:
   - native versus wrapper behavior
   - permutation and pivot-size semantics
   - residual and inertia parity
   - current CSC test and benchmark proof
4. Define the minimal comment policy for the touched implementation:
   - preserve algorithm truth
   - remove stale sprint-history prose where encountered
5. Record the design artifact and landing checklist.

### Deliverables
- LDLT CSC extraction design
- File-boundary ownership map
- Extraction invariants and checklist

### Completion Criteria
- The first LDLT CSC extraction boundary is explicit before code movement
- Ownership is defined at file and helper level, not just conceptually
- Comment-cleanup expectations are fixed before touched code is rewritten

---

## Day 5: LDLT CSC Decomposition Batch

**Title:** LDLT CSC Batch  
**Theme:** Land the first bounded ownership extraction from
`src/sparse_ldlt_csc.c`  
**Time estimate:** 12 hours

### Tasks
1. Extract the selected LDLT CSC helper/module slice and wire it into the
   remaining orchestration layer.
2. Add or update any needed private internal declarations.
3. Keep the public API, dispatch behavior, tests, and benchmark results
   unchanged.
4. Remove stale sprint-history narrative from touched LDLT CSC implementation
   blocks while preserving useful algorithm commentary.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- Landed LDLT CSC extraction patch
- Updated private declarations
- Reduced touched-file narrative noise

### Completion Criteria
- A real ownership seam is extracted from `src/sparse_ldlt_csc.c`
- The remaining orchestration file is smaller and clearer than before
- Full required validation passes after the extraction

---

## Day 6: `sparse_chol_csc.c` Residual Ownership Audit

**Title:** Cholesky CSC Audit  
**Theme:** Reduce `src/sparse_chol_csc.c` to concrete extraction seams using
the landed LDLT CSC state as a comparison point  
**Time estimate:** 10 hours

### Tasks
1. Audit the live ownership inside `src/sparse_chol_csc.c` and separate:
   - native storage/container management
   - scalar elimination and solve paths
   - supernodal helper clusters
   - writeback and validation seams
   - dispatch-specific glue
2. Identify which seams can be extracted cleanly without changing the Cholesky
   CSC dispatch contract.
3. Compare the Cholesky CSC seam map against the landed LDLT CSC split and
   note intentional family differences.
4. Rank the candidate seams by ownership clarity, risk, and proof cost.
5. Write the seam audit artifact and ranked landing order.

### Deliverables
- `sparse_chol_csc.c` seam audit
- Ranked Cholesky CSC extraction targets
- Proposed first Cholesky CSC extraction boundary

### Completion Criteria
- The Cholesky CSC decomposition problem is reduced to named ownership seams
- Family-specific differences are recorded instead of forcing fake symmetry
- Sprint 56 can start Cholesky CSC implementation work from a concrete map

---

## Day 7: Cholesky CSC Decomposition Design

**Title:** Cholesky CSC Design  
**Theme:** Freeze the first bounded Cholesky CSC extraction boundary before
code movement begins  
**Time estimate:** 12 hours

### Tasks
1. Select the Day 6 highest-value Cholesky CSC seam for the first landing.
2. Define the exact file ownership split:
   - what remains in `src/sparse_chol_csc.c`
   - what moves into a new owned helper/module file
   - what declarations stay in the current internal-header surface
3. Define the invariants the extraction must preserve:
   - scalar/supernodal parity
   - writeback semantics
   - dispatch and threshold behavior
   - CSC-specific regression and benchmark proof
4. Fix the bounded non-goal fence:
   - no broad CSC header taxonomy redesign
   - no dispatch redesign
   - no public API expansion
5. Record the design artifact and landing checklist.

### Deliverables
- Cholesky CSC extraction design
- File-boundary ownership map
- Extraction invariants and checklist

### Completion Criteria
- The first Cholesky CSC extraction boundary is explicit before code movement
- Ownership is defined concretely at file/helper level
- The non-goal fence is fixed before implementation starts

---

## Day 8: Cholesky CSC Decomposition Batch

**Title:** Cholesky CSC Batch  
**Theme:** Land the first bounded ownership extraction from
`src/sparse_chol_csc.c`  
**Time estimate:** 12 hours

### Tasks
1. Extract the selected Cholesky CSC helper/module slice and wire it into the
   remaining orchestration layer.
2. Add or update any needed private internal declarations.
3. Keep the public API, dispatch behavior, tests, and benchmark results
   unchanged.
4. Remove stale sprint-history narrative from touched Cholesky CSC
   implementation blocks while preserving durable algorithm commentary.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- Landed Cholesky CSC extraction patch
- Updated private declarations
- Reduced touched-file narrative noise

### Completion Criteria
- A real ownership seam is extracted from `src/sparse_chol_csc.c`
- The remaining orchestration file is smaller and clearer than before
- Full required validation passes after the extraction

---

## Day 9: `sparse_svd.c` Maintainability Audit

**Title:** SVD Audit  
**Theme:** Reduce `src/sparse_svd.c` to a bounded maintainability target
instead of an open-ended cleanup bucket  
**Time estimate:** 10 hours

### Tasks
1. Audit the live ownership inside `src/sparse_svd.c` and separate:
   - public orchestration
   - bidiagonalization helpers
   - rank-k / low-rank helper clusters
   - dense reconstruction and reporting helpers
2. Identify the highest-value maintainability improvement that can land this
   sprint without broad redesign.
3. Decide whether the SVD batch should emphasize:
   - helper extraction
   - internal helper regrouping
   - bounded file-local cleanup with ownership clarification
4. Rank the candidate SVD seams by clarity, risk, and proof cost.
5. Write the audit artifact and chosen landing direction.

### Deliverables
- `sparse_svd.c` seam audit
- Chosen SVD maintainability target
- SVD landing direction for Day 10

### Completion Criteria
- The SVD maintainability batch has a bounded target instead of a vague
  cleanup scope
- The selected target is justified by ownership clarity, not only line count
- Day 10 can proceed from an explicit landed-state plan

---

## Day 10: SVD Maintainability Batch

**Title:** SVD Batch  
**Theme:** Land the bounded SVD ownership/maintainability improvement selected
on Day 9  
**Time estimate:** 12 hours

### Tasks
1. Implement the chosen `src/sparse_svd.c` maintainability improvement.
2. Update any touched private declarations or local helper ordering as needed.
3. Keep public SVD behavior, tests, and examples unchanged.
4. Remove stale sprint-history narrative from touched SVD implementation
   blocks while preserving useful algorithm commentary.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full` if the batch is substantial enough to warrant
     the stronger reviewed baseline

### Deliverables
- Landed SVD maintainability patch
- Updated helper/declaration ownership
- Reduced touched-file narrative noise

### Completion Criteria
- The selected SVD maintainability target lands without reopening design scope
- Ownership is clearer than before in the touched SVD area
- Required validation passes after the batch

---

## Day 11: Touched-Doc and Comment Reconciliation

**Title:** Reconciliation Sweep  
**Theme:** Normalize touched implementation comments and any coupled wording
after the LDLT CSC, Cholesky CSC, and SVD batches  
**Time estimate:** 12 hours

### Tasks
1. Review touched permanent implementation files for stale sprint-history
   narrative that should not remain in production code.
2. Preserve durable algorithm, ownership, and dispatch commentary while
   removing sprint-local history phrasing.
3. Check whether any tightly coupled internal-header wording now drifts from
   the landed file ownership boundaries.
4. Apply only bounded wording cleanups justified by the landed implementation.
5. If `*.c` / `*.h` changes are required, run:
   - `make format`
   - `make lint`
   - `make test`

### Deliverables
- Comment/wording reconciliation patch
- Reduced narrative drift in touched implementation files
- Updated coupled wording where needed

### Completion Criteria
- Touched implementation commentary is durable and ownership-focused
- No stale sprint-history prose remains in the Sprint 56 touched permanent code
- Any coupled wording drift is closed without broad docs churn

---

## Day 12: Post-Landing Compatibility Audit

**Title:** Compatibility Audit  
**Theme:** Confirm the landed Sprint 56 decomposition still matches the
preserved public and implementation fences  
**Time estimate:** 8 hours

### Tasks
1. Audit the landed Sprint 56 branch against the preserved constraints:
   - no public API redesign
   - no solver-family support-boundary drift
   - no behavior-visible repeated-run lifecycle drift
2. Confirm the ownership reductions are real and measurable in the touched
   hotspot files.
3. Confirm the Makefile/CMake ownership surfaces stay aligned after the new
   extraction batches.
4. Identify any blocker-level residual drift before final validation.
5. Record the Day 13 validation checklist from the landed state.

### Deliverables
- Post-landing compatibility audit
- Ownership-reduction summary
- Day 13 validation checklist

### Completion Criteria
- The landed Sprint 56 branch still matches the preserved public fences
- Ownership gains are explicit and defensible
- No blocker-level drift remains before final validation

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full Sprint 56 validation contract from the landed
decomposition state  
**Time estimate:** 10 hours

### Tasks
1. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm reviewed CMake parity and truthfulness anchors.
3. Rerun the targeted Sprint 56 follow-ons:
   - `./build/test_chol_csc`
   - `./build/test_ldlt_csc`
   - `./build/test_cholesky`
   - `./build/test_ldlt`
   - `./build/test_etree`
   - `./build/test_svd`
   - `./build/test_integration`
   - `./build/bench_refactor_csc`
   - `./build/example_analysis`
4. Record representative retained behavior and any measurement-sensitive notes.
5. If any failure appears, reconcile it before closeout.

### Deliverables
- Full validation record
- Updated truthfulness-anchor results
- Focused rerun evidence for Sprint 56 touched surfaces

### Completion Criteria
- All required Sprint 56 validation gates pass
- Reviewed parity anchors remain exact
- No unresolved reconciliation queue remains before closeout

---

## Day 14: Closeout and Handoff

**Title:** Closeout  
**Theme:** Summarize Sprint 56’s landed decomposition work, preserved
contracts, validation results, and next bounded queue  
**Time estimate:** 8 hours

### Tasks
1. Write the closeout and handoff artifact from the validated Day 13 state.
2. Summarize:
   - LDLT CSC ownership reduction
   - Cholesky CSC ownership reduction
   - SVD maintainability improvement
   - touched comment/wording normalization
   - preserved public and validation fences
3. Check whether `docs/planning/EPIC_5/PROJECT_PLAN.md` needs any update based
   on actual Sprint 56 outcomes.
4. Record the future-facing residual queue for later decomposition phases.
5. Ensure working notes are internally consistent and ready for retrospective
   creation.

### Deliverables
- Sprint 56 closeout artifact
- Final working-notes synthesis
- Future-facing residual queue

### Completion Criteria
- Sprint 56 closes as one coherent validated decomposition package
- The preserved contract and validation baseline are restated from the final
  landed state
- The next queue is explicit and future-facing rather than a hidden Sprint 56
  defect
