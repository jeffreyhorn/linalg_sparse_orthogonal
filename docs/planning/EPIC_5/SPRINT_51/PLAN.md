# Sprint 51 Plan: Public Direct-Solver Lifecycle API Phase 1

**Sprint Duration:** 14 days  
**Goal:** Land the first public direct-solver lifecycle API batch for the main
factor-and-solve workflows while preserving one-shot compatibility wrappers.
This sprint implements the Sprint 51 section of
`docs/planning/EPIC_5/PROJECT_PLAN.md`.

**Starting Point:** Sprint 50 closed with the direct-solver lifecycle contract
frozen around the existing `sparse_analysis_t` / `sparse_factors_t`
analysis/factor/refactor path, explicit non-goals and compatibility fences,
and a fixed landing order for headers, implementation, adoption, compatibility
sweep, and validation. The remaining work is now implementation-shaped rather
than architectural: public headers must be extended, LU / Cholesky / LDL^T
must route through the bounded lifecycle path where appropriate, one-shot
entries must remain first-class, and focused regression coverage must prove the
new public story.

**End State:** Sprint 51 leaves behind the first implemented public
direct-solver lifecycle API batch across the main direct families, preserved
one-shot compatibility wrappers for the touched paths, focused regression
coverage for the new lifecycle surface, and validated implementation artifacts
that hand off a stable phase-1 end state to Sprint 52.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
148 hours, matching the Sprint 51 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 51 Scope Audit & Implementation Baseline

**Title:** Baseline Setup  
**Theme:** Convert the Sprint 51 project-plan items plus the Sprint 50 handoff
package into a bounded implementation map  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 51 section of `docs/planning/EPIC_5/PROJECT_PLAN.md`,
   the Sprint 50 closeout artifact, the Sprint 50 retrospective, and the
   highest-value Sprint 50 design artifacts.
2. Reconfirm the preserved Sprint 51 constraints:
   - keep one-shot LU / Cholesky / LDL^T APIs first-class
   - preserve the Sprint 50 non-goal and compatibility fence
   - center the existing analysis/factor/refactor path instead of inventing a
     broad new generic direct handle
   - preserve the strongest local reviewed baseline and parity claims
3. Define the Sprint 51 implementation workstreams explicitly:
   - public header surface
   - LU lifecycle integration
   - Cholesky lifecycle integration
   - LDL^T lifecycle integration
   - wrapper preservation
   - regression expansion
   - validation and closeout
4. Record the highest-risk seams for the sprint:
   - mutable `SparseMatrix` compatibility behavior in one-shot LU/Cholesky
   - additive API wording drift across direct headers
   - reuse/refactor semantics becoming overstated
   - behavior drift between one-shot and lifecycle entry points
5. Open Sprint 51 working notes and record starting assumptions, landing
   order, and touched-surface expectations.

### Deliverables
- Sprint 51 scope inventory
- Implementation baseline notes
- Working-notes starting assumptions

### Completion Criteria
- Sprint 51 starts from the Sprint 50 contract rather than re-opening API
  design
- Preserved compatibility and validation constraints are explicit before code
  edits begin
- The implementation workstreams are named before artifacts and patches land

---

## Day 2: Validation Baseline & Touched-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed local baseline and the exact rerun set
Sprint 51 code days must preserve  
**Time estimate:** 8 hours

### Tasks
1. Reconfirm the maintained reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity
   - current truthfulness-anchor counts
2. Reconfirm the mandatory gate for any later `*.c` / `*.h` lifecycle batch:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial public API batches:
   - `make quality-review-full`
4. Refresh the targeted follow-on binaries Sprint 51 is most likely to need:
   - `./build/example_analysis`
   - `./build/bench_refactor`
   - `./build/bench_refactor_csc`
   - `./build/test_cholesky`
   - `./build/test_ldlt`
   - `./build/test_etree`
   - `./build/test_chol_csc`
   - `./build/test_ldlt_csc`
5. Record the smallest authoritative validation boundary for docs-only notes
   versus code-touch implementation days.

### Deliverables
- Refreshed validation/truthfulness notes
- Sprint 51 touched-surface rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 51 uses the same baseline wording and parity anchors as the live repo
- The authoritative rerun set is explicit before header/source integration
- No validation ambiguity remains around the public API landing days

---

## Day 3: Public Header Surface Design-To-Code Map

**Title:** Header Map  
**Theme:** Turn the Sprint 50 public lifecycle contract into a concrete header
edit map across the direct-solver families  
**Time estimate:** 8 hours

### Tasks
1. Re-read and classify the live direct public headers:
   - `include/sparse_analysis.h`
   - `include/sparse_lu.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_ldlt.h`
2. Map which declarations, comments, examples, and cross-references must land
   in each header for phase 1.
3. Decide what stays analysis/factors-centric in shared vocabulary versus what
   remains family-local in LU / Cholesky / LDL^T docs.
4. Separate the true first header batch from later documentation-only
   follow-ons.
5. Write the public-header landing artifact.

### Deliverables
- Header-by-header landing map
- Shared-vs-family-local contract notes
- Phase-1 header batch boundary

### Completion Criteria
- The header implementation target is reduced to named edits instead of a broad
  “touch the direct headers” instruction
- Shared lifecycle vocabulary and family-local wording boundaries are explicit
- Sprint 51 can start header edits without re-deriving the contract

---

## Day 4: Header/API Batch I

**Title:** Header Batch  
**Theme:** Land the first public direct-solver lifecycle declarations and
caller contract wording  
**Time estimate:** 12 hours

### Tasks
1. Add the bounded phase-1 lifecycle declarations and comments to the touched
   direct-solver headers.
2. Keep the shared repeated-run contract explicit around:
   - zero/init
   - analyze once
   - factor / solve
   - refactor / solve many
   - free
3. Preserve the one-shot direct family entries as first-class peer APIs.
4. Ensure the header comments do not overpromise:
   - no raw internal storage exposure
   - no generic direct-handle redesign
   - no promise that reuse preserves old numeric factor state
5. Run the full required gate and stronger reviewed baseline:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- First public direct lifecycle header/API batch
- Updated header contract wording
- Validation output for the header batch

### Completion Criteria
- The public header phase-1 contract is live in the repo
- One-shot compatibility wording remains explicit and intact
- All required validation passes before moving to source integration

---

## Day 5: LU Lifecycle Integration

**Title:** LU Integration  
**Theme:** Route LU through the bounded lifecycle path while preserving the
existing one-shot caller story  
**Time estimate:** 12 hours

### Tasks
1. Audit the live LU implementation and identify the narrowest integration seam
   that matches the new public lifecycle surface.
2. Land the LU lifecycle integration through the analysis/factor/refactor path
   where appropriate.
3. Preserve one-shot LU behavior and matrix-copy expectations for callers that
   stay on the simple/default path.
4. Add or adjust internal comments only where the lifecycle routing would be
   unclear to maintainers.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched follow-ons justified by the LU surface

### Deliverables
- LU lifecycle integration patch
- Preserved one-shot LU wrapper behavior
- Validation output for the LU batch

### Completion Criteria
- LU uses the bounded lifecycle path without reopening the Sprint 50 contract
- One-shot LU callers still have the same supported public story
- Required validation passes before moving to Cholesky

---

## Day 6: Cholesky Lifecycle Integration

**Title:** Cholesky Integration  
**Theme:** Wire the phase-1 lifecycle path through Cholesky without breaking
the one-shot compatibility surface  
**Time estimate:** 12 hours

### Tasks
1. Audit the live Cholesky path and identify the narrowest lifecycle-routing
   seam consistent with the Sprint 50 design.
2. Land the Cholesky lifecycle integration through the analysis/factor/refactor
   path where appropriate.
3. Preserve the one-shot copied-matrix teaching and compatibility behavior.
4. Reconfirm any backend/reorder/telemetry fields touched by the new routing
   still behave consistently.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched follow-ons justified by the Cholesky surface

### Deliverables
- Cholesky lifecycle integration patch
- Preserved one-shot Cholesky wrapper behavior
- Validation output for the Cholesky batch

### Completion Criteria
- Cholesky now participates in the bounded public lifecycle story
- The copied-matrix one-shot compatibility path remains honest and supported
- Required validation passes before LDL^T work begins

---

## Day 7: LDL^T Lifecycle Integration

**Title:** LDLT Integration  
**Theme:** Extend the same bounded lifecycle path to LDL^T and keep the direct
family contracts aligned  
**Time estimate:** 12 hours

### Tasks
1. Audit the LDL^T public and implementation seams relevant to the new
   lifecycle path.
2. Land the matching phase-1 lifecycle integration for LDL^T.
3. Preserve the LDL^T factor-object and identity-permutation expectations
   already present on the one-shot side.
4. Recheck shared direct lifecycle wording for drift across LU, Cholesky, and
   LDL^T.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched follow-ons justified by the LDL^T surface

### Deliverables
- LDL^T lifecycle integration patch
- Direct-family contract-alignment notes
- Validation output for the LDL^T batch

### Completion Criteria
- LDL^T is aligned with the same phase-1 repeated-run direct story
- Family-specific direct-solver semantics remain explicit rather than flattened
- Required validation passes before wrapper cleanup

---

## Day 8: Wrapper Preservation Batch

**Title:** Wrapper Routing  
**Theme:** Preserve one-shot direct entry points by routing them through the
new lifecycle path where appropriate  
**Time estimate:** 12 hours

### Tasks
1. Audit the touched one-shot public entries and decide which can cleanly route
   through the new lifecycle path.
2. Land the wrapper-preservation cleanup for LU / Cholesky / LDL^T.
3. Reconfirm that one-shot direct usage still reads as the simple/default path
   for one-off solves.
4. Reconfirm that wrapper routing does not overstate reuse semantics or hide
   family-local compatibility behavior.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- Compatibility-preserving one-shot wrapper routing
- Wrapper-behavior validation notes
- Validation output for the wrapper batch

### Completion Criteria
- One-shot public entries remain first-class and supported
- The lifecycle path is real behind the touched wrappers where appropriate
- All required validation passes before regression expansion

---

## Day 9: Focused Regression Expansion Design & Inventory

**Title:** Test Inventory  
**Theme:** Identify the smallest high-signal regression additions needed for
the new public lifecycle surface  
**Time estimate:** 8 hours

### Tasks
1. Audit the current direct-solver regression surfaces:
   - `tests/test_cholesky.c`
   - `tests/test_ldlt.c`
   - `tests/test_etree.c`
   - `tests/test_chol_csc.c`
   - `tests/test_ldlt_csc.c`
2. Identify the direct public lifecycle behaviors that are not yet covered:
   - zero/init expectations
   - analyze/factor/refactor/solve sequencing
   - one-shot parity through wrapper routing
   - rejected misuse or invalid-state cases where justified
3. Separate mandatory phase-1 coverage from attractive but out-of-scope larger
   test refactors.
4. Write the focused regression-expansion artifact.

### Deliverables
- Focused lifecycle regression inventory
- Mandatory-vs-later test split
- Test landing map for Day 10

### Completion Criteria
- Regression expansion is reduced to a bounded target list
- The new lifecycle surface has a clear direct-test plan before edits land
- Sprint 51 avoids broad test churn unrelated to phase 1

---

## Day 10: Focused Regression Expansion Batch

**Title:** Regression Batch  
**Theme:** Add direct public lifecycle coverage for the touched direct-solver
families  
**Time estimate:** 12 hours

### Tasks
1. Add the bounded direct lifecycle regression tests identified on Day 9.
2. Cover the highest-value public behaviors across the touched families:
   - zero/init safety where applicable
   - analyze/factor/refactor/solve sequencing
   - one-shot wrapper parity for the touched paths
3. Keep the additions narrowly scoped to phase-1 lifecycle behavior.
4. Run:
   - `make format`
   - `make lint`
   - `make test`
   - touched direct-solver follow-on binaries

### Deliverables
- Focused direct lifecycle regression additions
- Updated direct-solver behavior coverage
- Validation output for the regression batch

### Completion Criteria
- The new public lifecycle path has direct regression proof
- Added tests stay bounded to the touched direct-solver families
- Required validation passes before docs/example adoption

---

## Day 11: High-Signal Example / Benchmark Adoption

**Title:** Adoption Batch  
**Theme:** Update the strongest repeated-run direct example and benchmark
surfaces to match the live lifecycle API  
**Time estimate:** 12 hours

### Tasks
1. Update `examples/example_analysis.c` if needed to reflect the landed public
   lifecycle path cleanly.
2. Update `benchmarks/bench_refactor.c` and the smallest related benchmark/docs
   surface justified by the implementation.
3. Fix the concrete carried-forward doc drifts if they are touched naturally:
   - `benchmarks/README.md` benchmark labeling
   - `examples/README.md` `example_analysis` omission
4. Keep the adoption batch bounded to the strongest repeated-run direct
   surfaces, not broad example/tutorial churn.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `./build/example_analysis`
   - touched benchmark binaries

### Deliverables
- Updated repeated-run direct example/benchmark surfaces
- Narrow docs alignment for touched adoption files
- Validation output for the adoption batch

### Completion Criteria
- The strongest shipped repeated-run direct example matches the live API
- The benchmark caller story aligns with the public lifecycle path
- Adoption remains bounded to the Sprint 50 target surfaces

---

## Day 12: Post-Landing Compatibility Audit

**Title:** Compatibility Audit  
**Theme:** Re-audit the landed Sprint 51 public lifecycle surface against the
Sprint 50 contract and compatibility fence  
**Time estimate:** 8 hours

### Tasks
1. Re-read the touched headers, source surfaces, tests, and adoption files from
   the perspective of the Sprint 50 contract.
2. Confirm that the live landed behavior still matches the intended rules:
   - one-shot entries remain first-class
   - repeated direct-run path stays analysis/factors-centric
   - reuse does not overpromise old numeric state preservation
   - non-goals remain closed
3. Identify any small residual drift that must be fixed before final validation.
4. Write the post-landing compatibility audit artifact.

### Deliverables
- Post-landing compatibility audit
- Residual-drift list if any
- Pre-validation checklist

### Completion Criteria
- The landed code is checked against the Sprint 50 fence, not only against
  compiler/test success
- Any remaining residual drift is explicit before Day 13
- No hidden scope creep remains in the landed API surface

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the authoritative validation closeout for the implemented phase-1
direct lifecycle surface  
**Time estimate:** 12 hours

### Tasks
1. Run the full required gate from the final landed state:
   - `make format`
   - `make lint`
   - `make test`
2. Run the stronger reviewed baseline:
   - `make quality-review-full`
3. Reconfirm the truthfulness anchors:
   - reviewed CMake `ctest -N`
   - Makefile/CMake parity
   - full reviewed CMake pass state
4. Run the targeted Sprint 51 follow-ons:
   - `./build/example_analysis`
   - `./build/bench_refactor`
   - `./build/bench_refactor_csc`
   - `./build/test_cholesky`
   - `./build/test_ldlt`
   - `./build/test_etree`
   - `./build/test_chol_csc`
   - `./build/test_ldlt_csc`
5. Record the final validation artifact and measured close state.

### Deliverables
- Full validation-sweep artifact
- Measured close-state metrics
- Final rerun evidence for touched binaries

### Completion Criteria
- All required validation passes from the final landed state
- The reviewed baseline and parity anchors remain truthful
- Sprint 51 has a measured close state rather than a claimed one

---

## Day 14: Closeout & Sprint 52 Handoff

**Title:** Closeout  
**Theme:** Consolidate Sprint 51’s landed lifecycle phase-1 state and hand the
remaining queue to Sprint 52 cleanly  
**Time estimate:** 12 hours

### Tasks
1. Summarize the landed Sprint 51 public lifecycle batch:
   - header/API surface
   - LU / Cholesky / LDL^T integration
   - one-shot wrapper preservation
   - focused regression coverage
   - adoption surfaces
2. Record the exact preserved compatibility and non-goal boundaries after the
   implementation sprint.
3. Identify the smallest real remaining queue for Sprint 52 rather than
   restating broad Epic 5 ambitions.
4. Check whether the Epic 5 project plan needs a bounded wording refresh based
   on the actual landed phase-1 state.
5. Write the closeout/handoff artifact and update working notes.

### Deliverables
- Sprint 51 closeout artifact
- Sprint 52 handoff notes
- Final Sprint 51 working-notes synthesis

### Completion Criteria
- Sprint 51 closes from the measured Day 13 baseline
- The remaining queue is handed forward as a small explicit list
- The sprint ends with a coherent implemented phase-1 package rather than only
  a list of merged patches
