# Sprint 57 Plan: Giant-Test Refactor & Lifecycle Regression Expansion

**Sprint Duration:** 14 days  
**Goal:** Reduce the largest remaining test-maintainability hotspots while
strengthening the final direct lifecycle, repeated-run, and factor-many
regression story without reopening the validated public contracts preserved
through Sprints 50-56.
This sprint implements the Sprint 57 section of
`docs/planning/EPIC_5/PROJECT_PLAN.md`.

**Starting Point:** Sprint 56 closed with the second bounded decomposition
phase complete and the public behavior fence still intact. The largest current
test-maintainability hotspots now dominate the next cleanup queue:
- `tests/test_chol_csc.c`
- `tests/test_svd.c`
- `tests/test_ldlt_csc.c`
- `tests/test_qr.c`
- `tests/test_iterative.c`
- `tests/test_integration.c`

The highest-value next step is not new solver capability. It is targeted test
refactor work that clarifies helper ownership, reduces giant-file friction,
and expands the final lifecycle and factor-many regression proof around the
steady-state direct and repeated-run public surfaces.

**End State:** Sprint 57 leaves behind materially more maintainable major test
surfaces, stronger lifecycle and factor-many regression coverage, and the same
reviewed validation baseline and public behavior truthfulness carried forward
from Sprint 56.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
144 hours, matching the Sprint 57 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 57 Scope Audit & Giant-Test Baseline

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 57 project-plan items plus the Sprint 56 close
state into a bounded giant-test and lifecycle-regression map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 57 section of `docs/planning/EPIC_5/PROJECT_PLAN.md`,
   the Sprint 56 closeout, and the Epic 5 review/todo notes related to giant
   tests and final lifecycle-proof gaps.
2. Reconfirm the preserved Sprint 57 constraints:
   - no public API redesign
   - no reopening the direct-solver lifecycle contract
   - no feature-first expansion disguised as test work
   - preserve reviewed validation and truthfulness anchors
3. Record the live giant-test ranking and likely Sprint 57 hotspots:
   - `tests/test_chol_csc.c`
   - `tests/test_svd.c`
   - `tests/test_ldlt_csc.c`
   - `tests/test_qr.c`
   - `tests/test_iterative.c`
   - `tests/test_integration.c`
4. Define the Sprint 57 workstreams explicitly:
   - large-test audit
   - direct-solver test refactor batch
   - iterative/eigensolver test refactor batch
   - lifecycle regression expansion
   - factor-many / compatibility regression expansion
   - validation and closeout
5. Open Sprint 57 working notes and record the initial landing order and
   touched-surface expectations.

### Deliverables
- Sprint 57 scope inventory
- Giant-test baseline notes
- Working-notes starting assumptions

### Completion Criteria
- Sprint 57 starts from the Sprint 56 validated state rather than reopening
  public design questions
- Preserved compatibility and scope fences are explicit before refactor or
  regression patches land
- The giant-test and lifecycle-regression workstreams are named before
  implementation begins

---

## Day 2: Validation Baseline & Touched-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed local baseline and the exact rerun set
Sprint 57 code days must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the maintained reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity
   - current truthfulness-anchor counts
2. Reconfirm the mandatory gate for later `*.c` / `*.h` test and regression
   batches:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial test-surface changes:
   - `make quality-review-full`
4. Refresh the targeted Sprint 57 follow-on binaries most likely to be needed:
   - `./build/test_chol_csc`
   - `./build/test_ldlt_csc`
   - `./build/test_svd`
   - `./build/test_qr`
   - `./build/test_iterative`
   - `./build/test_integration`
   - `./build/bench_refactor_csc`
   - `./build/bench_iterative_reuse`
   - `./build/bench_eigs_reuse`
   - `./build/example_analysis`
5. Record the authoritative validation boundary for docs-only audit/design days
   versus implementation days.

### Deliverables
- Refreshed validation/truthfulness notes
- Sprint 57 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 57 uses the same baseline wording and parity anchors as the live repo
- The authoritative giant-test and lifecycle rerun set is explicit before code
  work begins
- No validation ambiguity remains around test-refactor or regression days

---

## Day 3: Direct-Solver Giant-Test Audit

**Title:** Direct-Test Audit  
**Theme:** Reduce the largest direct-solver test files to concrete helper and
split seams before any permanent refactor lands  
**Time estimate:** 10 hours

### Tasks
1. Audit the live ownership shape inside the strongest direct-solver giant
   tests:
   - `tests/test_chol_csc.c`
   - `tests/test_ldlt_csc.c`
   - `tests/test_integration.c`
2. Separate each file into likely seam classes:
   - matrix/setup helpers
   - repeated assertion/reporting helpers
   - backend-routing proof groups
   - lifecycle/factor-many proof groups
   - one-shot compatibility proof groups
3. Rank the candidate refactor seams by:
   - helper reuse value
   - readability improvement
   - low regression risk
   - proof-surface clarity
4. Reject purely mechanical moves that would scatter related proof without
   reducing maintenance pain.
5. Write the direct-solver test audit artifact and ranked landing order.

### Deliverables
- Direct-solver giant-test seam audit
- Ranked direct-test refactor targets
- Proposed first refactor boundary

### Completion Criteria
- The direct-solver test-maintainability problem is reduced to named seams
- The first refactor target is justified by ownership clarity, not only line
  count
- Sprint 57 can start direct-test implementation work from a concrete map

---

## Day 4: Direct-Solver Test Refactor Design

**Title:** Direct-Test Design  
**Theme:** Freeze the first bounded direct-solver test refactor boundary
before editing permanent proof surfaces  
**Time estimate:** 10 hours

### Tasks
1. Select the Day 3 highest-value direct-test seam for the first landing.
2. Define the exact test-ownership split:
   - what remains in the current giant file
   - what becomes shared helper logic
   - what, if anything, moves into a dedicated test-local helper file
3. Define the invariants the refactor must preserve:
   - test names and intent
   - output/reporting truthfulness
   - corpus fixture coverage
   - one-shot versus repeated-run proof boundaries
4. Define the minimal cleanup policy for touched tests:
   - preserve useful assertion commentary
   - remove stale sprint-history narrative where encountered
5. Record the design artifact and landing checklist.

### Deliverables
- Direct-solver refactor design
- File/helper ownership map
- Refactor invariants and checklist

### Completion Criteria
- The first direct-test refactor boundary is explicit before code movement
- Ownership is defined at helper and proof-group level, not just conceptually
- Cleanup expectations are fixed before touched tests are rewritten

---

## Day 5: Direct-Solver Test Refactor Batch I

**Title:** Direct-Test Batch I  
**Theme:** Land the first bounded direct-solver giant-test refactor  
**Time estimate:** 11 hours

### Tasks
1. Extract the first owned direct-test helper or split seam.
2. Rewire the touched direct-solver tests onto the new helper boundaries.
3. Keep proof meaning, fixture usage, and output wording unchanged.
4. Remove stale sprint-history narrative from touched permanent test blocks
   while preserving useful proof commentary.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- Landed direct-solver refactor patch
- Reduced duplication in the strongest direct test hotspot
- Updated validation record

### Completion Criteria
- A real maintainability seam is extracted from a direct-solver giant test
- The remaining touched giant file is smaller and clearer than before
- Full required validation passes after the refactor

---

## Day 6: Direct-Solver Refactor Follow-Through Audit

**Title:** Direct-Test Follow-Through  
**Theme:** Re-audit the post-Day-5 direct-solver proof shape and fix the next
most valuable helper or grouping seam  
**Time estimate:** 10 hours

### Tasks
1. Re-audit the landed direct-test shape after Day 5.
2. Identify the next highest-value follow-through seam:
   - another helper extraction
   - assertion/reporting normalization
   - proof-group reordering for readability
3. Confirm which direct giant-test surfaces can intentionally stay dense in
   Sprint 57.
4. Record the updated seam map and follow-through landing boundary.
5. Prepare the handoff notes for the iterative/eigensolver audit days.

### Deliverables
- Post-Day-5 direct-test seam map
- Follow-through landing boundary
- Updated direct-proof assumptions

### Completion Criteria
- The direct-test queue is shaped by the landed Day 5 reality, not only the
  original estimate
- The remaining direct giant-test work is smaller and more concrete
- Sprint 57 can pivot cleanly to the solver-family test surfaces next

---

## Day 7: Iterative / Eigensolver Giant-Test Audit & Design

**Title:** Solver-Test Audit  
**Theme:** Reduce the iterative and eigensolver giant tests to concrete helper
and proof seams, then freeze the first bounded refactor boundary  
**Time estimate:** 11 hours

### Tasks
1. Audit the live ownership shape inside the strongest solver-family giant
   tests:
   - `tests/test_iterative.c`
   - `tests/test_svd.c`
   - `tests/test_qr.c`
2. Separate likely seam classes:
   - solver-setup helpers
   - repeated residual/iteration assertions
   - backend-specific parity groups
   - corpus-smoke groups
   - public-handle or lifecycle proof groups
3. Rank candidate seams by helper reuse value, proof clarity, and regression
   risk.
4. Select the first bounded iterative/eigensolver refactor target.
5. Record both the audit and the implementation design/checklist.

### Deliverables
- Iterative/eigensolver seam audit
- Ranked solver-family test refactor targets
- First solver-family refactor design

### Completion Criteria
- The iterative/eigensolver test-maintainability problem is reduced to named
  seams
- The first refactor boundary is explicit before code movement
- Sprint 57 can start solver-family implementation work from a concrete map

---

## Day 8: Iterative / Eigensolver Test Refactor Batch I

**Title:** Solver-Test Batch I  
**Theme:** Land the first bounded iterative or eigensolver giant-test refactor  
**Time estimate:** 10 hours

### Tasks
1. Extract the selected solver-family helper or proof-group seam.
2. Rewire the touched solver-family tests onto the new helper boundaries.
3. Preserve test names, assertion truthfulness, and benchmark/example parity.
4. Clean stale sprint-history prose from touched permanent test blocks where
   encountered.
5. Run:
   - `make format`
   - `make lint`
   - `make test`

### Deliverables
- Landed iterative/eigensolver refactor patch
- Reduced duplication in the touched solver-family giant test
- Updated validation record

### Completion Criteria
- A real maintainability seam is extracted from a solver-family giant test
- The touched giant file is smaller and clearer than before
- Required validation passes after the refactor

---

## Day 9: Iterative / Eigensolver Refactor Batch II or Helper Normalization

**Title:** Solver-Test Batch II  
**Theme:** Land the second bounded solver-family maintainability improvement
from the Day 7-8 seam map  
**Time estimate:** 10 hours

### Tasks
1. Re-audit the post-Day-8 solver-family test shape.
2. Land the highest-value remaining solver-family seam:
   - second helper extraction
   - assertion/helper normalization
   - proof-group split or reorder
3. Keep proof meaning and fixture coverage unchanged.
4. Record any intentionally deferred solver-family test density.
5. Run:
   - `make format`
   - `make lint`
   - `make test`

### Deliverables
- Second solver-family refactor patch
- Updated solver-family seam map
- Deferred queue notes

### Completion Criteria
- The solver-family giant-test queue is materially smaller than at sprint
  start
- Remaining density is intentional rather than accidental
- Required validation passes after the second solver-family batch

---

## Day 10: Lifecycle Regression Expansion Batch I

**Title:** Lifecycle Coverage I  
**Theme:** Expand final direct-solver and repeated-run lifecycle proof across
the public steady-state contract  
**Time estimate:** 11 hours

### Tasks
1. Identify the strongest still-implicit lifecycle regression gaps across the
   public direct path:
   - zero-init / free safety
   - analyze/factor/refactor sequencing
   - repeated solve behavior
   - mismatched state rejection
2. Add focused regression coverage in the highest-signal touched proof
   surfaces, likely centered around:
   - `tests/test_integration.c`
   - direct CSC proof surfaces where appropriate
3. Preserve the Sprint 50-56 public lifecycle wording and semantics exactly.
4. Avoid broad new behavior expansion outside the documented contract.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - targeted follow-ons if needed

### Deliverables
- Expanded lifecycle regression patch
- Updated lifecycle proof notes
- Targeted rerun record

### Completion Criteria
- The final public direct lifecycle is more explicitly proven than at sprint
  start
- Coverage additions stay bounded to the documented contract
- Required validation passes after the regression batch

---

## Day 11: Factor-Many / Compatibility Regression Expansion

**Title:** Factor-Many Coverage  
**Theme:** Add the highest-signal regression proof for factor-many workflows
and one-shot compatibility preservation  
**Time estimate:** 10 hours

### Tasks
1. Identify the strongest remaining factor-many and compatibility proof gaps:
   - same-pattern refactor expectations
   - repeated-run versus one-shot parity
   - old-factor preservation on failure
   - benchmark-facing workflow assumptions
2. Add focused regression coverage in the highest-value proof surfaces.
3. Keep benchmark-style truths and one-shot compatibility wording aligned with
   the live public contract.
4. Avoid broad new harness invention outside the bounded regression need.
5. Run:
   - `make format`
   - `make lint`
   - `make test`

### Deliverables
- Factor-many / compatibility regression patch
- Updated workflow-proof notes
- Targeted rerun record

### Completion Criteria
- Factor-many and one-shot compatibility proof are stronger than at sprint
  start
- Coverage additions stay tied to real public workflows
- Required validation passes after the regression batch

---

## Day 12: Post-Expansion Audit & Residual Queue Check

**Title:** Coverage Audit  
**Theme:** Re-audit the post-refactor, post-regression branch and confirm what
is intentionally deferred versus still drifted  
**Time estimate:** 10 hours

### Tasks
1. Re-audit the landed direct, iterative, eigensolver, and integration proof
   surfaces.
2. Confirm whether any touched helper ownership or regression wording still
   drifts from the intended steady-state contract.
3. Check benchmark/example-facing documentation for any high-signal proof drift
   exposed by the new regression work.
4. Record the bounded residual queue for Day 13-14.
5. Lock the final validation checklist from the landed state.

### Deliverables
- Post-expansion compatibility audit
- Residual queue notes
- Final validation checklist

### Completion Criteria
- Any remaining test-density or regression gaps are consciously deferred
- No blocker-level contract drift remains before the final validation sweep
- The final validation checklist is explicit and branch-accurate

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Reconfirm the full repo baseline plus the targeted giant-test and
lifecycle rerun set after the landed Sprint 57 work  
**Time estimate:** 11 hours

### Tasks
1. Run:
   - `make format`
   - `make lint`
   - `make test`
2. Run the stronger reviewed baseline:
   - `make quality-review-full`
3. Rerun the targeted Sprint 57 follow-ons if needed:
   - `./build/test_chol_csc`
   - `./build/test_ldlt_csc`
   - `./build/test_svd`
   - `./build/test_qr`
   - `./build/test_iterative`
   - `./build/test_integration`
   - selected workflow/benchmark/example binaries
4. Record exact parity counts, timings, and any anomalies.
5. Resolve any validation drift that appears before closeout.

### Deliverables
- Full validation-sweep notes
- Final parity/timing record
- Resolved validation anomalies or explicit blocker notes

### Completion Criteria
- The full required validation stack passes
- Reviewed Makefile/CMake parity remains exact
- No unresolved validation drift remains before closeout

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Consolidate Sprint 57 into a clear maintainability and regression
handoff package  
**Time estimate:** 10 hours

### Tasks
1. Summarize the landed giant-test maintainability improvements.
2. Summarize the landed lifecycle, factor-many, and compatibility regression
   additions.
3. Record the preserved public-behavior and validation fences explicitly.
4. Check whether `docs/planning/EPIC_5/PROJECT_PLAN.md` needs any follow-on
   correction based on Sprint 57 reality.
5. Write the closeout and handoff artifact for Sprint 58+.

### Deliverables
- Sprint 57 closeout artifact
- Final handoff summary
- Explicit deferred queue

### Completion Criteria
- Sprint 57 closes with one coherent giant-test and lifecycle-regression
  package
- Preserved behavior and validation fences remain explicit and unchanged
- The remaining queue is future-facing rather than a hidden Sprint 57 defect
