# Sprint 81 Plan: Core Product / Storage Modernization Phase 2

**Sprint Duration:** 14 days  
**Goal:** Reduce the highest-value linked-list-first product costs by making
compressed-first workflows more central, repeated-run direct paths more
coherent, and the orthogonal linked-list shell more clearly bounded. This
sprint implements the Sprint 81 section of
`docs/planning/EPIC_8/PROJECT_PLAN.md`.

**Starting Point:** Sprint 80 closed from a validated Epic 8 baseline with the
competitive target, external-oracle contract, benchmark-governance fence, and
non-goal/risk fence fixed in writing. The strongest remaining first-tier Epic 8
contradiction is still the linked-list-first storage/product ceiling.

The highest-value Sprint 81 work is therefore not generic cleanup. It is one
bounded product/storage modernization package that:
- audits the highest-cost storage/conversion seams against live workflows
- fixes one compressed-first architecture contract
- lands one first implementation batch on the strongest construction/import
  seams
- lands one bounded repeated-run workflow convergence follow-through
- proves the new boundary with focused regression and benchmark evidence

**End State:** Sprint 81 leaves behind:
- one refreshed storage/workflow hotspot map
- one explicit compressed-first architecture contract
- one bounded construction/import modernization landing
- one bounded workflow-convergence landing
- one focused proof and support-surface follow-through package
- one validated closeout baseline and Sprint 82-ready handoff

**Time budget:** Each day is capped at 12 hours as requested. Because that cap
allows at most `168` hours across 14 days, this day-by-day plan totals `168`
hours rather than the higher project-plan estimate of `~182` hours, while
preserving the Sprint 81 scope and ordering.

---

## Day 1: Sprint 81 Scope Audit & Storage Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 81 project-plan section and Sprint 80 closeout into
one bounded product/storage execution package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 81 section of
   `docs/planning/EPIC_8/PROJECT_PLAN.md`, the Sprint 80 closeout artifact, and
   the Sprint 80 retrospective.
2. Reconfirm the preserved Sprint 81 starting assumptions:
   - Sprint 80 already fixed the baseline and non-goal fence
   - the first implementation contradiction is the linked-list-first
     product/storage ceiling
   - Sprint 81 should not widen into backend, capability, or package/platform
     work
3. Define the Sprint 81 workstreams explicitly:
   - storage/conversion audit
   - compressed-first architecture design
   - construction/import landing
   - repeated-run workflow convergence
   - focused proof and benchmark follow-through
   - docs/examples/header alignment
   - validation and closeout
4. Record the strongest likely Sprint 81 touch surfaces:
   - `include/sparse_matrix.h`
   - `src/sparse_matrix.c`
   - the strongest conversion/publication seams
   - repeated-run direct-workflow owners
   - proof-owner tests and representative benchmarks
5. Open Sprint 81 working notes and record the intended landing order and
   validation expectations.

### Deliverables
- Sprint 81 scope inventory
- storage/workflow workstream map
- starting working-notes baseline

### Completion Criteria
- Sprint 81 starts from the validated Sprint 80 end state
- the first product/storage modernization target is explicit before deeper
  audits begin
- the non-goal fence is visible before any design or implementation work

---

## Day 2: Validation & Proof-Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, benchmark, and install proof split
before storage/workflow changes begin  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline and implementation-day gate:
   - `make quality-review-full`
   - `make format`
   - `make lint`
   - `make test`
2. Reconfirm reviewed CMake parity, canonical benchmark-report ownership, and
   install/export proof ownership.
3. Recheck the representative proof-owner surfaces most likely to move during
   Sprint 81:
   - direct workflow tests
   - sparse matrix tests
   - representative examples
   - canonical benchmark/reporting command surfaces
4. Fix the authoritative rerun list most likely to matter throughout Sprint 81.
5. Record the validation/proof split in working notes and a Day 2 artifact.

### Deliverables
- refreshed validation-baseline artifact
- preserved proof-owner map
- authoritative Sprint 81 rerun list

### Completion Criteria
- the strongest local validation contract is explicit before implementation
  work lands
- proof ownership across reviewed tests, benchmarks, and install/export
  surfaces is fixed in writing
- later code days have no ambiguity about the required validation gate

---

## Day 3: Storage / Conversion Hotspot Audit

**Title:** Hotspot Audit  
**Theme:** Re-rank the strongest linked-list-first storage and conversion costs
against the live direct workflows  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the highest-signal product/storage surfaces:
   - public matrix shell
   - construction/import paths
   - compression/conversion helpers
   - repeated-run direct-workflow publication seams
2. Identify where linked-list-first costs are highest:
   - import/setup paths
   - publication/writeback paths
   - repeated-run reuse boundaries
   - compatibility-only surfaces
3. Separate:
   - strongest first-batch implementation center
   - second-tier follow-through seams
   - support-only proof and docs surfaces
   - deliberate non-goals
4. Reconcile the audit against the Sprint 80 contradiction map.
5. Write the ranked hotspot artifact.

### Deliverables
- ranked storage/conversion hotspot artifact
- first-tier vs deferred seam map
- Sprint 80 carry-forward reconciliation notes

### Completion Criteria
- Sprint 81’s broad storage problem is reduced to one ranked live map
- the strongest implementation center is explicit before boundary design
- lower-value spillover work is clearly separated from the first lane

---

## Day 4: First Storage Boundary Freeze

**Title:** Boundary Freeze  
**Theme:** Fix the first bounded implementation fence for compressed-first
modernization  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 ranking against the Sprint 81 project-plan scope.
2. Decide the required first implementation center:
   - construction/import seam
   - publication/conversion seam
   - repeated-run workflow seam
3. Decide which support surfaces move only if forced:
   - proof-owner tests
   - benchmarks
   - headers
   - docs/examples
4. Fix the preserved non-goal fence for the first landing:
   - no broad API redesign
   - no backend/capability reopening
   - no generic whole-library workflow rewrite
5. Record the first implementation fence in working notes and a Day 4 artifact.

### Deliverables
- first storage-boundary artifact
- required vs support-only touch set
- preserved first-batch non-goal fence

### Completion Criteria
- Sprint 81 has one explicit first landing boundary
- support-only surfaces are clearly separated from the batch center
- Day 5 can design one implementation contract instead of a broad rewrite

---

## Day 5: Compressed-First Architecture Design

**Title:** Architecture Design  
**Theme:** Define the bounded product/storage contract Sprint 81 will actually
land  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 4 boundary and the strongest product/storage contradictions.
2. Define the ownership split for:
   - compressed-first construction/import
   - linked-list compatibility shell
   - conversion/publication
   - repeated-run workflow reuse
3. Decide how the first landing will preserve compatibility callers while
   reducing linked-list-first tax.
4. Fix the touch fence for tests, benchmarks, docs, and headers.
5. Write the Day 5 architecture contract artifact and working-notes design
   summary.

### Deliverables
- compressed-first architecture contract
- ownership split for touched seams
- preserved compatibility and non-goal fence

### Completion Criteria
- Sprint 81 has one explicit implementation contract
- ownership between compressed-first paths and compatibility shell is clear
- Day 6 can implement one bounded landing without reopening design questions

---

## Day 6: Construction / Import Batch 1

**Title:** Import Batch  
**Theme:** Land the first bounded compressed-first construction/import seam  
**Time estimate:** 12 hours

### Tasks
1. Implement the highest-value construction/import modernization seam from the
   Day 5 contract.
2. Keep the landing bounded to the required first implementation center.
3. Update any truly forced local proof-owner tests.
4. Record the landing in working notes and a Day 6 artifact.
5. Run the required validation gate for touched code.

### Deliverables
- first compressed-first construction/import landing
- any forced focused regression follow-through
- Day 6 implementation artifact

### Completion Criteria
- the first bounded storage/product batch lands inside the Day 5 fence
- compatibility behavior remains preserved on the touched paths
- the required validation gate passes

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the strongest remaining product/storage seam after the first
landing  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6 landing and the touched proof surfaces.
2. Decide whether the strongest remaining contradiction is now:
   - repeated-run workflow convergence
   - publication/writeback follow-through
   - support-surface alignment
3. Separate:
   - required next landing center
   - support-only proof or docs surfaces
   - later deferred work
4. Record the rerank and next-batch fence.
5. Use the rerank to fix the Day 8 design center.

### Deliverables
- post-landing audit artifact
- next-batch rerank
- support-only follow-through map

### Completion Criteria
- the strongest remaining Sprint 81 seam is explicit after Day 6
- the sprint does not blindly repeat the same-family landing without evidence
- Day 8 has one fixed design center

---

## Day 8: Workflow Convergence Design

**Title:** Workflow Design  
**Theme:** Define the bounded repeated-run workflow convergence follow-through  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 rerank against the Sprint 81 project-plan scope.
2. Define the exact repeated-run or publication/convergence seam to land next.
3. Fix the support-only surfaces if the next batch forces them:
   - tests
   - benchmarks
   - headers
   - docs/examples
4. Preserve the fence against broad workflow rewrite or API churn.
5. Record the Day 8 design artifact and working-notes contract.

### Deliverables
- workflow-convergence design artifact
- exact Day 9 touch set
- preserved second-batch fence

### Completion Criteria
- the next workflow/product batch is explicitly bounded
- the touched proof and support surfaces are fixed before implementation
- Day 9 can land one coherent follow-through batch

---

## Day 9: Workflow Convergence Batch

**Title:** Workflow Batch  
**Theme:** Land the bounded repeated-run product/storage convergence seam  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 8 workflow-convergence design.
2. Update any truly forced proof-owner tests or benchmark follow-through.
3. Avoid reopening unrelated product, backend, capability, or package lanes.
4. Record the landing in working notes and a Day 9 artifact.
5. Run the required validation gate for touched code.

### Deliverables
- bounded workflow-convergence landing
- any forced focused proof follow-through
- Day 9 implementation artifact

### Completion Criteria
- the second bounded Sprint 81 landing stays inside the Day 8 fence
- touched repeated-run/product seams read more coherently after the batch
- the required validation gate passes

---

## Day 10: Proof & Benchmark Follow-Through Design

**Title:** Proof Design  
**Theme:** Fix the exact proof, benchmark, and support follow-through needed
after the implementation batches  
**Time estimate:** 12 hours

### Tasks
1. Audit whether the landed batches changed:
   - proof ownership
   - benchmark measurability
   - public header wording
   - examples/docs truthfulness
2. Decide the exact Day 11 follow-through surfaces.
3. Separate required proof work from support-only documentation work.
4. Preserve the fence against broad benchmark, docs, or header churn.
5. Record the Day 10 follow-through design artifact.

### Deliverables
- proof/benchmark/docs follow-through design
- exact Day 11 touch set
- preserved support-only fence

### Completion Criteria
- Sprint 81 knows exactly what follow-through is required after the main
  landings
- proof-owner and benchmark needs are explicit before edits begin
- the support lane is bounded rather than generic cleanup

---

## Day 11: Docs / Examples / Header Alignment Batch

**Title:** Follow-Through  
**Theme:** Reconcile the touched product/storage surfaces with the landed
compressed-first and workflow changes  
**Time estimate:** 12 hours

### Tasks
1. Land the required docs/examples/header follow-through from the Day 10
   design.
2. Keep the batch bounded to surfaces genuinely moved by the implementation.
3. Record any explicit no-op outcomes where wording already stays truthful.
4. Write the Day 11 artifact and working-notes summary.
5. Reconfirm the branch remains inside the Sprint 81 fence.

### Deliverables
- bounded docs/examples/header alignment batch
- any explicit no-op support-surface decisions
- Day 11 follow-through artifact

### Completion Criteria
- touched support surfaces accurately describe the landed storage/workflow
  contract
- no unnecessary churn spreads into unrelated public or policy surfaces
- Sprint 81 is ready for final proof alignment

---

## Day 12: Final Proof Alignment & Validation Queue

**Title:** Proof Alignment  
**Theme:** Fix the exact Day 13 rerun set and final ownership map for Sprint 81  
**Time estimate:** 12 hours

### Tasks
1. Re-read the landed implementation, proof, benchmark, and support surfaces.
2. Reconfirm the strongest final proof-owner map for Sprint 81.
3. Fix the exact Day 13 validation queue around:
   - code-day gate
   - strongest reviewed proof-owner binaries
   - representative examples
   - any touched benchmark/reporting follow-ons
   - install/export proof if still relevant
4. Record any explicit no-op proof outcomes.
5. Write the Day 12 artifact and working-notes alignment note.

### Deliverables
- final proof-owner map
- authoritative Day 13 validation queue
- Day 12 alignment artifact

### Completion Criteria
- the final rerun set is explicit before validation starts
- proof ownership is fixed in writing
- Day 13 can execute from one stable measured queue

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Execute the full Sprint 81 validation queue and retain the closeout
baseline  
**Time estimate:** 12 hours

### Tasks
1. Run the full code-day validation baseline:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest reviewed validation baseline:
   - `make quality-review-full`
3. Re-run the authoritative Day 12 focused proof-owner binaries and
   representative examples.
4. Re-run any touched benchmark/reporting follow-ons.
5. Record retained anchors, representative outputs, and any real issue that
   must be fixed before closeout.

### Deliverables
- full validation-sweep artifact
- retained reviewed anchors
- explicit issue/fix note if validation exposes a real problem

### Completion Criteria
- the full Sprint 81 rerun set completes cleanly
- retained anchors and representative outputs are recorded
- any real issue discovered is resolved or explicitly escalated before Day 14

---

## Day 14: Sprint 81 Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 81 from the validated baseline and hand off the next
Epic 8 contradiction center  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 81 plan, landed artifacts, and the Day 13 validation
   baseline.
2. Summarize exactly what Sprint 81 changed in the product/storage contract.
3. Fix the ranked carry-forward queue for Sprint 82 and later Epic 8 work.
4. Recheck whether `docs/planning/EPIC_8/PROJECT_PLAN.md` needs any Sprint 81
   correction.
5. Write the Day 14 closeout/handoff artifact and finalize working notes.

### Deliverables
- Sprint 81 closeout/handoff artifact
- final working-notes close state
- handoff queue for Sprint 82

### Completion Criteria
- Sprint 81 closes from a validated baseline instead of from partial
  implementation state
- the next Epic 8 contradiction center is explicit
- the branch is ready for retrospective creation and handoff
