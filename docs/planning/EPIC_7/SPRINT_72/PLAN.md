# Sprint 72 Plan: Core Matrix/Product Model Convergence Phase 1

**Sprint Duration:** 14 days  
**Goal:** Reduce the structural cost of the linked-list-first public model by
clarifying ownership boundaries between mutable matrix construction, compressed
working formats, and explicit factor/workspace state. This sprint implements
the Sprint 72 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`.

**Starting Point:** Sprint 71 closed with a cleaner public/reference package
and a ranked implementation-facing carry-forward queue:
- the strongest local reviewed baseline remains `make quality-review-full`
- Sprint 70 fixed the Epic 7 product-model target and architecture fence
- Sprint 71 removed the strongest public/reference chronology drag
- the strongest next queue is now product-model convergence from the public
  direct-workflow seam
- `SparseMatrix` still carries too many roles at once:
  - mutable construction surface
  - generic arithmetic surface
  - permutation owner
  - factored-state carrier
  - interoperability shell
- the strongest compressed-path support seams remain centered on:
  - `src/sparse_matrix.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_lu_csr.c`
- the Sprint 70 non-goal fence is still in force:
  - no broad `SparseMatrix` rewrite
  - no capability widening disguised as ownership cleanup
  - no packaging/platform claim widening
  - no fake abstraction layer detached from actual workflow pain

The highest-value Sprint 72 work is therefore not a full matrix-model reset.
It is a bounded first convergence pass on the public direct-workflow seam and
its strongest compressed/factor-state ownership contradictions, with proof and
docs follow-through only where the landed implementation truly moves the
contract.

**End State:** Sprint 72 leaves behind a cleaner first-phase product model:
- a ranked live ownership map for `SparseMatrix`, compressed paths, and
  factor/workspace state
- one bounded direct-workflow hardening batch that reduces copy/mutation
  surprise
- one bounded compressed-path ownership batch that reduces publication or
  round-trip friction
- refreshed public contract/docs/example wording for the touched lanes
- focused regression expansion for the refined ownership boundary
- a validated Sprint 72 closeout package that lets Sprint 73 start from a
  cleaner implementation boundary without reopening the Sprint 70 architecture
  fence

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
158 hours, staying within the day-cap limit while covering the Sprint 72
implementation scope.

---

## Day 1: Sprint 72 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 72 project-plan scope plus the Sprint 70-71
handoff into a bounded first-phase product-model convergence sprint  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 72 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`,
   the Sprint 70 architecture contract, the Sprint 71 retrospective, and the
   Sprint 71 closeout artifact.
2. Reconfirm the preserved Sprint 72 constraints:
   - no broad `SparseMatrix` rewrite
   - no capability widening disguised as ownership cleanup
   - no packaging/platform/install contract widening
   - no fake generic abstraction layer detached from actual workflow pressure
3. Define the Sprint 72 workstreams explicitly:
   - product-model surface audit
   - ownership convergence design
   - direct-workflow hardening
   - compressed-path ownership cleanup
   - public contract/example follow-through
   - regression expansion
   - validation and closeout
4. Record the strongest likely Sprint 72 touch surfaces:
   - `include/sparse_matrix.h`
   - `include/sparse_analysis.h`
   - `src/sparse_matrix.c`
   - `src/sparse_chol_csc.c`
   - `src/sparse_ldlt_csc.c`
   - `src/sparse_lu_csr.c`
   - likely proof owners from Sprint 70
5. Open Sprint 72 working notes and record the intended landing order,
   required artifacts, and validation expectations.

### Deliverables
- Sprint 72 scope inventory
- Product-model convergence map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 72 starts from the Sprint 70-71 contract rather than reopening broad
  Epic 7 planning
- The implementation workstreams are explicit before deeper audit begins
- The sprint non-goal fence is fixed before design or landing work proceeds

---

## Day 2: Validation Baseline & Rerun Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the implementation-day validation contract and the
highest-signal rerun surfaces Sprint 72 must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 72 authority split:
   - `*.c` / `*.h` landing days require `make format`, `make lint`, and
     `make test`
   - substantial architecture or ownership-boundary batches default to
     `make quality-review-full`
   - docs-only audit/design days use targeted sanity checks only
3. Recheck the live proof surfaces Sprint 72 is most likely to stress:
   - direct workflow integration
   - Cholesky/LDL^T CSC proof owners
   - LU CSR proof owners
   - representative examples and benchmark surfaces
4. Refresh the targeted rerun set most likely to matter in Sprint 72.
5. Record the authoritative validation split in the working notes.

### Deliverables
- Refreshed validation notes
- Sprint 72 rerun checklist
- Preserved proof-surface checklist

### Completion Criteria
- Sprint 72 uses the same reviewed/truthfulness reading fixed in Sprint 70
- The code-day validation contract is explicit before convergence work starts
- No rerun ambiguity remains around the likely touched ownership surfaces

---

## Day 3: Product-Model Surface Audit I

**Title:** Ownership Audit I  
**Theme:** Re-rank where `SparseMatrix` remains the right public owner and
where it now behaves mainly as a compatibility shell  
**Time estimate:** 12 hours

### Tasks
1. Audit the current top product-model surfaces:
   - `include/sparse_matrix.h`
   - `include/sparse_analysis.h`
   - `include/sparse_lu.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_ldlt.h`
   - `src/sparse_matrix.c`
2. Classify the strongest remaining burdens:
   - copy/mutation surprise
   - mixed logical versus physical matrix-state semantics
   - duplicated publication or writeback ownership
   - factor/workspace ownership blur
3. Record where the current product model is already strong and where it still
   reads like a linked-list-first compatibility center around compressed work.
4. Rank the strongest contradiction centers by:
   - caller confusion cost
   - implementation ownership blur
   - likely bounded Sprint 72 payoff
5. Write the first product-model audit artifact.

### Deliverables
- Initial product-model audit
- Ranked ownership-contradiction list
- First convergence hotspot map

### Completion Criteria
- The broad Sprint 72 product-model problem is reduced to a concrete file and
  seam ranking
- The strongest direct-workflow and ownership contradictions are explicit
- Day 4 can proceed from a real current-state ranking

---

## Day 4: Product-Model Surface Audit II & First Landing Boundary

**Title:** Ownership Audit II  
**Theme:** Refine the ownership ranking and freeze the first convergence
boundary for the sprint  
**Time estimate:** 12 hours

### Tasks
1. Re-rank the Day 3 product-model seams against:
   - public direct-workflow pain
   - implementation leverage
   - compatibility risk
   - likely bounded cleanup payoff
2. Separate:
   - first-batch landing surfaces
   - support surfaces that move only if the first batch forces it
   - later or explicitly deferred product-model surfaces
3. Identify the strongest first Sprint 72 convergence fence:
   - direct one-shot workflow
   - repeated-run lifecycle handoff
   - compressed-path writeback/publication seam
   - factor/workspace ownership seam
4. Record the strongest non-goals:
   - no repo-wide matrix-model rewrite
   - no capability/type expansion hidden inside ownership work
   - no broad family-by-family redesign without a ranked center
5. Fix the Day 4 product-model boundary in writing.

### Deliverables
- Refined product-model ranking
- First convergence boundary
- Deferred/support-surface map

### Completion Criteria
- The first Sprint 72 landing fence is explicit before design begins
- Lower-value or higher-risk cleanup is clearly separated from the first lane
- Support surfaces are bounded rather than assumed

---

## Day 5: Ownership Convergence Design

**Title:** Convergence Design  
**Theme:** Define the bounded implementation contract for the first
product-model landing before edits begin  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 70 product-model target and non-goal fences against the
   first-batch surfaces.
2. Design the first landing around:
   - clearer direct-workflow ownership
   - reduced copy/mutation surprise
   - cleaner factor/workspace separation
   - preserved compatibility at the public API level
3. Decide what stays owned by `SparseMatrix`, what becomes clearer support
   state, and what must remain untouched in Sprint 72.
4. Fix the exact first-batch non-touch set:
   - unrelated solver families
   - capability/type surfaces
   - packaging/platform/workflow files
   - broad doc-cleanup spill
5. Record the Day 5 design artifact.

### Deliverables
- Ownership convergence design
- First-batch non-touch set
- Preserved compatibility checklist

### Completion Criteria
- The first convergence batch is explicitly designed before code edits begin
- The landing is bounded to ownership clarity, not generic refactoring
- Compatibility and non-goal fences are fixed in writing

---

## Day 6: Direct Workflow Hardening Batch I

**Title:** Workflow Batch I  
**Theme:** Land the highest-value direct-workflow ownership cleanup without
breaking the public contract  
**Time estimate:** 12 hours

### Tasks
1. Edit the first-batch implementation surfaces to reduce the strongest
   copy/mutation or factor/publication surprise in the direct workflow.
2. Keep the public API stable while improving the internal ownership story.
3. Add or update focused proof only where the landing changes an ownership
   boundary that callers can observe.
4. Run:
   - `make format`
   - `make lint`
   - `make test`
   - escalate to `make quality-review-full` if the landed batch crosses the
     substantial-boundary threshold
5. Record the Day 6 landing artifact.

### Deliverables
- First direct-workflow hardening batch
- Focused proof for the touched boundary
- Landing artifact and validation notes

### Completion Criteria
- The strongest first direct-workflow contradiction is materially reduced
- The batch stays inside the Day 5 fence
- Required validation passes

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the remaining product-model seams after the first code
landing  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Day 6 landing against the original Day 3-5 ranking.
2. Confirm what contradiction the first batch actually closed and what remains
   strongest.
3. Decide whether Sprint 72’s second implementation lane should stay on the
   direct path or move to the strongest compressed-path ownership seam.
4. Re-rank the likely proof and docs follow-through surfaces from the
   post-Day-6 state.
5. Record the rerank artifact and the exact Day 8 design target.

### Deliverables
- Post-landing audit
- Refined remaining-seam ranking
- Exact second-lane target

### Completion Criteria
- The next highest-value lane is fixed from live post-landing evidence
- The sprint does not drift into a fake second batch
- Day 8 can begin from a real reranked state

---

## Day 8: Compressed-Path Ownership Design

**Title:** Compressed-Path Design  
**Theme:** Define the bounded second implementation batch around the strongest
remaining CSC/CSR-backed ownership seam  
**Time estimate:** 12 hours

### Tasks
1. Re-read the post-Day-6 rerank against:
   - `src/sparse_chol_csc.c`
   - `src/sparse_ldlt_csc.c`
   - `src/sparse_lu_csr.c`
   - support headers or public lifecycle surfaces only if needed
2. Design the second batch around:
   - clearer compressed-path publication/writeback ownership
   - reduced round-trip friction
   - preserved public direct-workflow semantics
3. Fix the exact touched-file fence for the second landing.
4. Identify the proof-owner tests or integration surfaces that must move with
   the batch.
5. Record the Day 8 design artifact.

### Deliverables
- Compressed-path ownership design
- Second-batch touched-file fence
- Focused proof-home map

### Completion Criteria
- The second implementation batch is explicitly designed before edits begin
- The lane is bounded to the strongest remaining ownership seam
- Proof expectations are fixed before code lands

---

## Day 9: Compressed-Path Ownership Batch

**Title:** Workflow Batch II  
**Theme:** Land the bounded compressed-path ownership cleanup and prove the
refined boundary  
**Time estimate:** 12 hours

### Tasks
1. Edit the exact Day 8 touched surfaces to reduce the strongest remaining
   compressed-path ownership blur.
2. Keep public workflow compatibility stable while reducing publication or
   writeback friction.
3. Update or add focused regression/integration proof for the touched seam.
4. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
5. Record the Day 9 landing artifact with retained proof anchors.

### Deliverables
- Second implementation batch
- Focused regression expansion for the touched seam
- Landing artifact and validation notes

### Completion Criteria
- The strongest second ownership seam is materially reduced
- The batch stays inside the Day 8 fence
- Full required validation passes

---

## Day 10: Public Contract & Example Adoption Design

**Title:** Contract Design  
**Theme:** Define the exact public-header, doc, and example follow-through
required by the landed ownership work  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Day 6 and Day 9 landings against the current public-facing
   contract surfaces:
   - affected headers
   - `README.md`
   - `docs/tutorial.md`
   - `examples/example_analysis.c`
   - `examples/example_basic_solve.c`
2. Separate:
   - wording or example follow-through that is now required
   - surfaces that remain coherent and should not move
3. Fix the exact Day 11 non-code or bounded-code follow-through fence.
4. Preserve the Sprint 70 truthfulness and Sprint 71 public-surface cleanup
   gains while updating the product-model story.
5. Record the Day 10 design artifact.

### Deliverables
- Public contract/example design
- Exact Day 11 follow-through fence
- Preserved truthfulness checklist

### Completion Criteria
- Only truly necessary contract/adoption follow-through is scheduled
- The sprint avoids a generic documentation spill
- Day 11 can land from a bounded design

---

## Day 11: Public Contract & Example Adoption Batch

**Title:** Contract Batch  
**Theme:** Land the exact public-facing follow-through required by the Sprint
72 implementation work  
**Time estimate:** 12 hours

### Tasks
1. Edit the exact public headers/docs/examples required by the landed
   ownership changes.
2. Keep the public explanation aligned with the refined direct-workflow and
   compressed-path ownership story.
3. Avoid widening product, benchmark, platform, or capability claims.
4. Run:
   - `make format`
   - `make lint`
   - `make test`
5. Record the Day 11 artifact with the touched-surface summary.

### Deliverables
- Public contract/example follow-through batch
- Updated adoption/reference wording
- Day 11 artifact and validation notes

### Completion Criteria
- The landed implementation batches and the public contract now agree
- The follow-through stays bounded to actually moved ownership
- Required validation passes

---

## Day 12: Regression Expansion & Build Alignment

**Title:** Proof Alignment  
**Theme:** Tighten the regression surface and build/reference alignment around
the new ownership boundary  
**Time estimate:** 12 hours

### Tasks
1. Re-read the touched proof-owner tests, examples, and any affected
   benchmark/reference surfaces from the post-Day-11 state.
2. Add any last focused regression proof still needed for the refined boundary.
3. Tighten doc/build/test ownership wording only where the landed code changed
   the sustained contract.
4. Run:
   - `make format`
   - `make lint`
   - `make test`
5. Record the Day 12 artifact with the final touched proof surface.

### Deliverables
- Final focused regression expansion
- Build/reference ownership alignment
- Day 12 artifact and validation notes

### Completion Criteria
- The new ownership boundary has explicit proof where it matters
- No stale wording remains across the touched proof/adoption surfaces
- Required validation passes

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Re-run the strongest local reviewed baseline and the highest-signal
follow-ons across the Sprint 72 package  
**Time estimate:** 12 hours

### Tasks
1. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Recheck the reviewed CMake parity anchor and retained proof counts.
3. Run the highest-signal follow-ons for the touched Sprint 72 lanes:
   - direct-workflow proof owners
   - representative examples
   - maintained benchmark/reporting surfaces if the landed ownership work
     touched their contract
4. Record retained proof and runtime anchors in the Day 13 artifact.
5. Fix the exact Day 14 closeout queue from the validated state.

### Deliverables
- Full validation artifact
- Retained proof anchors
- Day 14 closeout queue

### Completion Criteria
- The strongest local reviewed baseline passes after the Sprint 72 landings
- The touched proof/adoption surfaces all pass from the validated state
- Day 14 can close from a real validated package

---

## Day 14: Sprint 72 Closeout & Handoff

**Title:** Closeout and Handoff  
**Theme:** Close Sprint 72 with one explicit first-phase product-model
convergence package and a clear carry-forward queue for Sprint 73  
**Time estimate:** 10 hours

### Tasks
1. Write the Sprint 72 closeout artifact summarizing what was fixed in:
   - direct-workflow ownership
   - compressed-path ownership
   - public contract/example follow-through
   - proof expansion
   - validated close state
2. Rank the strongest carry-forward items for Sprint 73 and beyond:
   - next product-model phase
   - configuration modernization
   - capability modernization
   - backend/performance maturity
   - later permanent-surface cleanup
3. Recheck whether `docs/planning/EPIC_7/PROJECT_PLAN.md` needs any Sprint 72
   correction after the landed ownership work.
4. Confirm the final Sprint 72 branch state and validation footprint.
5. Record the final handoff notes for the next sprint.

### Deliverables
- Sprint 72 closeout artifact
- Ranked Sprint 73 carry-forward queue
- Final handoff notes

### Completion Criteria
- Sprint 72 ends with a cleaner first-phase product-model package rather than
  a loose set of implementation edits
- Sprint 73 can begin from a ranked, bounded carry-forward queue
- Any project-plan correction need is explicitly resolved before handoff
