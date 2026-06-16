# Sprint 74 Plan: Capability Surface Modernization Phase 1

**Sprint Duration:** 14 days  
**Goal:** Start closing the biggest capability ceilings by defining and landing
the first bounded modernization path for index width and scalar-surface
expansion. This sprint implements the Sprint 74 section of
`docs/planning/EPIC_7/PROJECT_PLAN.md`.

**Starting Point:** Sprint 73 closed with a validated residual-configuration
modernization package and a ranked carry-forward queue:
- the strongest local reviewed baseline remains `make quality-review-full`
- Sprint 70 fixed the Epic 7 capability target and truthfulness fence
- Sprint 72 reduced the strongest first-phase product-model contradictions
  without widening into a broad core rewrite
- Sprint 73 reduced the strongest residual configuration contradictions
  without widening into public capability claims
- the strongest remaining capability ceilings are still:
  - 32-bit index width
  - real-only scalar support
  - narrower algorithm-surface breadth beyond the current symmetric eigensolver
    emphasis
- the Sprint 70 non-goal fence is still in force:
  - no one-shot full type-generic conversion
  - no fake complex-readiness story without end-to-end proof
  - no broad capability widening disguised as local cleanup
  - no widened platform/install/reviewed claims detached from shipped evidence

The highest-value Sprint 74 work is therefore not a full library-wide type
rewrite. It is a bounded first modernization phase on the strongest capability
ceiling, with end-to-end index-width readiness work on the highest-value seams,
plus the minimum scalar-surface preparation and docs/proof follow-through
needed to make that capability path real without overstating what the library
supports today.

**End State:** Sprint 74 leaves behind a smaller and more coherent capability
gap:
- a ranked, evidence-based capability ceiling map
- one explicit bounded architecture contract for index width and scalar-surface
  preparation
- one shipped first end-to-end index-width readiness seam on the highest-value
  touched paths
- one bounded scalar-surface preparation seam for later widening
- focused overflow, compatibility, and correctness proof on the touched paths
- a validated Sprint 74 closeout package that lets Sprint 75 start from a
  real first capability-modernization seam rather than a speculative target

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
150 hours, staying within the day-cap limit while covering the Sprint 74
implementation scope.

---

## Day 1: Sprint 74 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 74 project-plan scope plus the Sprint 70, Sprint
72, and Sprint 73 handoff into a bounded capability-modernization sprint  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 74 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`,
   the Sprint 70 capability target, the Sprint 72 product-model
   retrospective, and the Sprint 73 closeout artifact.
2. Reconfirm the preserved Sprint 74 constraints:
   - no one-shot full type-generic conversion
   - no fake complex-readiness story
   - no broad capability widening hidden inside local helper work
   - no widened reviewed/platform/install claims
3. Define the Sprint 74 workstreams explicitly:
   - capability audit
   - index/scalar architecture design
   - index-width integration batch
   - scalar-surface preparation batch
   - docs/packaging/test alignment
   - regression and safety checks
   - validation and closeout
4. Record the strongest likely Sprint 74 touch surfaces:
   - `include/sparse_types.h`
   - `include/sparse_matrix.h`
   - `src/sparse_types.c`
   - `src/sparse_matrix.c`
   - highest-value proof owners in matrix, integration, and overflow-sensitive
     paths
5. Open Sprint 74 working notes and record the intended landing order,
   required artifacts, and validation expectations.

### Deliverables
- Sprint 74 scope inventory
- Capability-modernization workstream map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 74 starts from the Sprint 70, Sprint 72, and Sprint 73 contract
  rather than reopening broad Epic 7 planning
- The implementation workstreams are explicit before deeper audit begins
- The sprint non-goal fence is fixed before design or landing work proceeds

---

## Day 2: Validation Baseline & Truth-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the implementation-day validation contract and the
highest-signal rerun surfaces Sprint 74 must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 74 authority split:
   - `*.c` / `*.h` landing days require `make format`, `make lint`, and
     `make test`
   - substantial architecture or capability-boundary batches default to
     `make quality-review-full`
   - docs-only audit/design days use targeted sanity checks only
3. Recheck the live proof surfaces Sprint 74 is most likely to stress:
   - matrix and integration proof owners
   - overflow-sensitive tests
   - representative examples
   - maintained benchmark/reporting surfaces
   - install/package proof scripts
4. Refresh the targeted rerun set most likely to matter in Sprint 74.
5. Record the authoritative validation split in the working notes.

### Deliverables
- Refreshed validation notes
- Sprint 74 rerun checklist
- Preserved proof-surface checklist

### Completion Criteria
- Sprint 74 uses the same reviewed/truthfulness reading fixed in Sprint 70
- The code-day validation contract is explicit before capability work starts
- No rerun ambiguity remains around the likely touched overflow and type seams

---

## Day 3: Capability Ceiling Audit

**Title:** Capability Audit  
**Theme:** Re-rank the current 32-bit, real-only, and algorithm-surface
ceilings by user value and implementation risk  
**Time estimate:** 12 hours

### Tasks
1. Audit the current capability ceilings across:
   - index width
   - scalar type breadth
   - algorithm-surface breadth that depends on those ceilings
2. Classify the strongest remaining burdens:
   - public dimension/nnz ceilings
   - pervasive `idx_t` assumptions
   - pervasive `double` assumptions
   - packaging and compatibility implications of widening either surface
3. Record where the current capability surface is already strong and where it
   still behaves like a narrow compatibility shell.
4. Rank the strongest contradiction centers by:
   - caller value
   - implementation leverage
   - proof cost
   - bounded Sprint 74 payoff
5. Write the first capability-ceiling audit artifact.

### Deliverables
- Capability ceiling audit
- Ranked contradiction list
- First modernization hotspot map

### Completion Criteria
- The broad Sprint 74 capability problem is reduced to a concrete seam ranking
- The highest-value ceilings are explicit
- Day 4 can proceed from a real current-state ranking

---

## Day 4: First Capability Boundary

**Title:** Boundary Freeze  
**Theme:** Refine the ranking and freeze the first modernization fence for the
sprint  
**Time estimate:** 12 hours

### Tasks
1. Re-rank the Day 3 capability ceilings against:
   - caller value
   - implementation leverage
   - compatibility risk
   - likely bounded cleanup payoff
2. Separate:
   - first-batch landing surfaces
   - support surfaces that move only if the first batch forces it
   - later or explicitly deferred capability work
3. Identify the strongest first Sprint 74 fence:
   - index-width readiness
   - scalar-surface preparation
   - later algorithm-family widening
4. Record the strongest non-goals:
   - no full-width flip across the entire repo
   - no broad scalar-generic conversion
   - no fake capability story beyond touched paths
5. Fix the Day 4 modernization boundary in writing.

### Deliverables
- Refined capability ranking
- First landing boundary
- Deferred/support-surface map

### Completion Criteria
- The first Sprint 74 landing fence is explicit before design begins
- Lower-value or higher-risk capability work is clearly separated from the
  first lane
- Support surfaces are bounded rather than assumed

---

## Day 5: Index / Scalar Architecture Design

**Title:** Architecture Design  
**Theme:** Define the bounded implementation contract for the first
capability-modernization landing before edits begin  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 70 capability target and non-goal fences against the
   first-batch surfaces.
2. Decide the ownership split for the strongest remaining capability work:
   - immediate index-width readiness seam
   - scalar-surface abstraction/preparation only where it helps later work
   - compatibility expectations for existing callers
3. Define the guarantees the batch must preserve:
   - current builds still behave correctly at current widths
   - overflow handling becomes clearer, not weaker
   - scalar-surface prep does not imply broad new numeric support
4. Fix the exact first-batch non-touch set:
   - unrelated solver families
   - broader algorithm-surface widening
   - platform/install/reviewed-claim spill
   - broad doc-cleanup spill
5. Record the Day 5 design artifact.

### Deliverables
- Index/scalar architecture design
- First-batch non-touch set
- Preserved compatibility checklist

### Completion Criteria
- The first modernization batch is explicitly designed before code edits begin
- The landing is bounded to capability ownership clarity, not generic
  refactoring
- Compatibility and non-goal fences are fixed in writing

---

## Day 6: Index-Width Integration Batch I

**Title:** Index Batch I  
**Theme:** Land the first end-to-end wider-index readiness seam on the
highest-value touched paths  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded first index-width readiness batch on the exact Day 5
   touched surfaces.
2. Preserve current behavior at the shipped width while making the width seam
   more explicit and less ambient.
3. Tighten overflow-aware allocation, counting, or boundary paths where the
   design says they belong.
4. Add the minimum focused regression needed to prove the landed width seam.
5. Run the full required validation for a substantial capability-boundary
   landing and record the results.

### Deliverables
- First index-width readiness landing
- Focused regression for the touched seam
- Validated Day 6 artifact

### Completion Criteria
- The highest-value width seam is materially clearer after the landing
- The batch does not widen into a repo-wide type flip
- Validation passes from the correct authority split

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the remaining capability seams after the first width batch  
**Time estimate:** 10 hours

### Tasks
1. Audit the post-Day-6 state against the Day 5 architecture contract.
2. Confirm what contradiction the Day 6 landing actually closed.
3. Rerank the remaining queue:
   - next width seam
   - scalar-surface preparation
   - later algorithm-surface widening
4. Decide whether the best next move is another width landing or the planned
   scalar-preparation lane.
5. Record the Day 7 audit artifact and revised queue.

### Deliverables
- Post-landing audit
- Revised seam ranking
- Exact Day 8 design fence

### Completion Criteria
- Sprint 74’s second lane is chosen from the live post-Day-6 state
- The queue is reranked from real ownership payoff, not the initial plan text
- Day 8 can proceed from a narrower bounded target

---

## Day 8: Scalar-Surface Preparation Design

**Title:** Scalar Design  
**Theme:** Define the smallest real scalar-abstraction seam needed for later
capability widening without overstating support today  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 rerank and the relevant public/header/implementation
   seams.
2. Decide the exact scalar-preparation lane:
   - API-local abstraction
   - internal helper seam
   - documentation boundary
3. Preserve the Sprint 74 truthfulness fence:
   - no fake complex support story
   - no broad numeric-generic conversion implication
4. Fix the Day 9 touched-file fence and the support-only/non-touch set.
5. Record the Day 8 design artifact.

### Deliverables
- Scalar-surface preparation design
- Exact Day 9 touched-file fence
- Preserved truthfulness checklist

### Completion Criteria
- The Day 9 scalar-preparation batch is explicitly bounded before edits begin
- The design prepares later widening without broad capability claims
- Non-touch surfaces are fixed in writing

---

## Day 9: Scalar-Surface Preparation Batch

**Title:** Scalar Batch  
**Theme:** Land the first bounded abstraction or API-local seam needed for
later scalar-type broadening  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded scalar-preparation batch from the Day 8 design.
2. Keep the landing narrow:
   - no broad scalar-generic conversion
   - no algorithm-family widening
3. Add the minimum focused proof required for the touched abstraction seam.
4. Run the full required validation for a substantial `*.c` / `*.h` landing.
5. Record the Day 9 implementation artifact and validation result.

### Deliverables
- First scalar-preparation landing
- Focused proof for the touched seam
- Validated Day 9 artifact

### Completion Criteria
- The touched scalar seam is materially clearer after the landing
- The batch remains bounded and truthful
- Validation passes from the correct authority split

---

## Day 10: Docs / Packaging / Test Alignment Design

**Title:** Follow-Through Design  
**Theme:** Decide the smallest maintained-surface follow-through actually
required by the landed capability work  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Day 6 and Day 9 landings against:
   - maintained public docs
   - install/package surfaces
   - support headers
   - proof-owner surfaces
2. Decide what must now be stated directly about:
   - current width/scalar support
   - bounded modernization status
   - proof-owner interpretation
3. Separate required follow-through from optional or unnecessary doc churn.
4. Fix the exact Day 11 touch set.
5. Record the Day 10 design artifact.

### Deliverables
- Follow-through design
- Exact maintained-surface touch set
- Preserved truthfulness checklist

### Completion Criteria
- Sprint 74 will move only the surfaces the landed code truly requires
- The docs/packaging/test batch is bounded before edits begin
- No broad cleanup spill is left implicit

---

## Day 11: Docs / Packaging / Test Alignment Batch

**Title:** Follow-Through Batch  
**Theme:** Align maintained public, install, and proof wording with the
landed capability contract  
**Time estimate:** 10 hours

### Tasks
1. Update the exact maintained surfaces fixed on Day 10.
2. State the landed capability split directly:
   - current shipped support
   - narrower touched modernization seam
   - still-deferred capability work
3. Keep the Sprint 70 truthfulness fence explicit:
   - no broadened reviewed/platform claims
   - no broader capability claims than the landed seam supports
4. Recheck whether any broader support surfaces actually need edits.
5. Record the Day 11 batch artifact and sanity-check result.

### Deliverables
- Maintained-surface follow-through batch
- Updated packaging/reference wording
- Final touched-surface map

### Completion Criteria
- The public/policy/reference surfaces match the landed code
- No broader wording spill remains on the touched capability lane
- The batch remains bounded to the Day 10 touch set

---

## Day 12: Regression Coverage & Safety Alignment

**Title:** Proof Alignment  
**Theme:** Confirm that the touched capability seams are owned by the right
proof surfaces and add only the minimum missing regression work  
**Time estimate:** 10 hours

### Tasks
1. Re-read the touched tests, headers, and maintained surfaces from the
   post-Day-11 state.
2. Decide whether any focused proof is still missing for:
   - overflow safety
   - touched width/scalar compatibility boundaries
   - narrowed capability-claim truthfulness
3. Add the minimum additional regression only if a real gap remains.
4. Align build/test ownership wording where the sustained contract moved.
5. Record the Day 12 artifact and final proof-surface map.

### Deliverables
- Final proof-surface review
- Any minimum required regression follow-through
- Build/reference alignment notes

### Completion Criteria
- The landed capability boundary has the right focused proof owners
- No redundant or weakly owned regression work is added
- The Day 13 validation queue is explicit

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Validate the landed Sprint 74 branch from the strongest reviewed
baseline and the touched capability surfaces before closeout  
**Time estimate:** 12 hours

### Tasks
1. Run the standard code-day gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest reviewed baseline:
   - `make quality-review-full`
3. Recheck reviewed CMake parity and final reviewed `ctest` counts.
4. Run the highest-signal Sprint 74 follow-ons:
   - touched proof-owner tests
   - representative examples
   - maintained benchmark/reporting surfaces if the landed contract touched
     them
5. Record retained proof/runtime anchors and fix the Day 14 closeout queue
   from the validated state.

### Deliverables
- Full validation artifact
- Retained proof/runtime anchors
- Day 14 closeout queue

### Completion Criteria
- Sprint 74 closes validation from the strongest reviewed baseline
- The touched capability lanes retain the expected proof signals
- Day 14 can close from a real validated package

---

## Day 14: Sprint 74 Closeout & Handoff

**Title:** Closeout and Handoff  
**Theme:** Close Sprint 74 with one explicit first-phase capability package
and a clear carry-forward queue for Sprint 75  
**Time estimate:** 6 hours

### Tasks
1. Write the Sprint 74 closeout artifact summarizing what was fixed in:
   - capability audit and ranking
   - index/scalar architecture contract
   - first index-width readiness landing
   - bounded scalar-surface preparation
   - docs/proof follow-through
   - validated close state
2. Rank the strongest carry-forward items for Sprint 75 and beyond:
   - next capability-modernization phase
   - later algorithm-family widening
   - backend/performance maturity
   - later permanent-surface cleanup
3. Recheck whether `docs/planning/EPIC_7/PROJECT_PLAN.md` needs any Sprint 74
   correction after the landed work.
4. Confirm the final Sprint 74 branch state and validation footprint.
5. Record the final handoff notes for the next sprint.

### Deliverables
- Sprint 74 closeout artifact
- Ranked Sprint 75 carry-forward queue
- Final handoff notes

### Completion Criteria
- Sprint 74 ends with a smaller, more coherent capability package rather than
  a loose set of type-surface edits
- Sprint 75 can begin from a ranked, bounded carry-forward queue
- Any project-plan correction need is explicitly resolved before handoff
