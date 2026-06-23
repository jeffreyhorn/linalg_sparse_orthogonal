# Sprint 85 Plan: Large-Source Maintainability Phase 5

**Sprint Duration:** 14 days  
**Goal:** Reduce the highest remaining implementation and giant-test
maintainability costs after Sprint 84’s bounded assurance widening. This
sprint implements the Sprint 85 section of
`docs/planning/EPIC_8/PROJECT_PLAN.md`.

**Starting Point:** Sprint 84 closed from a validated assurance baseline with
one maintained direct-family external differential lane, one deeper
deterministic large-`n` seeded-property owner, and one stronger shared
failure-path lifecycle owner. The strongest remaining first-tier Epic 8
contradiction is now maintainability cost:
- `src/sparse_iterative.c` still concentrates a large share of iterative
  ownership and review burden
- one major direct-family source hotspot still needs bounded decomposition
  after the Sprint 84 proof widening
- at least one giant proof-owner test still carries dense registration/helper
  concentration
- Sprint 84’s assurance widening should not be reopened as another generic
  proof-ownership rewrite
- the strongest local reviewed baseline remains `make quality-review-full`

The highest-value Sprint 85 work is therefore not generic “refactor more
files.” It is one bounded maintainability-modernization package that:
- reranks the largest remaining implementation and proof hotspots from the
  current tree
- fixes one explicit extraction/decomposition contract across the strongest
  source and giant-test seams
- lands one bounded cleanup inside `src/sparse_iterative.c`
- lands one bounded cleanup on the strongest next direct-family source hotspot
- lands one bounded giant-test architecture cleanup on the strongest next
  proof-owner concentration
- reconciles touched proof-owner and maintainer surfaces without weakening the
  bounded Sprint 84 assurance reading

**End State:** Sprint 85 leaves behind:
- a refreshed hotspot-priority and maintainability map
- one explicit extraction/decomposition contract
- one landed iterative-source cleanup batch
- one landed direct-family hotspot cleanup batch
- one landed giant-test architecture cleanup batch
- one focused proof/docs alignment package
- one validated closeout baseline and Sprint 86-ready handoff

**Time budget:** Each day is capped at 12 hours as requested. Because that cap
allows at most `168` hours across 14 days, this day-by-day plan totals `168`
hours rather than the higher project-plan estimate of `~178` hours, while
preserving the Sprint 85 scope and ordering.

---

## Day 1: Sprint 85 Scope Audit & Hotspot Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 85 project-plan section and Sprint 84 closeout into
one bounded maintainability execution package  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 85 section of
   `docs/planning/EPIC_8/PROJECT_PLAN.md`, the Sprint 84 closeout artifact, and
   the Sprint 84 retrospective.
2. Reconfirm the preserved Sprint 85 starting assumptions:
   - Sprint 80 already fixed the long-range hotspot map premise
   - Sprint 84 already widened the highest-value assurance seams that needed
     proof-first treatment
   - Sprint 85 should not reopen generic assurance widening
   - Sprint 85 should not claim architectural simplification outside the
     touched hotspots
3. Define the Sprint 85 workstreams explicitly:
   - hotspot rerank
   - decomposition design
   - iterative-source cleanup
   - direct-family hotspot cleanup
   - giant-test architecture cleanup
   - proof/docs alignment
   - validation and closeout
4. Record the strongest likely Sprint 85 touch surfaces:
   - `src/sparse_iterative.c`
   - the strongest remaining direct-family source hotspot
   - the strongest next giant proof-owner test
   - touched maintainer/proof-owner wording surfaces
5. Open Sprint 85 working notes and record intended landing order and
   validation expectations.

### Deliverables
- Sprint 85 scope inventory
- maintainability workstream map
- starting working-notes baseline

### Completion Criteria
- Sprint 85 starts from the validated Sprint 84 end state
- the first maintainability contradiction is explicit before deeper audit
  begins
- the non-goal fence is visible before design or implementation work

---

## Day 2: Validation & Proof-Surface Recheck

**Title:** Validation Recheck  
**Theme:** Refresh the strongest reviewed, proof-owner, benchmark, and
install/export validation split before maintainability changes begin  
**Time estimate:** 12 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline and implementation-day gate:
   - `make quality-review-full`
   - `make format`
   - `make lint`
   - `make test`
2. Reconfirm reviewed CMake parity, canonical benchmark-report ownership, and
   install/export proof ownership.
3. Recheck the representative proof-owner and source surfaces most likely to
   move during Sprint 85:
   - iterative proof owners
   - direct-family proof owners
   - strongest giant-test owners
   - representative examples
   - canonical benchmark/reporting command surfaces
   - install/export proof scripts
4. Fix the authoritative rerun list most likely to matter throughout Sprint
   85.
5. Record the validation/proof split in working notes and a Day 2 artifact.

### Deliverables
- refreshed validation-baseline artifact
- preserved proof-owner map
- authoritative Sprint 85 rerun list

### Completion Criteria
- the strongest local validation contract is explicit before implementation
  work lands
- proof ownership across reviewed tests, benchmarks, and install/export
  surfaces is fixed in writing
- later code days have no ambiguity about the required validation gate

---

## Day 3: Hotspot Rerank

**Title:** Hotspot Audit  
**Theme:** Re-rank the highest-value implementation and giant-test
maintainability hotspots from the live tree  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the highest-signal implementation and proof-owner hotspots:
   - `src/sparse_iterative.c`
   - strongest direct-family source hotspots
   - strongest giant proof-owner tests
   - support surfaces that genuinely follow from touched ownership changes
2. Identify where the current maintainability ceiling is strongest:
   - oversized mixed-responsibility sources
   - dense helper concentration
   - registration concentration in giant tests
   - ownership seams that are harder to review after Sprint 84 proof widening
3. Separate:
   - strongest first-batch implementation center
   - second-tier source and test follow-through seams
   - support-only maintainer/docs surfaces
   - deliberate non-goals
4. Reconcile the rerank against Sprint 78 cleanup history and Sprint 84
   assurance closeout.
5. Write the ranked hotspot artifact.

### Deliverables
- ranked hotspot artifact
- first-tier vs deferred seam map
- Sprint 78/84 carry-forward reconciliation notes

### Completion Criteria
- Sprint 85’s broad maintainability problem is reduced to one ranked live map
- the strongest implementation center is explicit before boundary design
- lower-value spillover work is clearly separated from the first lane

---

## Day 4: First Maintainability Boundary Freeze

**Title:** Boundary Freeze  
**Theme:** Fix the first bounded extraction/decomposition fence and the allowed
ownership movement  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 3 ranking against the Sprint 85 project-plan scope.
2. Decide the required first implementation center:
   - iterative source cleanup first
   - direct-family hotspot cleanup second
   - giant-test architecture cleanup only where the first batch truly forces
     it
3. Decide which support surfaces move only if forced:
   - proof-owner tests
   - maintainer wording
   - support docs
   - package/runtime surfaces
4. Fix the preserved non-goal fence for the first landing:
   - no broad algorithm rewriting detached from ownership cleanup
   - no proof dilution from moving helpers without preserving owners
   - no repo-wide “cleanup sweep” claim
   - no support-surface churn detached from a real landed hotspot seam
5. Record the first implementation fence in working notes and a Day 4
   artifact.

### Deliverables
- first maintainability-boundary artifact
- required vs support-only touch set
- preserved first-batch non-goal fence

### Completion Criteria
- Sprint 85 has one explicit first landing boundary
- support-only surfaces are clearly separated from the batch center
- Day 5 can design one decomposition contract instead of a broad cleanup
  rewrite

---

## Day 5: Decomposition / Ownership Architecture Design

**Title:** Decomposition Design  
**Theme:** Define the bounded extraction and ownership contract Sprint 85 will
actually land  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 4 boundary and the strongest maintainability
   contradictions.
2. Define the ownership split for:
   - iterative-source extraction seams
   - direct-family hotspot decomposition seams
   - giant-test helper and registration seams
   - retained proof-owner boundaries after extraction
3. Decide how the first landing preserves Sprint 84’s assurance owners while
   reducing source and proof concentration.
4. Fix the touch fence for code, tests, docs, and package/runtime surfaces.
5. Write the Day 5 architecture artifact and working-notes design summary.

### Deliverables
- decomposition / ownership architecture contract
- ownership split for touched seams
- preserved bounded-cleanup and non-goal fence

### Completion Criteria
- Sprint 85 has one explicit implementation contract
- ownership between source extraction and proof preservation is clear
- Day 6 can implement one bounded landing without reopening design questions

---

## Day 6: Iterative Source Cleanup Batch 1

**Title:** Iterative Batch  
**Theme:** Land the highest-value bounded cleanup inside `src/sparse_iterative.c`  
**Time estimate:** 12 hours

### Tasks
1. Implement the highest-value iterative cleanup seam from the Day 5
   contract.
2. Keep the landing bounded to the required first implementation center.
3. Preserve the existing proof-owner split and avoid accidental algorithm or
   contract movement outside the touched ownership seam.
4. Reconcile only the directly forced iterative proof-owner or maintainer
   follow-through.
5. Run the required implementation-day validation gate and record the batch
   result.

### Deliverables
- landed bounded iterative-source cleanup batch
- any directly forced proof-owner/support follow-through
- Day 6 artifact and working-notes implementation record

### Completion Criteria
- one real iterative ownership or decomposition contradiction is reduced in
  code
- proof ownership remains explicit after the cleanup
- the batch stays inside the Day 5 fence and passes the required validation

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the remaining Sprint 85 contradiction map after the first
source cleanup lands  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 6 landing and measure what contradiction it actually
   removed.
2. Re-rank the remaining Sprint 85 queue:
   - direct-family hotspot cleanup
   - giant-test architecture cleanup
   - support-only alignment
3. Decide whether the strongest next move is still a source hotspot or has
   shifted to a giant-test concentration seam.
4. Fix the exact Day 8 design center in writing.
5. Record the rerank in working notes and a Day 7 artifact.

### Deliverables
- post-landing rerank artifact
- updated remaining contradiction map
- fixed Day 8 design center

### Completion Criteria
- Sprint 85’s second implementation center is explicit after the Day 6 result
- lower-value spillover work remains clearly separated
- Day 8 can design the right next seam instead of assuming the original rank
  order survived unchanged

---

## Day 8: Direct-Family Hotspot Design

**Title:** Direct-Family Design  
**Theme:** Define the next bounded direct-family maintainability cleanup seam  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 rerank and the strongest remaining direct-family source
   hotspot.
2. Decide the exact extraction or ownership cleanup that reduces review and
   reasoning cost without reopening solver-family contracts broadly.
3. Fix which proof-owner tests and support surfaces move only if forced.
4. Separate the Day 9 required center from lower-value adjacent cleanup.
5. Write the Day 8 design artifact and working-notes summary.

### Deliverables
- direct-family hotspot design artifact
- exact Day 9 touch fence
- preserved bounded-cleanup reading

### Completion Criteria
- Sprint 85 has one explicit second implementation contract
- the next direct-family cleanup is bounded and reviewable
- Day 9 can land without reopening generic family-wide refactoring

---

## Day 9: Direct-Family Hotspot Batch

**Title:** Direct-Family Batch  
**Theme:** Land the next bounded cleanup on one major direct-family source
hotspot  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 8 direct-family cleanup seam.
2. Preserve the public and proof-owner contract while reducing local source
   concentration.
3. Reconcile only the directly forced proof-owner, helper, or maintainer
   follow-through.
4. Run the required implementation-day validation gate.
5. Record the landed batch in working notes and a Day 9 artifact.

### Deliverables
- landed direct-family hotspot cleanup batch
- any directly forced follow-through
- Day 9 artifact and implementation record

### Completion Criteria
- one real direct-family maintainability hotspot is reduced in code
- proof ownership remains explicit after the cleanup
- the batch stays inside the Day 8 fence and passes the required validation

---

## Day 10: Giant-Test Architecture Design

**Title:** Giant-Test Design  
**Theme:** Define the strongest next bounded proof-owner concentration cleanup  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 9 landing and rerank the remaining giant-test
   concentration seams.
2. Decide the exact test-owner architecture cleanup to land next:
   - helper extraction
   - registration reduction
   - local fixture organization
3. Fix which support-only surfaces move only if truly forced.
4. Keep the design bounded so proof ownership becomes clearer rather than more
   diffuse.
5. Write the Day 10 design artifact and working-notes summary.

### Deliverables
- giant-test architecture design artifact
- exact Day 11 touch fence
- preserved proof-owner clarity fence

### Completion Criteria
- Sprint 85 has one explicit third implementation contract
- the giant-test cleanup is bounded and ownership-preserving
- Day 11 can land one clear proof-surface cleanup without weakening
  correctness ownership

---

## Day 11: Giant-Test Architecture Batch

**Title:** Giant-Test Batch  
**Theme:** Reduce the densest remaining proof-owner registration/helper
concentration in one giant test  
**Time estimate:** 12 hours

### Tasks
1. Implement the bounded giant-test architecture cleanup from the Day 10
   contract.
2. Preserve the touched proof-owner’s authority while reducing local review
   and reasoning cost.
3. Reconcile only the directly forced helper, support, or maintainer
   follow-through.
4. Run the required implementation-day validation gate.
5. Record the landed batch in working notes and a Day 11 artifact.

### Deliverables
- landed giant-test architecture cleanup batch
- any directly forced follow-through
- Day 11 artifact and implementation record

### Completion Criteria
- one real giant-test concentration hotspot is reduced
- proof ownership remains explicit and truthful after cleanup
- the batch stays inside the Day 10 fence and passes the required validation

---

## Day 12: Proof / Docs Alignment & Validation Queue Freeze

**Title:** Alignment  
**Theme:** Reconcile touched proof-owner and maintainer surfaces with the
landed hotspot boundaries and fix the final validation queue  
**Time estimate:** 12 hours

### Tasks
1. Re-read the landed Day 6, Day 9, and Day 11 cleanup surfaces.
2. Recheck whether any touched proof-owner, README, or maintainer wording
   still overstates or understates the refined hotspot boundaries.
3. Reconfirm whether install/export, package, benchmark, or runtime support
   surfaces truly need movement.
4. Fix the exact Day 13 validation queue in writing.
5. Record the final proof/docs alignment and validation queue in working notes
   and a Day 12 artifact.

### Deliverables
- proof/docs alignment artifact
- frozen Day 13 validation queue
- final touched-surface truth map

### Completion Criteria
- no support-only drift remains before the full sweep
- the final validation queue is explicit and unambiguous
- Day 13 can execute from a fixed truth map rather than re-deciding scope

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full Sprint 85 validation queue and capture the measured
close baseline  
**Time estimate:** 12 hours

### Tasks
1. Run the implementation-day gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest reviewed baseline:
   - `make quality-review-full`
3. Reconfirm reviewed CMake parity.
4. Rerun the focused reviewed proof-owner tests, representative examples, and
   relevant benchmark/reporting commands fixed on Day 12.
5. Capture retained outputs and any non-blocking runtime notes in a Day 13
   artifact and working notes.

### Deliverables
- full validation artifact
- validated close baseline
- explicit retained proof-owner outputs

### Completion Criteria
- the full Sprint 85 validation queue passes
- the reviewed anchors stay exact
- Day 14 can close from measured evidence instead of implementation state

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 85 from the validated baseline and fix the Sprint 86
handoff queue  
**Time estimate:** 12 hours

### Tasks
1. Re-read the full Sprint 85 branch package:
   - rerank
   - decomposition contract
   - iterative cleanup
   - direct-family cleanup
   - giant-test cleanup
   - alignment and validation
2. Recheck the Sprint 85 project-plan section against the landed work.
3. Reconcile the branch outcome against the next Epic 8 sprint ordering.
4. Write the Day 14 closeout and handoff artifact.
5. Record the final Sprint 85 close state in working notes.

### Deliverables
- Sprint 85 closeout/handoff artifact
- final working-notes closeout entry
- explicit Sprint 86-ready carry-forward queue

### Completion Criteria
- Sprint 85 closes from the validated Day 13 baseline
- the landed maintainability package is summarized truthfully and boundedly
- the next Epic 8 contradiction center is fixed explicitly for the following
  sprint
