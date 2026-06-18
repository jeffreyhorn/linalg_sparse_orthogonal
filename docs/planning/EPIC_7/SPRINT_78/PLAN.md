# Sprint 78 Plan: Large-Source Maintainability Phase 4 & Giant-Test Architecture

**Sprint Duration:** 14 days  
**Goal:** Reduce the strongest remaining permanent implementation and proof
hotspots without reopening broad subsystem redesign, while removing avoidable
sprint-history debt from the touched long-lived source and test files. This
sprint implements the Sprint 78 section of
`docs/planning/EPIC_7/PROJECT_PLAN.md`.

**Starting Point:** Sprint 77 closed with a validated packaging/install/platform
package and preserved the bounded Epic 7 truthfulness fence. Sprint 78 starts
from the strongest remaining permanent maintainability pressure:
- Sprint 70 fixed the hotspot and proof-surface inventory
- Sprint 72-77 reduced several product, capability, backend, benchmark, and
  packaging seams without reopening broad subsystem rewrites
- the strongest local reviewed baseline remains `make quality-review-full`
- the highest-value remaining Sprint 78 work is now:
  - source hotspot rerank
  - one bounded source decomposition batch
  - giant-test rerank
  - one bounded giant-test architecture batch
  - chronology/comment cleanup on touched permanent files
  - docs and proof-ownership alignment
  - validation and closeout
- the preserved Epic 7 non-goal fence is still in force:
  - no broad subsystem redesign disguised as maintainability cleanup
  - no giant-test breakup wave across every proof owner
  - no fake history purge that removes durable algorithm explanation
  - no widening of platform, backend, or capability claims beyond maintained
    evidence

The highest-value Sprint 78 work is therefore not “make every large file
small.” It is one bounded maintainability phase on the strongest remaining
permanent review hotspots, with clearer ownership seams, less chronology debt,
and more durable proof architecture on the touched giant-test surfaces.

**End State:** Sprint 78 leaves behind a smaller and more coherent permanent
review surface:
- a refreshed ranked source-hotspot map
- one landed bounded source decomposition or ownership split
- a refreshed ranked giant-test map
- one landed bounded giant-test architecture cleanup
- less sprint-history/comment debt in touched permanent code and proof files
- aligned docs and maintainer-policy reading of the refined ownership seams
- a validated Sprint 78 closeout package that lets Sprint 79 start from a
  stronger long-source/giant-test baseline instead of a mixed hotspot backlog

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
158 hours, staying within the day-cap limit while covering the Sprint 78
implementation scope.

---

## Day 1: Sprint 78 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 78 project-plan scope plus the Sprint 70 hotspot
inventory into one bounded long-source and giant-test sprint  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 78 section of
   `docs/planning/EPIC_7/PROJECT_PLAN.md`, the Sprint 70 hotspot artifacts,
   and the Sprint 77 closeout package.
2. Reconfirm the preserved Sprint 78 constraints:
   - no broad subsystem redesign
   - no giant-test breakup campaign across every proof owner
   - no durable algorithm explanation removed as “history cleanup”
   - no reopening of unrelated product, capability, backend, or platform work
3. Define the Sprint 78 workstreams explicitly:
   - source hotspot rerank
   - bounded source decomposition
   - giant-test rerank
   - bounded giant-test architecture batch
   - chronology/comment cleanup
   - docs and proof-ownership alignment
   - validation and closeout
4. Record the strongest likely Sprint 78 touch surfaces:
   - the largest remaining implementation files
   - the largest remaining permanent proof-owner tests
   - support docs or maintainer policy likely to move if ownership boundaries
     change
5. Open Sprint 78 working notes and record the intended landing order,
   artifacts, and validation expectations.

### Deliverables
- Sprint 78 scope inventory
- Long-source and giant-test workstream map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 78 starts from the Sprint 70 hotspot inventory rather than reopening
  broad Epic 7 planning
- The implementation and proof workstreams are explicit before deeper audit
  begins
- The sprint non-goal fence is fixed before design or landing work proceeds

---

## Day 2: Validation Baseline & Hotspot Truth Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the implementation-day validation contract and the
highest-signal rerun surfaces Sprint 78 must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 78 authority split:
   - `*.c` / `*.h` landing days require `make format`, `make lint`, and
     `make test`
   - substantial maintainability or proof-architecture batches default to
     `make quality-review-full`
   - docs-only audit/design days use targeted sanity checks only
3. Recheck the live proof surfaces Sprint 78 is most likely to stress:
   - large proof-owner tests
   - reviewed CMake proof-owner tests and representative examples
   - maintainer-policy surfaces that describe touched proof ownership
4. Refresh the targeted rerun set most likely to matter in Sprint 78.
5. Record the authoritative validation split in the working notes.

### Deliverables
- Refreshed validation notes
- Sprint 78 rerun checklist
- Preserved proof-surface checklist

### Completion Criteria
- Sprint 78 uses the same reviewed/truthfulness reading fixed in Sprint 70
- The code-day validation contract is explicit before source/test work starts
- No rerun ambiguity remains around the likely touched maintainability seams

---

## Day 3: Source Hotspot Re-audit

**Title:** Source Audit  
**Theme:** Re-rank the largest remaining implementation files by review pain,
ownership ambiguity, and bounded extraction value  
**Time estimate:** 12 hours

### Tasks
1. Audit the current large implementation surfaces across:
   - size and chronology density
   - mixed ownership roles
   - helper extraction potential
   - coupling risk
2. Classify the strongest remaining source-hotspot contradictions:
   - mixed orchestration and algorithm detail
   - helper-density without local boundaries
   - policy/comment spill
   - proof-sensitive ownership ambiguity
3. Record where the current large sources are already coherent and where they
   still behave like review bottlenecks.
4. Rank the strongest contradiction centers by:
   - downstream review payoff
   - compatibility risk
   - extraction safety
   - bounded Sprint 78 value
5. Write the source-hotspot audit artifact.

### Deliverables
- Source-hotspot re-audit
- Ranked contradiction list
- First implementation-hotspot map

### Completion Criteria
- The broad Sprint 78 source problem is reduced to a concrete seam ranking
- The highest-value implementation hotspot is explicit
- Day 4 can proceed from a real current-state ranking

---

## Day 4: Source Boundary Freeze

**Title:** Source Boundary  
**Theme:** Refine the ranking and freeze the first implementation fence for
the sprint  
**Time estimate:** 11 hours

### Tasks
1. Re-rank the Day 3 source hotspots against:
   - review payoff
   - extraction clarity
   - compatibility risk
   - bounded cleanup value
2. Separate:
   - first-batch landing surfaces
   - support surfaces that move only if the first batch forces them
   - later or explicitly deferred maintainability work
3. Identify the strongest first Sprint 78 fence:
   - implementation decomposition lane
   - helper ownership lane
   - chronology cleanup lane
4. Record the strongest non-goals:
   - no broad algorithm rewrite
   - no subsystem API redesign
   - no helper explosion without ownership gain
   - no unrelated proof or platform work
5. Fix the Day 4 boundary in writing.

### Deliverables
- Refined source-hotspot ranking
- First landing boundary
- Deferred/support-surface map

### Completion Criteria
- The first Sprint 78 implementation fence is explicit before design begins
- Lower-value or higher-risk source work is clearly separated from the first
  lane
- Support surfaces are bounded rather than assumed

---

## Day 5: Source Decomposition Design

**Title:** Source Design  
**Theme:** Define the bounded implementation contract for the first source
decomposition batch before edits begin  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 4 source boundary against the strongest first-batch files.
2. Decide the ownership split for the first landing:
   - orchestration owner
   - helper owner
   - local policy/comment owner
   - proof-sensitive boundary owner
3. Define the guarantees the batch must preserve:
   - behavior compatibility
   - current validation surface
   - local helper visibility and call sequencing
   - current public contract
4. Fix the exact first-batch non-touch set:
   - unrelated giant tests
   - unrelated backend/capability code
   - broader taxonomy cleanup
   - non-touched subsystem comments
5. Record the Day 5 design artifact.

### Deliverables
- Source decomposition design
- First-batch non-touch set
- Preserved compatibility checklist

### Completion Criteria
- The first source batch is explicitly designed before edits begin
- The landing is bounded to implementation ownership cleanup
- Compatibility and non-goal fences are fixed in writing

---

## Day 6: Source Decomposition Batch

**Title:** Source Batch  
**Theme:** Land the highest-value bounded implementation extraction or
ownership split justified by the refreshed audit  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 5 bounded source decomposition changes.
2. Keep the batch inside the first-fence touch set and preserve the current
   behavior contract.
3. Reconcile touched helper ownership or local comment boundaries across the
   minimal required surfaces only.
4. Run the required validation gate for the touched surface type.
5. Record the landing artifact and note any giant-test or support follow-through
   needed for Day 7 or Day 8.

### Deliverables
- Landed source decomposition batch
- Updated bounded implementation ownership surface
- Recorded validation result for the landing day

### Completion Criteria
- One real implementation maintainability batch lands
- The batch does not widen into subsystem redesign
- The touched source still reads as one coherent bounded ownership surface

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-audit the permanent hotspot surface after the first landing to
find the strongest remaining seam  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Day 6 landed source surfaces.
2. Re-rank the remaining Sprint 78 contradictions against:
   - review friction removed
   - proof gaps still visible
   - chronology debt still visible
   - bounded follow-through value
3. Decide whether the strongest remaining seam is now:
   - giant-test architecture
   - chronology/comment cleanup
   - docs/proof-ownership drift
4. Fix the exact Day 8 audit/design center in writing.
5. Record the rerank artifact.

### Deliverables
- Post-landing audit
- Updated seam ranking
- Exact next-batch center

### Completion Criteria
- Sprint 78 does not repeat the first landing unnecessarily
- The strongest remaining seam is explicitly reranked from the landed state
- Day 8 starts from a current-state ranking rather than the original backlog

---

## Day 8: Giant-Test Re-audit

**Title:** Giant-Test Audit  
**Theme:** Re-rank the largest remaining permanent proof files by review cost,
chronology density, and platform or proof value  
**Time estimate:** 12 hours

### Tasks
1. Audit the current giant-test surfaces across:
   - file size and chronology density
   - mixed proof roles
   - helper extraction potential
   - platform- or workflow-sensitive ownership
2. Classify the strongest remaining giant-test contradictions:
   - mixed lifecycle/property/regression roles
   - inline helper density
   - chronology/comment spill
   - weak proof taxonomy boundaries
3. Record where the current large proof files are already coherent and where
   they still behave like review bottlenecks.
4. Rank the strongest contradiction centers by:
   - review payoff
   - proof-owner clarity
   - extraction safety
   - bounded Sprint 78 value
5. Write the giant-test audit artifact.

### Deliverables
- Giant-test re-audit
- Ranked proof-hotspot list
- Exact architecture-batch candidate map

### Completion Criteria
- The broad giant-test problem is reduced to a concrete seam ranking
- The highest-value permanent proof hotspot is explicit
- Day 9 can proceed from a real proof-architecture target

---

## Day 9: Giant-Test Architecture Design

**Title:** Giant-Test Design  
**Theme:** Define the bounded helper/split/taxonomy cleanup for the strongest
remaining permanent proof surface  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Day 8 giant-test rerank artifact.
2. Decide the exact Day 10 touch set across:
   - strongest giant-test file(s)
   - helper or harness support surfaces
   - support-only maintainer/doc surfaces if truly forced
3. Define the preserved proof guarantees:
   - current behavior coverage
   - current proof-owner reading
   - platform-sensitive proof interpretation
   - current validation path
4. Fix the exact non-touch set:
   - unrelated source files
   - unrelated giant tests
   - broad taxonomy rewrites
   - unrelated workflow or platform claims
5. Record the Day 9 design artifact.

### Deliverables
- Giant-test architecture design
- Day 10 touch set
- Preserved proof checklist

### Completion Criteria
- The giant-test architecture batch is explicitly bounded before edits begin
- The proof-owner and validation fence is fixed in writing
- Day 10 can land without reopening the whole proof surface

---

## Day 10: Giant-Test Architecture Batch

**Title:** Giant-Test Batch  
**Theme:** Land the highest-value bounded helper, split, or taxonomy cleanup
on the strongest remaining permanent proof surface  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 9 bounded giant-test architecture changes.
2. Keep the batch inside the preserved proof-owner and validation fence.
3. Update only the support surfaces truly required by the landing.
4. Run the required validation gate for the touched surface type.
5. Record the landing artifact, retained proof, and any remaining chronology or
   support drift.

### Deliverables
- Landed giant-test architecture batch
- Updated permanent proof surface
- Recorded validation result for the landing day

### Completion Criteria
- One real giant-test maintainability batch lands
- The landing does not widen into broad proof-taxonomy churn
- The touched proof surface still reads as one coherent bounded owner

---

## Day 11: Chronology & Comment Cleanup

**Title:** Chronology Cleanup  
**Theme:** Remove stale sprint-history debt from the touched permanent source
and test files while preserving durable explanation  
**Time estimate:** 11 hours

### Tasks
1. Re-read the Day 6 and Day 10 landed permanent files for chronology spill.
2. Distinguish removable sprint-history/comment debt from durable explanation:
   - algorithm rationale
   - ownership rationale
   - validation-sensitive caveats
3. Clean the touched permanent files so they keep:
   - durable technical explanation
   - accurate ownership language
   - no stale sprint-number chronology where it no longer adds value
4. Recheck that the cleanup does not erase proof-sensitive context.
5. Record the Day 11 cleanup artifact.

### Deliverables
- Touched permanent files with less chronology debt
- Recorded chronology-cleanup pass
- Explicit preserved durable-explanation checklist

### Completion Criteria
- Touched permanent files no longer carry avoidable sprint-history spill
- Durable algorithm and proof explanations remain intact
- The sprint does not turn chronology cleanup into content erasure

---

## Day 12: Docs & Proof-Ownership Alignment

**Title:** Ownership Alignment  
**Theme:** Reconcile touched docs and maintainer policy with the refined
source and proof ownership boundaries  
**Time estimate:** 11 hours

### Tasks
1. Re-read the touched source, proof, and chronology-cleaned surfaces.
2. Re-read the strongest support surfaces likely to drift:
   - `docs/maintainer_guide.md`
   - any touched public docs that describe the refined owner
3. Decide whether any new focused regression code is truly needed or whether
   the remaining gap is ownership wording only.
4. Fix the exact Day 13 validation queue in writing.
5. Record the Day 12 alignment artifact.

### Deliverables
- Docs/proof-ownership alignment pass
- Final Sprint 78 validation queue
- Explicit no-op note if no new support edits are required

### Completion Criteria
- The touched maintainability package is described by the right support owners
- The final validation queue is explicit before Day 13
- No ownership ambiguity remains around the landed Sprint 78 package

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Validate the full Sprint 78 package from the landed Day 12 state  
**Time estimate:** 12 hours

### Tasks
1. Run the standard code-day validation gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest reviewed baseline:
   - `make quality-review-full`
3. Reconfirm the reviewed CMake parity anchor.
4. Re-run the focused Sprint 78 follow-ons from the Day 12 queue:
   - touched source/proof owners
   - representative reviewed tests/examples
   - any touched support or workflow surfaces
5. Record the retained outputs, reviewed anchors, and any non-blocking runtime
   notes.

### Deliverables
- Full validation-sweep artifact
- Explicit reviewed anchor set
- Final Sprint 78 validated baseline

### Completion Criteria
- All required validation passes
- Reviewed parity remains exact
- The branch has one explicit validated close baseline before Day 14

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Finalize Sprint 78, fix the preserved maintainability contract, and
hand off the ranked next queue for Sprint 79  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 78 project-plan section and the Day 13 validation
   artifact.
2. Summarize the final Sprint 78 package:
   - refreshed source-hotspot audit
   - landed bounded source decomposition work
   - refreshed giant-test audit
   - landed bounded giant-test architecture work
   - chronology/comment cleanup
   - docs/proof-ownership alignment
3. Restate the preserved Sprint 78 truthfulness and non-goal fence.
4. Rank the carry-forward queue for Sprint 79 and later Epic 7 work.
5. Record the closeout artifact and handoff notes.

### Deliverables
- Sprint 78 closeout artifact
- Ranked Sprint 79 carry-forward queue
- Explicit preserved maintainability/non-goal contract

### Completion Criteria
- Sprint 78 ends from one explicit validated close baseline
- The maintained source/proof ownership contract is fixed in writing
- Sprint 79 inherits a ranked hotspot queue rather than a mixed maintainability
  backlog
