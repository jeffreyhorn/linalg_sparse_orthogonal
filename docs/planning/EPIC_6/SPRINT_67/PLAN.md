# Sprint 67 Plan: Large-Source Maintainability Phase 3

**Sprint Duration:** 14 days  
**Goal:** Reduce the strongest remaining large-source ownership hotspots after
Epic 5 and the Epic 6 packaging/productization close, especially where
graph/reorder, CSC, iterative, configuration, and stale sprint-local
commentary still compete inside oversized permanent implementation files. This
sprint implements the Sprint 67 section of
`docs/planning/EPIC_6/PROJECT_PLAN.md`.

**Starting Point:** Sprint 66 closed with the packaging, ABI, and
platform-quality convergence package landed and validated:
- `make quality-review-full` remains the strongest local reviewed baseline
- reviewed CMake parity remains a maintained truthfulness anchor
- the install/export/package story is now explicitly static-first and bounded
- the platform truthfulness split is clarified enough to avoid reopening build
  surfaces during maintainability work
- the next Epic 6 priority is no longer packaging/productization churn; it is
  shrinking and clarifying the strongest remaining implementation hotspots

The next highest-value work is not another feature sprint. It is a bounded
maintainability sprint focused on re-ranking the remaining hotspot files,
extracting the most clearly owned graph/reorder and CSC/iterative seams,
removing stale sprint-history commentary from touched permanent surfaces,
realigning build/tests around the new boundaries, and closing from the
reviewed baseline.

**End State:** Sprint 67 leaves behind one coherent maintainability package:
- a refreshed ranked audit of the strongest remaining oversized implementation
  hotspots
- bounded graph/reorder decomposition on the highest-value ownership seams
- bounded CSC or iterative residual decomposition where it still pays off after
  Epic 5
- less sprint-local chronology in permanent implementation/header surfaces
- updated build/test alignment around the resulting ownership boundaries
- full validation and closeout from the landed state

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
156 hours, matching the Sprint 67 estimate and staying below the 168-hour
limit.

---

## Day 1: Sprint 67 Scope Audit & Maintainability Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 67 project-plan scope plus the Sprint 66 validated
close into a bounded large-source maintainability implementation map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 67 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 66 retrospective, and
   the strongest Sprint 66 closeout artifacts.
2. Reconfirm the preserved Sprint 67 constraints:
   - no fake maintainability wins that blur real ownership
   - no broad feature work disguised as decomposition
   - no reopening packaging/platform/build-surface churn unless a touched seam
     truly requires it
   - no weakening of the reviewed truthfulness contract
3. Define the Sprint 67 workstreams explicitly:
   - residual hotspot audit
   - graph/reorder decomposition
   - CSC/iterative residual decomposition
   - comment/chronology cleanup
   - build/regression alignment
   - validation and closeout
4. Record the strongest likely Sprint 67 touch surfaces:
   - remaining oversized implementation files
   - touched public headers
   - tests/build surfaces likely to move with extracted ownership seams
5. Open Sprint 67 working notes and record intended landing order, required
   artifacts, and validation expectations.

### Deliverables
- Sprint 67 scope inventory
- Maintainability baseline map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 67 starts from the Sprint 66 validated close rather than reopening
  packaging/platform-first work
- The maintainability workstreams are explicit before deeper audit begins
- The sprint non-goal fence is fixed before design or code edits land

---

## Day 2: Validation Baseline & Hotspot/Proof Rerun Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed baseline and rerun set that Sprint 67
decomposition work must preserve  
**Time estimate:** 8 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity counts
   - current quality/truthfulness wording
2. Reconfirm the mandatory gate for later `*.c` / `*.h` days:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial decomposition or
   ownership-boundary work:
   - `make quality-review-full`
4. Refresh the targeted rerun set most likely to matter in Sprint 67:
   - graph/reorder proofs
   - CSC and iterative proofs
   - representative examples and maintained benchmark/reporting surfaces that
     should not drift
5. Record the authoritative validation split for docs-only, bounded code-day,
   and substantial decomposition days.

### Deliverables
- Refreshed validation notes
- Sprint 67 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 67 uses the same reviewed baseline wording and parity anchors as the
  live repo
- The authoritative rerun set is explicit before implementation work begins
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Residual Hotspot Audit

**Title:** Hotspot Audit I  
**Theme:** Re-rank the remaining oversized production files by ownership pain,
proof burden, and payoff  
**Time estimate:** 12 hours

### Tasks
1. Inventory the strongest remaining large-source production files across:
   - graph/reorder
   - CSC direct paths
   - iterative/eigensolver residuals
   - configuration/orchestration spillover
2. Classify each hotspot by:
   - mixed ownership pain
   - extraction safety
   - likely proof burden
   - likely user-visible risk if left untouched
3. Identify the strongest current contradictions:
   - orchestration mixed with owned local helpers
   - family-local logic mixed with cross-family policy
   - stale sprint-local chronology obscuring durable explanations
4. Rank the most valuable Sprint 67 decomposition candidates.
5. Write the audit artifact with the explicit hotspot map.

### Deliverables
- Live hotspot inventory
- Ranked hotspot-candidate list
- Initial decomposition shortlist

### Completion Criteria
- The broad “large-source maintainability” claim is reduced to a concrete file
  and seam map
- The strongest ownership contradictions are explicit before redesign begins
- Day 4 can proceed from a real current-state hotspot ranking instead of generic
  cleanup concerns

---

## Day 4: Hotspot Follow-Through & First-Landing Boundary

**Title:** Hotspot Audit II  
**Theme:** Separate the must-land Phase 3 ownership seams from later or lower
value cleanup candidates  
**Time estimate:** 11 hours

### Tasks
1. Re-rank the Day 3 hotspot set against the Epic 6 maintainability target.
2. Separate:
   - first landing graph/reorder seams
   - second landing CSC/iterative seams
   - later residuals that should stay out of Sprint 67
3. Confirm which touched files likely belong in:
   - extraction batches
   - proof-only support
   - comment/chronology cleanup only
4. Fix the first Sprint 67 implementation target set in writing.
5. Record the residual queue that Sprint 67 should not absorb.

### Deliverables
- Refined hotspot ranking
- First landing boundary
- Deferred residual map

### Completion Criteria
- The Sprint 67 target set is smaller and sharper than the original epic-level
  review
- The first landing boundary is explicit before design begins
- Lower-value cleanup is clearly separated from the bounded Sprint 67 lane

---

## Day 5: Graph/Reorder Decomposition Design

**Title:** Graph Design  
**Theme:** Define the extraction and ownership contract for the strongest
remaining graph/reorder hotspot seams  
**Time estimate:** 12 hours

### Tasks
1. Design the graph/reorder decomposition contract for the selected first
   landing:
   - owned helper boundaries
   - orchestration versus family-local responsibilities
   - touched header/internal interface expectations
2. Define the preserved compatibility rules:
   - no behavior drift disguised as refactoring
   - no fake abstraction layer with unclear ownership
   - no widening into unrelated graph or solver families
3. Decide which pieces belong in:
   - new or extracted implementation helpers
   - touched existing implementation files
   - proof/support files only
4. Record the exact safety contract for the first implementation batch.
5. Fix the likely file fence for the Day 6-8 graph/reorder landing set.

### Deliverables
- Graph/reorder decomposition design artifact
- Explicit safety/compatibility contract
- First implementation fence

### Completion Criteria
- The first graph/reorder extraction story is explicit before edits start
- Ownership boundaries are separated clearly enough to prevent churn
- The converged design is tight enough to support a bounded landing

---

## Day 6: Graph/Reorder Decomposition Batch 1

**Title:** Graph Batch I  
**Theme:** Land the first bounded graph/reorder extraction on the highest-value
owned seam  
**Time estimate:** 12 hours

### Tasks
1. Implement the first graph/reorder extraction batch inside the Day 5 fence.
2. Keep the batch bounded to:
   - highest-value owned helper movement
   - minimal touched-call-site rewiring
   - proof updates required by the new ownership boundary
3. Remove or tighten stale local commentary only where the touched code would
   otherwise become harder to read.
4. Run required code-day validation:
   - `make format`
   - `make lint`
   - `make test`
5. Record landed behavior, touched files, and any residual queue sharpened by
   the implementation.

### Deliverables
- First landed graph/reorder decomposition batch
- Updated proof surface
- Day 6 implementation artifact

### Completion Criteria
- The first graph/reorder seam is smaller and more clearly owned than before
- Behavior and validation remain clean
- The batch stays inside the Day 5 fence without widening into unrelated files

---

## Day 7: Graph/Reorder Decomposition Batch 2

**Title:** Graph Batch II  
**Theme:** Finish the bounded graph/reorder follow-through required by the first
extraction  
**Time estimate:** 11 hours

### Tasks
1. Land only the remaining graph/reorder follow-through justified by Day 6:
   - companion helper extraction
   - remaining call-site cleanup
   - necessary proof tightening
2. Confirm the new ownership story reads coherently in the touched permanent
   files.
3. Keep the batch out of:
   - CSC/iterative seams
   - packaging/build churn
   - broad documentation rewrite
4. Run required code-day validation:
   - `make format`
   - `make lint`
   - `make test`
5. Record what the graph/reorder lane no longer needs in Sprint 67.

### Deliverables
- Completed bounded graph/reorder landing
- Refined residual queue
- Day 7 implementation artifact

### Completion Criteria
- The first target family now has a coherent ownership split
- Residual graph/reorder work is explicitly smaller after the landing
- Validation remains clean and the batch does not widen

---

## Day 8: Post-Graph Audit & CSC/Iterative Rerank

**Title:** Post-Graph Audit  
**Theme:** Re-rank the remaining Sprint 67 queue after the graph/reorder
landing and fix the second implementation target  
**Time estimate:** 10 hours

### Tasks
1. Audit the post-Day-7 branch state across the remaining hotspot seams.
2. Re-rank the strongest residual candidates among:
   - CSC decomposition
   - iterative residual decomposition
   - chronology cleanup only
3. Decide whether CSC or iterative owns the stronger second landing.
4. Fix the exact likely file fence for the Day 9-10 implementation batch.
5. Record the proof surfaces most likely to move with that second landing.

### Deliverables
- Post-graph residual ranking
- Exact second target selection
- Updated implementation fence

### Completion Criteria
- The remaining queue is smaller and more concrete after the first landing
- The second target is explicit before more code moves
- Lower-value residuals remain clearly deferred

---

## Day 9: CSC/Iterative Residual Decomposition Design

**Title:** Residual Design  
**Theme:** Define the extraction and safety contract for the strongest remaining
CSC or iterative maintainability seam  
**Time estimate:** 12 hours

### Tasks
1. Design the selected second landing:
   - owned helper boundaries
   - orchestrator versus local algorithm responsibilities
   - touched proof/support expectations
2. Define the preserved compatibility rules:
   - no semantic drift under the cover of source decomposition
   - no fake symmetry between unrelated solver families
   - no reopening closed Sprint 66 productization surfaces
3. Decide which pieces belong in:
   - new/extracted helpers
   - touched existing implementation files
   - tests only
4. Record the exact safety contract for the implementation batch.
5. Fix the likely file fence for Day 10 and any optional support-only files.

### Deliverables
- CSC/iterative decomposition design artifact
- Explicit compatibility contract
- Second implementation fence

### Completion Criteria
- The second landing contract is explicit before edits start
- Ownership boundaries are concrete enough to implement without churn
- The file fence is narrow and defensible

---

## Day 10: CSC/Iterative Residual Decomposition Batch

**Title:** Residual Batch  
**Theme:** Land the bounded second decomposition batch on the selected CSC or
iterative seam  
**Time estimate:** 12 hours

### Tasks
1. Implement the second decomposition batch inside the Day 9 fence.
2. Keep the batch bounded to:
   - the selected owned seam
   - minimal rewiring required by extraction
   - proof updates required by the new ownership boundary
3. Remove or tighten stale local chronology only where the touched code would
   otherwise stay misleading.
4. Run required code-day validation:
   - `make format`
   - `make lint`
   - `make test`
5. Record landed behavior, touched files, and what remains explicitly deferred.

### Deliverables
- Landed CSC/iterative decomposition batch
- Updated proof surface
- Day 10 implementation artifact

### Completion Criteria
- The second target seam is smaller and more clearly owned than before
- Behavior and validation remain clean
- The batch stays inside the Day 9 fence without reopening unrelated surfaces

---

## Day 11: Comment & Chronology Cleanup

**Title:** Chronology Cleanup  
**Theme:** Remove stale sprint-history commentary from touched permanent files
while preserving durable technical explanations  
**Time estimate:** 9 hours

### Tasks
1. Re-read the permanent implementation/header files touched in Days 6-10.
2. Identify comments that are:
   - stale sprint chronology
   - temporary landing notes
   - no longer the strongest durable explanation
3. Replace or remove only the commentary whose continued presence now harms
   maintainability.
4. Keep durable technical explanation where it still pays for future readers.
5. Run the appropriate validation for the actual touched files and record the
   cleanup rules used.

### Deliverables
- Reduced stale chronology in touched permanent files
- Comment-cleanup rules recorded in artifact/notes
- Cleaner post-landing source narrative

### Completion Criteria
- Permanent touched files read less like sprint archaeology
- Durable explanations are preserved where they still matter
- Cleanup does not widen into unrelated stylistic churn

---

## Day 12: Build & Regression Alignment

**Title:** Build Alignment  
**Theme:** Update build/test/regression surfaces for the new ownership
boundaries and touched files  
**Time estimate:** 10 hours

### Tasks
1. Reconcile build and proof surfaces affected by the landed decompositions:
   - target/source lists
   - local helper visibility
   - touched tests
   - maintained command/reporting surfaces if needed
2. Confirm the new ownership boundaries are reflected truthfully in build/test
   organization.
3. Land only the bounded follow-through required by the new file boundaries.
4. Run the stronger validation path if the touched surfaces justify it:
   - `make quality-review-full`
   - plus any bounded focused reruns needed by touched proof surfaces
5. Record the exact maintained contract after alignment.

### Deliverables
- Build/test alignment batch
- Maintained contract notes
- Day 12 alignment artifact

### Completion Criteria
- Build and proof surfaces match the landed ownership boundaries
- No stale source-list or regression-surface contradiction remains
- Validation is strong enough for the touched alignment scope

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Prove the full Sprint 67 branch from the landed maintainability
state and retained reviewed anchors  
**Time estimate:** 12 hours

### Tasks
1. Run the full required validation set:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm the maintained reviewed anchors:
   - reviewed CMake parity counts
   - Makefile/CMake parity
   - full reviewed CMake `ctest`
3. Rerun any targeted family-local or integration surfaces most affected by the
   decomposition batches.
4. Capture representative retained outputs and any non-blocking residual notes.
5. Write the full validation artifact.

### Deliverables
- Full validation record
- Retained parity/quality metrics
- Day 13 validation artifact

### Completion Criteria
- The full Sprint 67 branch validates cleanly
- Maintained reviewed anchors remain exact
- The maintained source decomposition story is backed by real proof, not just
  structure changes

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 67 from the validated branch state and hand off the
remaining residual maintainability queue cleanly  
**Time estimate:** 5 hours

### Tasks
1. Summarize the landed Sprint 67 package:
   - hotspot audit and rerank
   - graph/reorder decomposition
   - CSC/iterative residual decomposition
   - comment/chronology cleanup
   - build/regression alignment
   - validated close
2. Record the preserved compatibility and truthfulness fence.
3. Rank the remaining post-Sprint-67 maintainability queue for the next sprint.
4. Recheck the Sprint 67 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md` for any correction needed after the
   landed branch state.
5. Write the closeout/handoff artifact and working-notes finish.

### Deliverables
- Sprint 67 closeout summary
- Ranked carry-forward queue
- Day 14 handoff artifact

### Completion Criteria
- Sprint 67 closes from the validated landed state rather than intent alone
- The maintained compatibility/truthfulness fence is explicit
- The next-sprint queue is ranked clearly enough to avoid reopening today’s
  decisions
