# Sprint 68 Plan: Giant-Test Refactor Phase 2 & Numerical Assurance Expansion

**Sprint Duration:** 14 days  
**Goal:** Continue reducing giant-test maintenance cost while adding stronger
second-layer numerical and workflow assurance on the hardest remaining paths.
This sprint implements the Sprint 68 section of
`docs/planning/EPIC_6/PROJECT_PLAN.md`.

**Starting Point:** Sprint 67 closed with the strongest remaining large-source
ownership seams reduced and validated:
- `make quality-review-full` remains the strongest local reviewed baseline
- reviewed CMake parity remains a maintained truthfulness anchor
- graph/reorder ownership extraction and shared ND policy convergence are now
  landed enough to freeze those seams
- the large-`n` Cholesky analysis-to-CSC handoff is aligned enough that Sprint
  68 can focus on giant-test maintainability and second-layer assurance rather
  than re-opening the same implementation boundaries
- the next Epic 6 priority is no longer packaging, ABI, or source-ownership
  phase 3 work; it is shrinking the maintenance cost of the largest remaining
  tests while strengthening confidence on the hardest numerical lanes

The highest-value next work is not broad new functionality. It is a bounded
test-and-assurance sprint focused on re-ranking the remaining giant-test
hotspots, extracting or splitting the highest-value seams, adding stronger
cross-method or oracle checks where the hardest paths still deserve more
assurance, expanding bounded property/fuzz coverage where it materially pays
off, tightening the platform-test confidence story, and closing from the
reviewed baseline.

**End State:** Sprint 68 leaves behind one coherent giant-test and assurance
package:
- a refreshed ranked map of the largest remaining giant-test hotspots
- bounded helper extraction or file-splitting on the highest-value test seams
- stronger differential/oracle coverage on the hardest numerical paths
- bounded property/fuzz expansion where it materially improves assurance
- a clearer platform-specific test-confidence story
- full validation and closeout from the landed state

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
164 hours, matching the Sprint 68 estimate and staying below the 168-hour
limit.

---

## Day 1: Sprint 68 Scope Audit & Giant-Test Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 68 project-plan scope plus the Sprint 67 validated
close into a bounded giant-test and assurance implementation map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 68 section of
   `docs/planning/EPIC_6/PROJECT_PLAN.md`, the Sprint 67 retrospective, and
   the strongest Sprint 67 closeout artifacts.
2. Reconfirm the preserved Sprint 68 constraints:
   - no fake assurance wins that only add brittle golden outputs
   - no broad solver-feature work disguised as test refactoring
   - no weakening of the reviewed truthfulness contract
   - no widening the platform-confidence story beyond reviewed evidence
3. Define the Sprint 68 workstreams explicitly:
   - giant-test residual audit
   - giant-test refactor batch
   - differential/oracle coverage
   - property/fuzz expansion
   - platform-test follow-through
   - validation and closeout
4. Record the strongest likely Sprint 68 touch surfaces:
   - remaining giant tests
   - supporting proof/oracle helpers
   - user-facing confidence/docs surfaces
5. Open Sprint 68 working notes and record intended landing order, required
   artifacts, and validation expectations.

### Deliverables
- Sprint 68 scope inventory
- Giant-test baseline map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 68 starts from the Sprint 67 validated close rather than reopening
  implementation maintainability phase 3 work
- The giant-test and assurance workstreams are explicit before deeper audit
  begins
- The sprint non-goal fence is fixed before design or code edits land

---

## Day 2: Validation Baseline & Giant-Test/Proof Rerun Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed baseline and rerun set that Sprint 68
test-refactor and assurance work must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity counts
   - current quality/truthfulness wording
2. Reconfirm the mandatory gate for later `*.c` / `*.h` days:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial assurance or
   test-architecture work:
   - `make quality-review-full`
4. Refresh the targeted rerun set most likely to matter in Sprint 68:
   - giant direct-family tests
   - giant graph/reorder and iterative/eigensolver tests
   - representative examples and maintained benchmark/reporting surfaces that
     should not drift
5. Record the authoritative validation split for docs-only, bounded code-day,
   and substantial assurance days.

### Deliverables
- Refreshed validation notes
- Sprint 68 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 68 uses the same reviewed baseline wording and parity anchors as the
  live repo
- The authoritative rerun set is explicit before implementation work begins
- No validation ambiguity remains around docs-only versus code-touching days

---

## Day 3: Giant-Test Residual Audit

**Title:** Giant-Test Audit I  
**Theme:** Re-rank the largest remaining test surfaces by maintenance cost,
oracle weakness, and refactor payoff  
**Time estimate:** 12 hours

### Tasks
1. Inventory the strongest remaining giant tests across:
   - CSC direct families
   - graph/reorder
   - iterative/eigensolver
   - integration and numerical cross-check lanes
2. Classify each hotspot by:
   - maintenance pain
   - helper-extraction safety
   - likely oracle weakness
   - likely user-visible confidence risk if left untouched
3. Identify the strongest current contradictions:
   - too many unrelated scenarios in one permanent test file
   - repeated fixture/build helpers embedded locally
   - hard paths with only one proof style when a second oracle would pay off
4. Rank the most valuable Sprint 68 refactor and assurance candidates.
5. Write the audit artifact with the explicit hotspot map.

### Deliverables
- Live giant-test inventory
- Ranked giant-test candidate list
- Initial refactor/oracle shortlist

### Completion Criteria
- The broad “giant-test refactor and assurance expansion” claim is reduced to a
  concrete file and seam map
- The strongest maintenance and assurance contradictions are explicit before
  redesign begins
- Day 4 can proceed from a real current-state hotspot ranking instead of
  generic testing concerns

---

## Day 4: Audit Follow-Through & First-Landing Boundary

**Title:** Giant-Test Audit II  
**Theme:** Separate the must-land Sprint 68 test seams from later or lower
value cleanup candidates  
**Time estimate:** 12 hours

### Tasks
1. Re-rank the Day 3 hotspot set against the Epic 6 giant-test target.
2. Separate:
   - first landing giant-test refactor seams
   - second landing oracle/property seams
   - later residuals that should stay out of Sprint 68
3. Confirm which touched files likely belong in:
   - helper extraction or file split batches
   - proof-only support
   - confidence/docs alignment only
4. Fix the first Sprint 68 implementation target set in writing.
5. Record the residual queue that Sprint 68 should not absorb.

### Deliverables
- Refined hotspot ranking
- First landing boundary
- Deferred residual map

### Completion Criteria
- The Sprint 68 target set is smaller and sharper than the original epic-level
  review
- The first landing boundary is explicit before design begins
- Lower-value cleanup is clearly separated from the bounded Sprint 68 lane

---

## Day 5: Giant-Test Refactor Design

**Title:** Refactor Design  
**Theme:** Define the extraction or split contract for the strongest remaining
giant-test seams  
**Time estimate:** 12 hours

### Tasks
1. Design the test refactor contract for the selected first landing:
   - owned helper boundaries
   - file split versus local extraction criteria
   - fixture/oracle reuse expectations
2. Define the preserved compatibility rules:
   - no behavioral drift disguised as test cleanup
   - no fake abstraction layer with unclear ownership
   - no widening into unrelated solver or build surfaces
3. Decide which pieces belong in:
   - new helper sections or support files
   - touched existing giant tests
   - proof/support files only
4. Record the exact safety contract for the first implementation batch.
5. Fix the likely file fence for the Day 6-7 landing set.

### Deliverables
- Giant-test refactor design artifact
- Explicit safety/compatibility contract
- First implementation fence

### Completion Criteria
- The first giant-test extraction story is explicit before edits start
- Ownership boundaries are separated clearly enough to prevent churn
- The converged design is tight enough to support a bounded landing

---

## Day 6: Giant-Test Refactor Batch 1

**Title:** Refactor Batch I  
**Theme:** Land the first bounded helper extraction or split on the
highest-value giant-test seam  
**Time estimate:** 12 hours

### Tasks
1. Implement the first giant-test refactor batch inside the Day 5 fence.
2. Keep the batch bounded to:
   - highest-value helper or split movement
   - minimal touched-call-site rewiring
   - proof updates required by the new boundary
3. Remove or tighten stale local test commentary only where the touched code
   would otherwise become harder to read.
4. Run required code-day validation:
   - `make format`
   - `make lint`
   - `make test`
5. Record landed behavior, touched files, and any residual queue sharpened by
   the implementation.

### Deliverables
- First landed giant-test refactor batch
- Updated proof surface
- Validation notes

### Completion Criteria
- The first giant-test seam is materially easier to maintain
- The batch stays inside the bounded fence rather than widening into generic
  test churn
- Required validation passes before the sprint proceeds

---

## Day 7: Post-Landing Audit & Assurance Rerank

**Title:** Post-Landing Audit  
**Theme:** Re-rank the remaining queue after the first test refactor batch and
fix the next assurance target  
**Time estimate:** 12 hours

### Tasks
1. Re-read the landed Day 6 state and measure what maintenance pressure it
   actually removed.
2. Re-rank the remaining Sprint 68 queue across:
   - residual giant-test seams
   - differential/oracle opportunities
   - property/fuzz candidates
3. Decide whether a second refactor batch is still the best next move or
   whether assurance now has higher value.
4. Fix the exact Day 8-10 target set in writing.
5. Record the remaining deferred test-maintenance queue explicitly.

### Deliverables
- Post-landing audit artifact
- Assurance target rerank
- Updated deferred queue

### Completion Criteria
- The next Sprint 68 move is chosen from the landed branch state, not from the
  original backlog wording
- The strongest remaining assurance gap is explicit before design begins
- The queue is smaller and sharper than it was at Day 5

---

## Day 8: Differential/Oracle Coverage Design

**Title:** Oracle Design  
**Theme:** Define the strongest bounded second-layer numerical assurance batch
for the hardest remaining path  
**Time estimate:** 12 hours

### Tasks
1. Choose the highest-value oracle/parity target from the Day 7 rerank.
2. Design the assurance contract around:
   - cross-method parity
   - external-style oracle inputs where appropriate
   - expected tolerances and failure classification
3. Decide which touched files likely belong in:
   - giant tests
   - support helpers
   - examples/benchmarks/docs only if wording must move
4. Record the exact safety contract for the assurance landing.
5. Fix the likely file fence for the Day 9 batch.

### Deliverables
- Oracle coverage design artifact
- Explicit tolerance and safety contract
- Bounded implementation fence

### Completion Criteria
- The new assurance batch is explicit before code moves
- The second-layer proof is additive rather than duplicative or brittle
- The file fence is small enough to support a truthful bounded landing

---

## Day 9: Differential/Oracle Coverage Batch

**Title:** Oracle Batch  
**Theme:** Land the bounded second-layer numerical assurance batch on the
chosen hard path  
**Time estimate:** 12 hours

### Tasks
1. Implement the Day 8 assurance batch within the defined fence.
2. Keep the batch bounded to:
   - one hard-path oracle/parity lane
   - minimal helper support
   - proof wording updates only where the landed semantics require it
3. Run required code-day validation:
   - `make format`
   - `make lint`
   - `make test`
4. Record landed proof behavior and any sharpened residual queue.
5. Recheck that the batch improved assurance without widening into feature
   work.

### Deliverables
- Landed oracle/parity batch
- Updated proof surface
- Validation notes

### Completion Criteria
- One hard numerical lane now has stronger second-layer assurance
- The batch remains bounded and behavior-preserving
- Required validation passes before the sprint proceeds

---

## Day 10: Property/Fuzz Expansion Batch

**Title:** Property/Fuzz Batch  
**Theme:** Add bounded generative or property coverage where it materially
improves assurance  
**Time estimate:** 12 hours

### Tasks
1. Choose the highest-value bounded property/fuzz candidate from the current
   post-Day-9 state.
2. Implement the smallest meaningful expansion across:
   - existing fuzz/property surfaces
   - helper support only if necessary
   - failure reporting that stays interpretable
3. Keep the batch bounded to materially useful invariants rather than generic
   randomized volume.
4. Run required code-day validation:
   - `make format`
   - `make lint`
   - `make test`
5. Record what the new property/fuzz lane now proves and what still remains
   deferred.

### Deliverables
- Landed property/fuzz expansion
- Updated assurance notes
- Validation notes

### Completion Criteria
- The new coverage adds real assurance rather than noisy random execution
- The expansion remains bounded and maintainable
- Required validation passes before the sprint proceeds

---

## Day 11: Platform-Test Confidence Follow-Through

**Title:** Platform Follow-Through  
**Theme:** Improve or better document the reduced-platform test-confidence
story where Sprint 68 actually moved proof ownership  
**Time estimate:** 12 hours

### Tasks
1. Re-read the live platform-specific test-confidence wording after the landed
   Sprint 68 batches.
2. Identify where giant-test or assurance changes require:
   - clearer platform confidence wording
   - narrower or sharper workflow comments
   - explicit documentation of reduced coverage
3. Update only the touched command/workflow/docs surfaces that the landed test
   changes actually affect.
4. Run targeted sanity checks or stronger validation if the touched command
   story requires it.
5. Record the preserved deferred platform queue explicitly.

### Deliverables
- Platform-test follow-through artifact
- Updated confidence wording
- Deferred residual map

### Completion Criteria
- Platform-specific confidence claims remain truthful after Sprint 68 changes
- The batch stays bounded to touched proof/command surfaces
- No fake platform closure is implied beyond reviewed evidence

---

## Day 12: Docs & Regression-Surface Alignment

**Title:** Alignment Follow-Through  
**Theme:** Align maintained docs and regression-surface ownership wording to
the landed Sprint 68 test and assurance boundaries  
**Time estimate:** 12 hours

### Tasks
1. Re-read the live post-Day-11 proof split across:
   - giant tests
   - property/fuzz surfaces
   - examples/benchmarks/docs
2. Update maintained truth surfaces only where the landed Sprint 68 work moved
   ownership or confidence interpretation.
3. Confirm the docs say plainly:
   - which tests own the new assurance lanes
   - which benchmark/example surfaces do not own those guarantees
   - which platform lanes are reviewed versus supplemental
4. Run targeted docs-only sanity checks:
   - diff review
   - terminology checks
   - touched-surface measurement
5. Record the final pre-validation alignment state.

### Deliverables
- Docs/regression alignment artifact
- Updated maintained truth surfaces
- Final pre-validation notes

### Completion Criteria
- The maintained docs match the landed Sprint 68 proof split
- No stale ownership wording remains on touched surfaces
- The branch is ready for full validation

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Prove the landed Sprint 68 branch from the strongest reviewed
baseline plus the targeted assurance surfaces  
**Time estimate:** 12 hours

### Tasks
1. Run the bounded code-day validation gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the stronger reviewed baseline:
   - `make quality-review-full`
3. Re-run the targeted Sprint 68 follow-on surfaces:
   - giant tests touched directly
   - representative examples
   - maintained benchmark/reporting surfaces
4. Capture retained representative outputs and parity anchors.
5. Record any non-blocking residual notes that remain after validation.

### Deliverables
- Full validation artifact
- Retained representative outputs
- Final parity and reviewed-baseline notes

### Completion Criteria
- The full validation sweep passes from the landed Sprint 68 tree
- Reviewed parity anchors stay explicit and exact
- Any remaining note is clearly non-blocking and documented

---

## Day 14: Closeout & Handoff

**Title:** Closeout  
**Theme:** Close Sprint 68 from the validated baseline and hand off one
truthful next-step queue  
**Time estimate:** 12 hours

### Tasks
1. Re-read the landed Sprint 68 branch from the Day 13 validated state.
2. Write the closeout and handoff artifact covering:
   - shipped test-maintainability outcomes
   - shipped second-layer assurance outcomes
   - preserved compatibility/truthfulness fences
   - ranked carry-forward queue
3. Recheck whether the Sprint 68 section of the project plan needs any
   correction based on the actual landing.
4. Confirm the branch close state and final deferred queue explicitly.
5. Prepare the sprint for retrospective and PR handoff.

### Deliverables
- Sprint 68 closeout artifact
- Ranked carry-forward queue
- Ready-for-retrospective close state

### Completion Criteria
- Sprint 68 closes from a validated branch state
- The shipped giant-test and assurance package is explicit in writing
- The handoff queue is sharper and more honest than the sprint’s starting
  backlog
