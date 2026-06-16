# Sprint 73 Plan: Configuration Surface Modernization Phase 2

**Sprint Duration:** 14 days  
**Goal:** Finish the highest-value residual configuration modernization work so
advanced solver, graph, and reordering behavior is less dependent on
process-global env vars, compatibility toggles, and debug/profile switches.
This sprint implements the Sprint 73 section of
`docs/planning/EPIC_7/PROJECT_PLAN.md`.

**Starting Point:** Sprint 72 closed with a validated first-phase
product-model convergence package and a ranked carry-forward queue:
- the strongest local reviewed baseline remains `make quality-review-full`
- Sprint 70 fixed the Epic 7 configuration-modernization target and
  truthfulness fence
- Sprint 61 already established the typed-precedence Phase 1 contract
- Sprint 72 reduced product-model ownership blur without reopening public
  docs, capability surfaces, or platform claims
- the strongest remaining configuration pressure is now concentrated in:
  - `SPARSE_FM_*` graph/FM controls
  - residual debug/profile env vars
  - SVD-routing and advanced compatibility controls
  - public wording that still over-relies on process-global control surfaces
- the Sprint 70 non-goal fence is still in force:
  - no generic env-var purge detached from real ownership cost
  - no fake product widening through undocumented internal controls
  - no platform/install claim widening
  - no broad capability or product-model redesign hidden inside config work

The highest-value Sprint 73 work is therefore not a repo-wide configuration
rewrite. It is a bounded second-phase convergence pass on the strongest
residual control seams, with typed or internal-policy integration only where
the live ownership map justifies it, plus proof and docs follow-through only
where the landed behavior truly moves the maintained contract.

**End State:** Sprint 73 leaves behind a smaller, more coherent configuration
surface:
- a ranked residual env-var and compatibility-control map
- one bounded FM/graph policy integration batch
- one bounded debug/profile and advanced-control rationalization batch
- clearer typed vs internal-policy vs compatibility-only ownership wording
- focused regression coverage for touched precedence and compatibility lanes
- a validated Sprint 73 closeout package that lets Sprint 74 start from a
  cleaner modernized configuration contract without reopening the Sprint 70
  truthfulness fence

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
154 hours, staying within the day-cap limit while covering the Sprint 73
implementation scope.

---

## Day 1: Sprint 73 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 73 project-plan scope plus the Sprint 70 and Sprint
72 handoff into a bounded configuration-modernization sprint  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 73 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`,
   the Sprint 70 architecture contract, the Sprint 72 retrospective, and the
   Sprint 72 closeout artifact.
2. Reconfirm the preserved Sprint 73 constraints:
   - no generic env-var purge detached from real ownership cost
   - no public capability widening disguised as configuration cleanup
   - no platform/install/reviewed-contract widening
   - no broad backend or product-model rewrite hidden inside control-surface
     modernization
3. Define the Sprint 73 workstreams explicitly:
   - residual env-var inventory
   - typed/internal policy design
   - FM/graph policy integration
   - debug/profile surface rationalization
   - docs/maintainer/header updates
   - regression coverage
   - validation and closeout
4. Record the strongest likely Sprint 73 touch surfaces:
   - `src/sparse_graph.c`
   - `src/sparse_graph_internal.h`
   - `src/sparse_reorder_nd.c`
   - `include/sparse_analysis.h`
   - `docs/maintainer_guide.md`
   - likely proof owners in graph/reorder and integration surfaces
5. Open Sprint 73 working notes and record the intended landing order,
   required artifacts, and validation expectations.

### Deliverables
- Sprint 73 scope inventory
- Configuration-modernization workstream map
- Working-notes starting assumptions

### Completion Criteria
- Sprint 73 starts from the Sprint 70 and Sprint 72 contract rather than
  reopening broad Epic 7 planning
- The implementation workstreams are explicit before deeper audit begins
- The sprint non-goal fence is fixed before design or landing work proceeds

---

## Day 2: Validation Baseline & Truth-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the implementation-day validation contract and the
highest-signal rerun surfaces Sprint 73 must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 73 authority split:
   - `*.c` / `*.h` landing days require `make format`, `make lint`, and
     `make test`
   - substantial architecture or precedence-boundary batches default to
     `make quality-review-full`
   - docs-only audit/design days use targeted sanity checks only
3. Recheck the live proof surfaces Sprint 73 is most likely to stress:
   - graph/reorder proof owners
   - integration proof owners
   - representative examples and maintained benchmark surfaces
4. Refresh the targeted rerun set most likely to matter in Sprint 73.
5. Record the authoritative validation split in the working notes.

### Deliverables
- Refreshed validation notes
- Sprint 73 rerun checklist
- Preserved proof-surface checklist

### Completion Criteria
- Sprint 73 uses the same reviewed/truthfulness reading fixed in Sprint 70
- The code-day validation contract is explicit before modernization work
  starts
- No rerun ambiguity remains around the likely touched precedence surfaces

---

## Day 3: Residual Env-Var Inventory Audit

**Title:** Inventory Audit  
**Theme:** Re-rank the remaining configuration surfaces by live ownership
cost instead of by historical familiarity  
**Time estimate:** 12 hours

### Tasks
1. Audit the current residual configuration surfaces across:
   - `SPARSE_FM_*` graph/FM controls
   - debug/profile switches
   - SVD-routing controls
   - remaining compatibility env vars
2. Classify the strongest remaining burdens:
   - public process-global surprise
   - duplicated typed/default/env precedence
   - developer-only switches leaking into the public story
   - control surfaces whose behavior is now better owned internally
3. Record where the current configuration surface is already strong and where
   it still behaves like a legacy compatibility shell.
4. Rank the strongest contradiction centers by:
   - caller confusion cost
   - implementation ownership blur
   - likely bounded Sprint 73 payoff
5. Write the first residual-control audit artifact.

### Deliverables
- Residual control-surface audit
- Ranked contradiction list
- First modernization hotspot map

### Completion Criteria
- The broad Sprint 73 configuration problem is reduced to a concrete file and
  seam ranking
- The strongest residual env-var and compatibility contradictions are explicit
- Day 4 can proceed from a real current-state ranking

---

## Day 4: First Modernization Boundary

**Title:** Boundary Freeze  
**Theme:** Refine the ranking and freeze the first modernization fence for the
sprint  
**Time estimate:** 12 hours

### Tasks
1. Re-rank the Day 3 residual controls against:
   - public process-global surprise
   - implementation leverage
   - compatibility risk
   - likely bounded cleanup payoff
2. Separate:
   - first-batch landing surfaces
   - support surfaces that move only if the first batch forces it
   - later or explicitly deferred configuration surfaces
3. Identify the strongest first Sprint 73 fence:
   - FM/graph policy convergence
   - debug/profile rationalization
   - SVD-routing or advanced compatibility cleanup
4. Record the strongest non-goals:
   - no repo-wide removal of every env var
   - no premature public option-surface widening
   - no broad backend or algorithm-family redesign
5. Fix the Day 4 modernization boundary in writing.

### Deliverables
- Refined modernization ranking
- First landing boundary
- Deferred/support-surface map

### Completion Criteria
- The first Sprint 73 landing fence is explicit before design begins
- Lower-value or higher-risk cleanup is clearly separated from the first lane
- Support surfaces are bounded rather than assumed

---

## Day 5: Typed/Internal Policy Design

**Title:** Policy Design  
**Theme:** Define the bounded implementation contract for the first
configuration-modernization landing before edits begin  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 70 configuration target and non-goal fences against the
   first-batch surfaces.
2. Decide the ownership split for the strongest remaining controls:
   - public typed options
   - internal typed policy
   - compatibility-only env overrides
   - retirements or narrowed debug-only behavior
3. Define the precedence rules the batch must preserve:
   - typed values win where the public contract already says they do
   - compatibility env vars remain narrow rather than silently broadened
   - internal developer-only knobs do not leak into the public story
4. Fix the exact first-batch non-touch set:
   - unrelated solver families
   - capability/type surfaces
   - packaging/platform/workflow files
   - broad doc-cleanup spill
5. Record the Day 5 design artifact.

### Deliverables
- Typed/internal policy design
- First-batch non-touch set
- Preserved precedence checklist

### Completion Criteria
- The first modernization batch is explicitly designed before code edits begin
- The landing is bounded to configuration ownership clarity, not generic
  refactoring
- Precedence and non-goal fences are fixed in writing

---

## Day 6: FM/Graph Policy Integration Batch I

**Title:** Policy Batch I  
**Theme:** Land the highest-value typed or internal-policy convergence on the
FM/graph lane  
**Time estimate:** 12 hours

### Tasks
1. Implement the first bounded policy-integration batch in the chosen
   graph/FM configuration center.
2. Remove or narrow the strongest process-global control dependence where the
   design says typed or internal policy should now own the behavior.
3. Preserve documented precedence and compatibility on untouched controls.
4. Add or update the strongest focused proof owner for the landed boundary.
5. Run the required validation gates for touched `*.c` / `*.h` surfaces.

### Deliverables
- First FM/graph policy-integration batch
- Focused proof for the landed boundary
- Validated implementation notes

### Completion Criteria
- One real residual env-var or compatibility seam is materially reduced
- The batch stays inside the Day 5 fence
- Validation passes on the touched implementation surfaces

---

## Day 7: Post-Landing Audit & Rerank

**Title:** Audit and Rerank  
**Theme:** Reassess the residual configuration queue after the first policy
landing  
**Time estimate:** 10 hours

### Tasks
1. Re-read the landed Day 6 batch and its proof surface.
2. Check whether the strongest remaining contradiction is still in the same
   family or has shifted.
3. Re-rank the remaining residual controls:
   - FM/graph follow-through
   - debug/profile rationalization
   - SVD-routing or other compatibility controls
4. Decide whether Sprint 73 should take a second code batch in the same lane
   or switch to the next strongest seam.
5. Record the Day 7 audit artifact.

### Deliverables
- Post-landing audit
- Updated residual-control ranking
- Exact Day 8 target fence

### Completion Criteria
- The post-Day-6 queue is explicit rather than assumed
- The next batch is chosen from the live post-landing state
- Sprint 73 remains bounded instead of drifting into a generic cleanup wave

---

## Day 8: Debug/Profile Rationalization Design

**Title:** Rationalization Design  
**Theme:** Design the bounded second modernization batch around the strongest
remaining developer-only or compatibility spill  
**Time estimate:** 12 hours

### Tasks
1. Re-read the Day 7 rerank and confirm the strongest second batch center.
2. Design the second landing around:
   - removing or bounding developer-only public spill
   - moving residual controls into clearer internal-policy ownership
   - preserving compatibility only where the maintained contract still
     requires it
3. Decide whether any public header or maintainer wording must move with the
   second batch.
4. Fix the exact non-touch set for the second code batch.
5. Record the Day 8 design artifact.

### Deliverables
- Second modernization-batch design
- Support-only follow-through map
- Preserved compatibility checklist

### Completion Criteria
- The second batch is explicitly designed before more code edits begin
- The remaining control seams are narrowed to one ranked center
- Support surfaces are bounded rather than assumed

---

## Day 9: Debug/Profile Rationalization Batch

**Title:** Policy Batch II  
**Theme:** Land the strongest remaining bounded control-surface cleanup from
the post-Day-7 ranking  
**Time estimate:** 12 hours

### Tasks
1. Implement the second bounded configuration batch in the chosen center.
2. Remove or narrow the strongest remaining developer-only or
   compatibility-only public spill from the landed lane.
3. Add or update focused proof for the new precedence or compatibility rule.
4. Keep unrelated solver families, capability work, and platform surfaces out
   of scope.
5. Run the required validation gates for touched `*.c` / `*.h` surfaces.

### Deliverables
- Second modernization batch
- Focused precedence/compatibility proof
- Validated implementation notes

### Completion Criteria
- A second real residual configuration seam is materially reduced
- The landing stays inside the Day 8 fence
- Validation passes on the touched implementation surfaces

---

## Day 10: Docs / Maintainer / Header Follow-Through Design

**Title:** Follow-Through Design  
**Theme:** Decide the smallest maintained-surface follow-through required by
the landed configuration work  
**Time estimate:** 10 hours

### Tasks
1. Re-read the landed Day 6 and Day 9 batches against:
   - `docs/maintainer_guide.md`
   - likely touched public headers
   - support surfaces only if wording truly moved
2. Decide what must now be stated directly about:
   - typed vs internal-policy vs compatibility-only ownership
   - preserved precedence rules
   - developer-only surface boundaries
3. Separate required follow-through from optional or unnecessary doc churn.
4. Fix the exact Day 11 touch set.
5. Record the Day 10 design artifact.

### Deliverables
- Follow-through design
- Exact maintained-surface touch set
- Preserved truthfulness checklist

### Completion Criteria
- Sprint 73 will move only the surfaces the landed code truly requires
- The docs/maintainer/header batch is bounded before edits begin
- No broad cleanup spill is left implicit

---

## Day 11: Docs / Maintainer / Header Follow-Through Batch

**Title:** Follow-Through Batch  
**Theme:** Align maintained public, header, and policy wording with the landed
configuration contract  
**Time estimate:** 10 hours

### Tasks
1. Update the exact maintained surfaces fixed on Day 10.
2. State the landed typed/internal-policy/compatibility split directly.
3. Keep the Sprint 70 truthfulness fence explicit:
   - no broadened reviewed/platform claims
   - no broader capability claims
   - no silent product-surface widening
4. Recheck whether any broader support surfaces actually need edits.
5. Record the Day 11 batch artifact and sanity-check result.

### Deliverables
- Maintained-surface follow-through batch
- Updated policy/reference wording
- Final touched-surface map

### Completion Criteria
- The public/policy/reference surfaces match the landed code
- No broader wording spill remains on the touched configuration lane
- The batch remains bounded to the Day 10 touch set

---

## Day 12: Regression Coverage & Build Alignment

**Title:** Proof Alignment  
**Theme:** Confirm that the touched precedence and compatibility lanes are
owned by the right proof surfaces and build contract  
**Time estimate:** 10 hours

### Tasks
1. Re-read the touched tests, headers, and maintainer surfaces from the
   post-Day-11 state.
2. Decide whether any focused proof is still missing for:
   - typed/default/env precedence
   - narrowed compatibility-only overrides
   - internal developer-only control boundaries
3. Add the minimum additional regression only if a real gap remains.
4. Align build/test ownership wording where the sustained contract moved.
5. Record the Day 12 artifact and final proof-surface map.

### Deliverables
- Final proof-surface review
- Any minimum required regression follow-through
- Build/reference alignment notes

### Completion Criteria
- The landed configuration boundary has the right focused proof owners
- No redundant or weakly owned regression work is added
- The Day 13 validation queue is explicit

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Validate the landed Sprint 73 branch from the strongest reviewed
baseline and the touched configuration surfaces before closeout  
**Time estimate:** 12 hours

### Tasks
1. Run the standard code-day gate:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest reviewed baseline:
   - `make quality-review-full`
3. Recheck reviewed CMake parity and final reviewed `ctest` counts.
4. Run the highest-signal Sprint 73 follow-ons:
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
- Sprint 73 closes validation from the strongest reviewed baseline
- The touched configuration lanes retain the expected proof signals
- Day 14 can close from a real validated package

---

## Day 14: Sprint 73 Closeout & Handoff

**Title:** Closeout and Handoff  
**Theme:** Close Sprint 73 with one explicit configuration-modernization
package and a clear carry-forward queue for Sprint 74  
**Time estimate:** 10 hours

### Tasks
1. Write the Sprint 73 closeout artifact summarizing what was fixed in:
   - residual env-var inventory and ranking
   - FM/graph policy integration
   - debug/profile or compatibility-surface rationalization
   - docs/maintainer/header follow-through
   - proof alignment
   - validated close state
2. Rank the strongest carry-forward items for Sprint 74 and beyond:
   - next configuration-modernization phase
   - capability modernization
   - backend/performance maturity
   - later permanent-surface cleanup
3. Recheck whether `docs/planning/EPIC_7/PROJECT_PLAN.md` needs any Sprint 73
   correction after the landed work.
4. Confirm the final Sprint 73 branch state and validation footprint.
5. Record the final handoff notes for the next sprint.

### Deliverables
- Sprint 73 closeout artifact
- Ranked Sprint 74 carry-forward queue
- Final handoff notes

### Completion Criteria
- Sprint 73 ends with a smaller, more coherent residual configuration package
  rather than a loose set of control-surface edits
- Sprint 74 can begin from a ranked, bounded carry-forward queue
- Any project-plan correction need is explicitly resolved before handoff
