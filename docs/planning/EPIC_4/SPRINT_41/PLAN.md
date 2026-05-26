# Sprint 41 Plan: Shared Allocation, Overflow & Internal Utility Consolidation

**Sprint Duration:** 14 days  
**Goal:** Centralize overflow checks, allocation sizing, and related internal safety helpers so the repository stops carrying multiple local safety idioms that drift over time, while beginning the bounded internal-first groundwork Sprint 40 handed forward for later lifecycle and hotspot refactors. This sprint implements the Sprint 41 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`.

**Starting Point:** Sprint 40 closed with a stable Epic 4 architecture contract: lifecycle/state taxonomy, handle-model migration strategy, quality-truth ownership map, public migration-risk audit, and a validation anchor. Sprint 41 begins from that defined baseline and focuses on internal utility consolidation rather than public API churn.

**End State:** Sprint 41 leaves behind a shared internal allocation/overflow utility layer, migrated core helper hotspots, broader `src/` safety alignment, selected auxiliary-surface cleanup, bounded internal-first prep rules for later refactors, and a validated safety-helper baseline for Sprint 42 and beyond.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 136 hours, matching the Sprint 41 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 41 Scope Audit & Safety-Pattern Baseline

**Title:** Baseline Setup  
**Theme:** Convert the Sprint 41 project-plan items into a measured helper-consolidation work inventory  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 41 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`, the Sprint 40 architecture-contract synthesis, and the Epic 4 remediation plan so Sprint 41 starts from the documented helper-consolidation scope.
2. Reconfirm the preserved Sprint 40 contracts that Sprint 41 must not reopen:
   - internal-first groundwork
   - quality-truth ownership map
   - validation anchor
   - public migration-risk boundaries
3. Define the Sprint 41 workstreams explicitly:
   - helper-pattern inventory
   - shared utility design
   - first core-module migration
   - broader `src/` migration
   - auxiliary-surface alignment
   - prep-rule documentation
   - validation closeout
4. Record the authoritative helper hotspots already named in the project plan and review:
   - `src/sparse_dense.c`
   - `src/sparse_svd.c`
   - `src/sparse_eigs.c`
   - `src/sparse_etree.c`
5. Open Sprint 41 working notes and record the scope, assumptions, and migration order.

### Deliverables
- Sprint 41 scope inventory
- Initial helper-consolidation workstream map
- Working-notes baseline assumptions

### Completion Criteria
- Sprint 41 starts from the documented Epic 4 plan and Sprint 40 contract
- The helper-consolidation surfaces are explicit before code migration begins
- The preserved internal-first and validation contracts are named up front

---

## Day 2: Helper-Pattern Inventory Across Core Modules

**Title:** Pattern Inventory  
**Theme:** Measure the local allocation/overflow helper patterns that Sprint 41 must consolidate  
**Time estimate:** 8 hours

### Tasks
1. Inventory local size-multiplication, bounds-check, and allocation helper idioms in:
   - `src/sparse_dense.c`
   - `src/sparse_svd.c`
   - `src/sparse_eigs.c`
   - `src/sparse_etree.c`
2. Classify the helper patterns into explicit buckets:
   - size multiplication overflow checks
   - `idx_t`/`size_t` representability checks
   - count-to-bytes conversions
   - common allocation/free/reset helpers
3. Identify where helpers are structurally identical versus semantically specialized.
4. Record which helper patterns are already close to a shared internal utility API and which still need interface design.
5. Write the helper-pattern baseline artifact.

### Deliverables
- Helper-pattern inventory
- Raw helper-duplication map
- Initial shared-vs-specialized helper classification

### Completion Criteria
- The first migration targets are grounded in actual duplicated logic
- Helper families are classified before shared API design starts
- Later implementation work can distinguish true consolidation from necessary specialization

---

## Day 3: Shared Utility API Design

**Title:** Utility Design  
**Theme:** Define the shared internal safety-helper layer before implementing it  
**Time estimate:** 10 hours

### Tasks
1. Design the shared internal utility interface for:
   - size multiplication overflow checks
   - `idx_t`/`size_t` bounds and representability checks
   - safe allocation-size derivation helpers
   - common allocation convenience wrappers where justified
2. Decide the target file layout and naming for the new utility layer.
3. Define which helpers should be macro-like, inline, or normal function entry points.
4. Record how the new utility layer should preserve current error semantics and avoid widening public API surface.
5. Write the design artifact for the shared helper layer.

### Deliverables
- Shared internal utility API design
- Proposed file/layout plan
- Error-semantics preservation notes

### Completion Criteria
- The shared helper layer is designed concretely before coding
- The interface distinguishes reusable helpers from module-specific logic
- Public API exposure remains explicitly out of scope

---

## Day 4: Core Utility Implementation

**Title:** Utility Implementation  
**Theme:** Add the shared internal helper layer and its first local proof points  
**Time estimate:** 10 hours

### Tasks
1. Implement the shared internal helper header/source based on Day 3’s design.
2. Wire the first low-risk internal uses or assertions to prove the utility layer compiles and links cleanly.
3. Keep the implementation narrow: no broad public behavior changes, no opportunistic refactors outside the helper layer.
4. Add succinct internal comments where the helper semantics are not obvious from the signature alone.
5. Run the required code-quality gate for the new helper implementation.

### Deliverables
- Shared internal allocation/overflow utility layer
- First integration points in-tree
- Initial validation result

### Completion Criteria
- The new utility layer exists and builds cleanly
- The initial integration proves the layer is practical
- The required code-quality gate passes

---

## Day 5: First Core-Module Migration Batch I

**Title:** Core Batch I  
**Theme:** Replace duplicated local helpers in the first core modules  
**Time estimate:** 10 hours

### Tasks
1. Migrate the first pair of high-signal core modules from local helper logic to the shared utility layer.
2. Remove duplicated helper code only where the new utility layer is semantically equivalent.
3. Keep behavior-preserving boundaries explicit and avoid mixing in unrelated cleanup.
4. Record any residual specialized helper logic that should remain local.
5. Run the required code-quality gate and any targeted checks justified by the touched modules.

### Deliverables
- First core-module migration batch
- Reduced local helper duplication
- Residual specialized-helper notes

### Completion Criteria
- At least two planned core modules are migrated cleanly
- No behavior drift is introduced alongside the helper consolidation
- Validation passes for the touched code

---

## Day 6: First Core-Module Migration Batch II

**Title:** Core Batch II  
**Theme:** Finish the planned first-wave migrations in the named hotspot modules  
**Time estimate:** 10 hours

### Tasks
1. Migrate the remaining planned first-wave modules from the Sprint 41 prerequisite list.
2. Reconcile any mismatched local helper semantics uncovered in Day 5.
3. Document where the shared helper layer still needs later extension to cover broader `src/` patterns.
4. Keep the touched-module changes tightly scoped to helper consolidation.
5. Run the required code-quality gate and any targeted checks justified by the touched modules.

### Deliverables
- Completed first-wave core-module migration
- Shared-helper gap list for broader `src/` pass
- Validation result for the completed first-wave batch

### Completion Criteria
- The named first-wave core modules are migrated or explicitly justified as partially deferred
- Shared-helper gaps for the broader pass are explicit
- Validation passes for the completed batch

---

## Day 7: Broader `src/` Migration Audit

**Title:** Broader Audit  
**Theme:** Identify the rest of `src/` that should adopt the shared helper layer  
**Time estimate:** 10 hours

### Tasks
1. Audit the broader `src/` tree for equivalent size/overflow/allocation logic.
2. Prioritize the broader migration list using Sprint 40 hotspot and lifecycle-prep context.
3. Separate:
   - easy direct substitutions
   - moderate helper-adapter cases
   - truly local specializations that should remain file-specific
4. Record the next migration order for Days 8-9.
5. Write the broader `src/` migration audit artifact.

### Deliverables
- Broader `src/` migration inventory
- Prioritized migration order
- Specialized-helper keep list

### Completion Criteria
- The broader `src/` migration queue is explicit before coding resumes
- Easy substitutions are separated from special cases
- Later helper growth pressure is visible before the broader pass starts

---

## Day 8: Broader `src/` Migration Batch I

**Title:** Broader Batch I  
**Theme:** Start the wider source-tree migration to the shared helper layer  
**Time estimate:** 10 hours

### Tasks
1. Migrate the highest-value broader `src/` targets identified on Day 7.
2. Extend the shared helper layer only if the broader migration proves the current API is too narrow.
3. Keep the batch behavior-preserving and avoid mixing in unrelated architectural changes.
4. Record any modules that are better handled in a later sprint because the local logic is too specialized.
5. Run the required code-quality gate and any targeted follow-on checks justified by the touched surfaces.

### Deliverables
- First broader `src/` migration batch
- Shared-helper extension notes if needed
- Residual later-sprint keep/defer notes

### Completion Criteria
- The broader pass lands real consolidation outside the original hotspot list
- Any helper-layer extension is justified by live migration pressure
- Validation passes

---

## Day 9: Broader `src/` Migration Batch II

**Title:** Broader Batch II  
**Theme:** Finish the main broader source-tree helper consolidation pass  
**Time estimate:** 10 hours

### Tasks
1. Complete the remaining mainline broader `src/` substitutions chosen for Sprint 41.
2. Reconcile any new helper interface drift introduced by Day 8.
3. Confirm that the resulting shared helper layer is sufficient for the current intended coverage set.
4. Record what remains intentionally local and why.
5. Run the required code-quality gate and any targeted follow-on checks justified by the touched surfaces.

### Deliverables
- Main broader `src/` consolidation pass
- Final Sprint 41 local-specialization keep list
- Validation result for the broader pass

### Completion Criteria
- The main Sprint 41 source-tree consolidation target is complete
- Remaining local helper logic is intentional rather than accidental
- Validation passes

---

## Day 10: Auxiliary-Surface Alignment Audit

**Title:** Auxiliary Audit  
**Theme:** Audit examples, benchmarks, and scripts for the same safety patterns under the internal-first rule  
**Time estimate:** 8 hours

### Tasks
1. Audit the highest-signal examples, benchmarks, and scripts for duplicated size/overflow/allocation safety patterns.
2. Rank the auxiliary surfaces by:
   - easy alignment now
   - defer until later public-facing work
   - keep local because semantics are too specialized
3. Apply the Sprint 40 migration-risk boundary so public-risk-ranked surfaces are only touched when the benefit is clear.
4. Write the auxiliary-surface alignment audit.
5. Pick the bounded Day 11 implementation batch.

### Deliverables
- Auxiliary-surface safety-pattern audit
- Ranked auxiliary migration list
- Bounded Day 11 alignment batch plan

### Completion Criteria
- Auxiliary-surface alignment is scoped by risk and value, not by blanket cleanup
- Public-risk-ranked surfaces are distinguished from internal-first surfaces
- The Day 11 batch is explicitly bounded

---

## Day 11: Auxiliary-Surface Alignment Batch

**Title:** Auxiliary Batch  
**Theme:** Land the highest-value auxiliary safety alignments without drifting into broad public churn  
**Time estimate:** 8 hours

### Tasks
1. Apply the bounded Day 10 alignment batch to the highest-value auxiliary surfaces.
2. Keep examples/benchmarks/scripts aligned with the shared helper conventions only where the semantics are truly parallel.
3. Avoid expanding the batch into broad usability or CLI modernization work that belongs to later sprints.
4. Run the required code-quality gate if any `*.c` / `*.h` surfaces changed, plus targeted checks appropriate to the touched auxiliary surfaces.
5. Record residual auxiliary alignment work for later sprints.

### Deliverables
- First auxiliary-surface alignment batch
- Residual auxiliary follow-on list
- Validation result

### Completion Criteria
- High-value auxiliary alignment lands without opening a larger public-facing cleanup batch
- Later-sprint auxiliary work is explicitly bounded
- Validation passes for the touched surfaces

---

## Day 12: Safety-Style Documentation & Prep Rules

**Title:** Prep Rules  
**Theme:** Document the shared-helper usage expectations and bounded internal-first prep rules  
**Time estimate:** 12 hours

### Tasks
1. Write the internal guidance for when new code must use the shared helper layer rather than ad hoc local arithmetic.
2. Capture the bounded internal-first prep rules Sprint 40 handed forward and Sprint 41 exercised:
   - preserve public semantics
   - keep early work internal-first
   - use the validation anchor
   - avoid premature public migration work
3. Record the main specialization exceptions where local helpers may still be acceptable.
4. Link the new guidance back to the Sprint 40 architecture contract rather than duplicating it loosely.
5. Write the safety-style / prep-rule artifact and update working notes.

### Deliverables
- Shared-helper usage guidance
- Internal-first prep-rule note
- Local-specialization exception guidance

### Completion Criteria
- Future Epic 4 implementation work has a stable helper-usage rule
- The Sprint 40 internal-first posture is documented as an execution rule, not just historical context
- Exception cases are explicit rather than implicit

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Reconfirm the Sprint 41 end-state against the Epic 4 validation anchor  
**Time estimate:** 12 hours

### Tasks
1. Run the full required code-quality gate for the Sprint 41 code changes:
   - `make format`
   - `make lint`
   - `make test`
2. Run the strongest local reviewed baseline appropriate for the breadth of the sprint:
   - `make quality-review-full`
3. Run targeted follow-on checks justified by the touched surfaces:
   - serial dead-code path if support surfaces changed
   - tooling/examples/benchmark checks if those surfaces changed
4. Record all measured results and any reconciliation needed.
5. Write the full-validation artifact and update working notes.

### Deliverables
- Full Sprint 41 validation sweep
- Measured command/timing results
- Reconciliation notes if any issues were corrected

### Completion Criteria
- Sprint 41 closes from a measured validated state
- The validation evidence matches the Sprint 40 anchor
- Any operational caveats are documented explicitly

---

## Day 14: Sprint 41 Consolidation & Handoff

**Title:** Consolidation  
**Theme:** Synthesize the sprint into a stable helper-consolidation handoff for Sprint 42+  
**Time estimate:** 10 hours

### Tasks
1. Synthesize the helper-pattern inventory, shared utility design, migration batches, auxiliary alignment, prep rules, and validation results into one coherent Sprint 41 closeout package.
2. Reconcile any contradictions between the helper-usage guidance, internal-first posture, and touched auxiliary surfaces.
3. Record the exact Sprint 42 prerequisites inherited from Sprint 41.
4. Confirm the sprint’s day-by-day outputs satisfy every item promised in the Sprint 41 section of `PROJECT_PLAN.md`.
5. Write the final Sprint 41 summary artifact and update working notes.

### Deliverables
- Sprint 41 closeout summary
- Final Sprint 41 working-notes synthesis
- Explicit Sprint 42 handoff prerequisites

### Completion Criteria
- Sprint 41 ends with one coherent helper-consolidation handoff, not disconnected migration notes
- Sprint 42 prerequisites are explicit and consistent with the Sprint 40 architecture contract
- The full Sprint 41 scope from `PROJECT_PLAN.md` is covered within the documented 136-hour budget
