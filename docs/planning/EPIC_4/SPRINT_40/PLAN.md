# Sprint 40 Plan: Baseline, Lifecycle Inventory & Epic 4 Architecture Contract

**Sprint Duration:** 14 days  
**Goal:** Establish the measured Epic 4 baseline, convert the review/todo findings into a concrete architectural contract, and define the future ownership model for matrix state, factor state, and quality-policy surfaces before implementation-heavy refactors begin. This sprint implements the Sprint 40 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`.

**Starting Point:** Epic 3 closed with a validated quality baseline: `make quality-review-full` is the strongest local reviewed command, reviewed CMake parity is stable, the dead-code compile-db gap is closed, and the cross-platform contract is documented honestly. Epic 4 begins from that stable state and shifts from cleanup toward structural review and architecture definition.

**End State:** Sprint 40 leaves behind a measured Epic 4 baseline, a complete lifecycle/state inventory for `SparseMatrix` and factor-related APIs, a written future handle-model design, a quality-contract ownership map, a public migration-risk inventory, and a stable validation anchor for the rest of Epic 4.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 124 hours, matching the Sprint 40 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 40 Scope Audit & Baseline Setup

**Title:** Baseline Setup  
**Theme:** Convert the Sprint 40 project-plan items into a concrete measured-work inventory  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 40 section of `docs/planning/EPIC_4/PROJECT_PLAN.md`, the Epic 4 review, and the Epic 4 remediation plan so the sprint starts from the documented scope rather than inferred follow-on work.
2. Confirm the inherited Epic 3 baseline that Sprint 40 must preserve: reviewed local baseline, reviewed CMake parity, dead-code reporting contract, and cross-platform contract wording.
3. Define the Sprint 40 workstreams explicitly:
   - baseline capture
   - lifecycle inventory
   - state-model taxonomy
   - future handle-model design
   - quality-contract ownership map
   - public migration-risk audit
   - validation anchor
4. Open Sprint 40 working notes and record the scope, assumptions, and audit order.
5. Identify which repo surfaces will be treated as authoritative inputs during the sprint.

### Deliverables
- Sprint 40 scope inventory
- Initial workstream map
- Working-notes baseline assumptions

### Completion Criteria
- Sprint 40 starts from the documented Epic 4 plan and review artifacts
- The sprint’s architecture/audit surfaces are explicit before deeper inventory work begins
- The inherited Epic 3 baseline is named as a preserve-not-reopen contract

---

## Day 2: Reviewed-Quality Baseline Capture

**Title:** Quality Baseline  
**Theme:** Record the current maintained local reviewed baseline as the reference point for Epic 4  
**Time estimate:** 8 hours

### Tasks
1. Capture the current direct and wrapper quality-command surfaces, including `format`, `lint`, `test`, `quality-review`, `quality-review-cmake`, and `quality-review-full`.
2. Record the authoritative local baseline semantics for reviewed Makefile vs reviewed CMake paths.
3. Inventory current maintained operator-facing quality docs in `README.md`, `Makefile`, and workflow notes.
4. Note any known operational caveats that matter as baseline context, including the still-serialized dead-code path.
5. Write the baseline note for the maintained quality-command surface.

### Deliverables
- Reviewed-quality baseline note
- Local quality-command inventory
- Initial caveat list relevant to Epic 4 refactors

### Completion Criteria
- The strongest local reviewed baseline is documented concretely
- Maintained quality-command ownership is visible before later ownership mapping
- Known operational caveats are recorded as baseline context, not rediscovered later

---

## Day 3: Reviewed CMake, Dead-Code & Platform Baseline Capture

**Title:** Support Baseline  
**Theme:** Capture the remaining validated baseline surfaces beyond the direct local wrappers  
**Time estimate:** 8 hours

### Tasks
1. Record the current reviewed CMake parity baseline, including the active CTest registration count and its relationship to `quality-review-cmake`.
2. Record the current dead-code report/check contract, including what is enforced, what is informational, and what remains serialized by design.
3. Summarize the current Linux/macOS/Windows enforced/staged/excluded baseline relevant to later quality-contract ownership decisions.
4. Capture the current benchmark/example/tooling support surfaces that matter as “do not regress” context during Epic 4.
5. Write the combined support-baseline note.

### Deliverables
- Reviewed CMake parity baseline note
- Dead-code baseline note
- Cross-platform/support-surface baseline summary

### Completion Criteria
- The full Epic 4 starting baseline is documented beyond just local reviewed commands
- The dead-code and platform contracts are captured as architecture inputs
- The CMake/test-truthfulness baseline is explicit before lifecycle auditing starts

---

## Day 4: Hotspot, Allocation-Density & Large-File Inventory

**Title:** Hotspot Inventory  
**Theme:** Measure the structural hotspots the later Epic 4 refactors will target  
**Time estimate:** 10 hours

### Tasks
1. Inventory the largest and most structurally concentrated source, test, benchmark, and script files.
2. Record the highest-signal allocation-heavy and helper-duplication surfaces in `src/`, `tests/`, `benchmarks/`, and `examples/`.
3. Identify the first-tier architectural hotspots already called out in the review, such as matrix lifecycle surfaces and `src/sparse_graph.c`.
4. Separate large files that are strong feature-owner files from those that are true maintainability hotspots.
5. Write the hotspot/allocation-density baseline artifact for comparison in later sprints.

### Deliverables
- Hotspot-file inventory
- Allocation-density/support-surface inventory
- First-tier structural-refactor target list

### Completion Criteria
- Structural hotspots are measured rather than described generically
- The later Epic 4 refactor queue is grounded in actual file/surface concentration
- Large-file ownership is distinguished from true maintainability debt

---

## Day 5: Matrix/Factor Lifecycle Inventory I

**Title:** Lifecycle Inventory I  
**Theme:** Audit the first half of the stateful matrix/factor API surface  
**Time estimate:** 8 hours

### Tasks
1. Audit the public and internal LU, Cholesky, and LDLT surfaces for:
   - in-place mutation
   - original-matrix assumptions
   - permutation assumptions
   - `factored`-state dependencies
   - copy-before-reuse requirements
2. Record where current headers, README, tutorial prose, and implementation behavior carry the lifecycle contract.
3. Identify the first cluster of APIs that are already close to explicit factor-handle boundaries versus those still overloaded onto `SparseMatrix`.
4. Capture initial risk notes around cancellation and partially-mutated state.
5. Write the first lifecycle-inventory artifact.

### Deliverables
- Lifecycle inventory for LU/Cholesky/LDLT surfaces
- Initial mutation/precondition map
- Cancellation and partial-mutation risk notes

### Completion Criteria
- The first major stateful API cluster is inventoried concretely
- Mutation/precondition contracts are documented from actual code/docs, not memory
- The handle-model design work has grounded inputs from real lifecycle surfaces

---

## Day 6: Matrix/Factor Lifecycle Inventory II

**Title:** Lifecycle Inventory II  
**Theme:** Audit the remaining stateful matrix/factor and analysis-related API surface  
**Time estimate:** 10 hours

### Tasks
1. Audit QR, SVD, analysis/reordering, iterative, eigensolver, and related matrix-consuming APIs for the same lifecycle questions as Day 5.
2. Classify where APIs require:
   - original/unfactored matrix view
   - identity permutations
   - separate factor object use
   - implicit or explicit mutable cached state
3. Identify the places where current documentation compensates for a hard-to-read lifecycle model.
4. Record the strongest cross-subsystem lifecycle inconsistencies.
5. Write the second lifecycle-inventory artifact and merge it conceptually with Day 5.

### Deliverables
- Lifecycle inventory for QR/SVD/analysis/iterative/eigs surfaces
- Cross-subsystem lifecycle inconsistency list
- Full raw lifecycle input set for taxonomy work

### Completion Criteria
- The full major stateful API surface is inventoried
- Cross-subsystem lifecycle mismatches are explicit
- Later taxonomy and handle-model work can proceed from a complete input set

---

## Day 7: State-Model Taxonomy

**Title:** Taxonomy  
**Theme:** Classify the inventoried APIs into a stable lifecycle/state model  
**Time estimate:** 8 hours

### Tasks
1. Convert the raw lifecycle inventory into explicit API classes:
   - original-matrix consumers
   - analysis/factor builders
   - factor consumers
   - read-only query paths
   - mutation-heavy transitional paths
2. Group related APIs by lifecycle role rather than by file alone.
3. Identify the current transitional/overloaded classes that most strongly justify explicit handles later.
4. Note where taxonomy boundaries are clean already and where they remain mixed.
5. Write the taxonomy artifact that will anchor the handle-model design.

### Deliverables
- State-model taxonomy
- Grouped API-role map
- Mixed-boundary hotspot list

### Completion Criteria
- The lifecycle inventory is reduced to a stable classification model
- APIs are grouped by role in a way later refactors can target directly
- Overloaded/mixed boundaries are explicit before design work begins

---

## Day 8: Lifecycle Precondition, Mutation & Cancellation Contract Map

**Title:** Contract Map  
**Theme:** Turn the taxonomy into an explicit contract map for callers and future internals  
**Time estimate:** 10 hours

### Tasks
1. Map the key preconditions across the taxonomy:
   - original state required
   - identity permutations required
   - factored state required
   - caller must copy before reuse
2. Map the mutation boundaries:
   - fully in-place
   - partially mutating
   - cache-mutating
   - read-only
3. Isolate the cancellation-sensitive paths and record what they can mutate before first callback observation.
4. Distinguish behavior that should remain public contract from behavior that should become internal-only under future handles.
5. Write the contract-map artifact for the design phase.

### Deliverables
- Lifecycle contract map
- Mutation-boundary map
- Cancellation-sensitive-path inventory

### Completion Criteria
- The key caller-visible lifecycle contracts are explicit and cross-checked
- Mutation semantics are mapped in a way that supports later design decisions
- Cancellation-sensitive behavior is recorded as architecture input, not just a doc caveat

---

## Day 9: Future Handle-Model Design I

**Title:** Handle Design I  
**Theme:** Design the target explicit lifecycle model from the audited contract map  
**Time estimate:** 8 hours

### Tasks
1. Define the target high-level object model for:
   - analysis handles
   - factor handles
   - solve/operation context boundaries
2. Decide which current `SparseMatrix` responsibilities should remain on the matrix object and which should migrate behind explicit handles.
3. Sketch the safest staged migration shape that preserves current public behavior while reducing hidden state internally.
4. Identify the minimal compatibility scaffolding later sprints will need.
5. Write the first handle-model design note.

### Deliverables
- High-level future handle-model design
- Responsibility split between matrix objects and future handles
- Initial compatibility-scaffolding concept

### Completion Criteria
- The design moves beyond “explicit handles would be better” into a concrete object model
- Current overloaded matrix responsibilities are explicitly split into keep/move buckets
- Later lifecycle-refactor sprints have a written starting architecture

---

## Day 10: Future Handle-Model Design II & Migration Strategy

**Title:** Handle Design II  
**Theme:** Complete the migration contract from current hidden state to future explicit handles  
**Time estimate:** 10 hours

### Tasks
1. Refine the Day 9 design into a staged migration strategy with internal-first and public-compatibility phases.
2. Identify where compatibility wrappers, deprecation layers, or documentation shims will likely be required.
3. Define the earliest safe insertion points for internal handles before public API churn.
4. Record the highest-risk migration edges:
   - factorization entry points
   - analysis/reorder surfaces
   - cancellation semantics
   - copy-before-reuse pitfalls
5. Write the completed future handle-model and migration-strategy artifact.

### Deliverables
- Completed future handle-model design
- Staged migration strategy
- Compatibility/deprecation risk inventory

### Completion Criteria
- The handle-model design includes a realistic migration path, not just an end state
- High-risk transition edges are explicit
- Sprint 42 and later lifecycle refactors have a concrete architecture contract to implement

---

## Day 11: Quality-Contract Ownership Audit

**Title:** Ownership Audit  
**Theme:** Define where quality, policy, and workflow truth should live after Epic 4  
**Time estimate:** 8 hours

### Tasks
1. Audit current command, behavior, policy, and contract duplication across:
   - `Makefile`
   - scripts
   - CI workflows
   - `README.md`
   - headers/tutorial docs
2. Separate operator entry points from machine behavior and from policy text.
3. Define the target ownership model for each truth surface:
   - commands
   - enforcement behavior
   - CI matrix truth
   - maintainer policy
   - user-facing summaries
4. Identify the strongest current duplication hotspots that later sprints should reduce.
5. Write the quality-contract ownership-map artifact.

### Deliverables
- Quality-contract ownership map
- Duplication hotspot inventory
- Target ownership model for commands/behavior/policy/docs

### Completion Criteria
- Quality-contract truth is mapped to stable homes before later simplification work
- Duplication hotspots are named concretely
- Later Epic 4 doc/tooling cleanup has an explicit ownership target

---

## Day 12: Public Migration-Risk Audit

**Title:** Migration Risk  
**Theme:** Identify the public-facing surfaces most likely to need compatibility help during Epic 4  
**Time estimate:** 8 hours

### Tasks
1. Audit the public API, docs, examples, and tooling surfaces most sensitive to lifecycle and workspace changes.
2. Identify where later explicit-handle or workspace work is most likely to require:
   - compatibility wrappers
   - doc rewrites
   - examples updates
   - migration notes
3. Separate low-risk internal-only refactor zones from public-facing migration-risk zones.
4. Rank the migration-risk surfaces by likely review burden and compatibility sensitivity.
5. Write the migration-risk artifact.

### Deliverables
- Public migration-risk inventory
- Internal-only vs public-facing refactor boundary map
- Prioritized compatibility-sensitive surface list

### Completion Criteria
- Later Epic 4 implementation sprints know where compatibility risk is concentrated
- Public-facing migration debt is identified before code churn begins
- Internal-only opportunities are separated from externally visible change zones

---

## Day 13: Validation Anchor & Refactor-Sprint Command Matrix

**Title:** Validation Anchor  
**Theme:** Define the authoritative validation matrix for the implementation-heavy Epic 4 sprints  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the exact validation commands that must anchor later refactor work:
   - full C/C header gate
   - reviewed wrapper paths
   - reviewed CMake parity
   - serial dead-code path
   - targeted benchmark/example reruns where applicable
2. Distinguish which validation paths are mandatory for all `*.c` / `*.h` changes and which are targeted follow-ons.
3. Record how later sprints should preserve truthfulness around CTest counts, dead-code serialization, and cross-platform wording.
4. Build a compact validation matrix by change type.
5. Write the validation-anchor artifact.

### Deliverables
- Epic 4 validation-anchor note
- Change-type vs validation-command matrix
- Preserved-truthfulness checklist for later refactor sprints

### Completion Criteria
- Later Epic 4 implementation work has a stable validation reference
- Mandatory vs targeted validation is explicit
- Known truthfulness constraints are embedded in the validation anchor

---

## Day 14: Sprint 40 Architecture Contract Synthesis

**Title:** Contract Synthesis  
**Theme:** Consolidate the sprint outputs into one stable architectural starting point for Sprint 41+  
**Time estimate:** 10 hours

### Tasks
1. Synthesize the measured baseline, lifecycle inventory, taxonomy, handle-model design, ownership map, migration-risk audit, and validation anchor into a coherent Sprint 40 closeout package.
2. Reconcile any contradictions between the lifecycle contract, ownership model, and migration strategy.
3. Write the Sprint 40 summary artifact and update working notes with the final decisions and preserved constraints.
4. Name the exact prerequisites Sprint 41 and Sprint 42 should inherit.
5. Confirm the sprint’s day-by-day outputs together satisfy every item promised in the Sprint 40 section of `PROJECT_PLAN.md`.

### Deliverables
- Sprint 40 architecture-contract summary
- Final Sprint 40 working-notes synthesis
- Explicit handoff prerequisites for Sprint 41 and Sprint 42

### Completion Criteria
- Sprint 40 ends with one coherent architectural starting point, not a pile of disconnected audits
- Sprint 41+ prerequisites are explicit and internally consistent
- The full Sprint 40 scope from `PROJECT_PLAN.md` is covered within the documented 124-hour budget
