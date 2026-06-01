# Sprint 50 Plan: Direct-Solver Lifecycle Baseline & API Design

**Sprint Duration:** 14 days  
**Goal:** Freeze the post-Epic-4 direct-solver lifecycle baseline, inventory
the live one-shot and analysis/refactor caller surfaces, and define the bounded
public direct-solver lifecycle model Epic 5 will implement. This sprint
implements the Sprint 50 section of `docs/planning/EPIC_5/PROJECT_PLAN.md`.

**Starting Point:** Epic 4 closed with the iterative/eigensolver repeated-run
story publicly exposed in bounded form, the maintainer-guide / README ownership
split in place, and the reviewed validation baseline preserved. The remaining
high-value gap is now concentrated on the direct-solver side: mutable
`SparseMatrix` state remains the main compatibility-facing tradeoff, while
`sparse_analysis_t` / `sparse_factors_t` already provide a partial lifecycle
precedent that is not yet the dominant caller model.

**End State:** Sprint 50 leaves behind a documented and bounded Epic 5
direct-solver lifecycle target, a refreshed baseline and surface inventory, an
explicit non-goal / compatibility fence, a validation and landing plan for the
later public API batches, and the Day 14 handoff artifacts needed to start
Sprint 51 from a shared design rather than inferred intent.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
132 hours, matching the Sprint 50 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 50 Scope Audit & Baseline Refresh

**Title:** Baseline Setup  
**Theme:** Convert the Sprint 50 project-plan items into a bounded
direct-solver lifecycle design and baseline-refresh execution map  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 50 section of `docs/planning/EPIC_5/PROJECT_PLAN.md`,
   the Epic 5 review and remediation todo, and the key Epic 4 closeout docs
   that define the inherited baseline.
2. Reconfirm the preserved constraints Sprint 50 must not violate:
   - preserve the current `make quality-review-full` truthfulness baseline
   - avoid broad solver API redesign in the design sprint
   - treat one-shot direct-solver APIs as compatibility surfaces, not
     immediate removal targets
   - preserve Epic 4’s README / maintainer-guide ownership split
3. Define the Sprint 50 workstreams explicitly:
   - baseline recheck
   - direct-solver surface inventory
   - public lifecycle design
   - non-goal and compatibility fence
   - validation / landing design
   - closeout and handoff
4. Record the highest-risk seams:
   - hidden mutable `SparseMatrix` state
   - uneven direct-solver repeated-workflow guidance
   - partial analysis/refactor public lifecycle precedent
   - documentation drift between README, headers, examples, and benchmarks
5. Open Sprint 50 working notes and record scope, assumptions, and initial
   landing order.

### Deliverables
- Sprint 50 scope inventory
- Baseline-refresh notes
- Working-notes starting assumptions

### Completion Criteria
- Sprint 50 starts from the documented Epic 5 baseline rather than ad hoc API
  brainstorming
- Preserved validation and compatibility constraints are explicit before deeper
  audit work begins
- The direct-solver lifecycle workstreams are named before artifacts are written

---

## Day 2: Validation Baseline and Truthfulness Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed local baseline and the truthfulness anchors
that later lifecycle work must preserve  
**Time estimate:** 8 hours

### Tasks
1. Reconfirm the maintained reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity
   - current test-count truthfulness anchors
2. Refresh the Day 1 baseline notes around the quality contract relevant to
   Sprint 50:
   - command authority
   - reviewed baseline meaning
   - what later public API sprints will need to rerun
3. Inspect the current touched direct-solver examples and benchmarks to see
   which follow-on binaries are likely to matter once public direct lifecycle
   work begins.
4. Record the smallest authoritative validation set Sprint 50 needs as a
   design sprint versus what later implementation sprints must rerun after
   `*.c` / `*.h` edits.
5. Write the baseline-validation artifact notes.

### Deliverables
- Refreshed validation/truthfulness notes
- Sprint 50 design-sprint validation boundary
- Candidate touched-surface follow-on list for later sprints

### Completion Criteria
- The design sprint uses the same validation truthfulness language as the live
  repo
- Later implementation validation needs are explicit before API design begins
- No baseline ambiguity remains about the maintained reviewed close state

---

## Day 3: Direct-Solver Public Surface Inventory

**Title:** Surface Inventory  
**Theme:** Re-map the one-shot, analysis/factor/refactor, solve, example, and
benchmark surfaces before choosing the lifecycle landing boundary  
**Time estimate:** 8 hours

### Tasks
1. Inventory the live direct-solver public API surface across:
   - LU
   - Cholesky
   - LDL^T
   - QR where it informs lifecycle/state expectations
   - `sparse_analysis_t`
   - `sparse_factors_t`
2. Classify the current public direct-solver workflows into bounded buckets:
   - one-shot in-place factor-and-solve
   - analysis/factor/refactor
   - analysis-aware CSC and factor-many side paths
   - examples and benchmark caller models
3. Identify where callers still rely on hidden state or documentation-only
   discipline rather than explicit lifecycle affordances.
4. Separate true first landing targets from later compatibility, benchmark, and
   documentation verification surfaces.
5. Write the direct-solver public-surface inventory artifact.

### Deliverables
- Refreshed direct-solver surface inventory
- One-shot vs analysis/refactor classification
- First landing-target notes

### Completion Criteria
- The direct-solver lifecycle problem is reduced to named public seams rather
  than a generic “state model” complaint
- True landing targets are distinguished from later validation surfaces
- The inventory is grounded in the current public headers and caller docs

---

## Day 4: Internal Lifecycle and Structural Precedent Inventory

**Title:** Precedent Audit  
**Theme:** Map the internal and existing public lifecycle precedents that
Sprint 50 can reuse instead of inventing a new model from scratch  
**Time estimate:** 8 hours

### Tasks
1. Audit the existing public lifecycle precedent in `include/sparse_analysis.h`
   and the factor/refactor implementation seams behind it.
2. Refresh the relevant Epic 4 repeated-run handle precedents for iterative and
   eigensolver code, focusing on terminology and lifecycle shape rather than
   solver-specific behavior.
3. Inventory the internal direct-solver structural seams that may matter to
   later public lifecycle exposure:
   - analysis-aware paths
   - CSC/native dispatch points
   - factor containers and solve wrappers
4. Classify what Sprint 50 should borrow from those precedents versus what must
   remain direct-solver-specific.
5. Write the lifecycle-precedent inventory artifact.

### Deliverables
- Lifecycle precedent inventory
- Public-vs-internal precedent mapping
- Reuse notes for Sprint 50 API design

### Completion Criteria
- Sprint 50 has a concrete precedent set before public API design starts
- The direct-solver lifecycle design is positioned as an extension of existing
  repo patterns rather than a fresh model
- Direct-solver-specific constraints are separated from generic lifecycle
  precedent

---

## Day 5: Direct-Solver Lifecycle Gap Analysis

**Title:** Gap Analysis  
**Theme:** Turn the public-surface and precedent inventories into an explicit
gap map for usability, correctness, efficiency, and maintainability  
**Time estimate:** 12 hours

### Tasks
1. Compare the current one-shot and analysis/refactor caller stories against the
   target qualities Epic 5 wants:
   - explicit lifecycle
   - reduced hidden mutable state
   - stronger factor-many guidance
   - compatibility preservation
2. Identify the highest-value lifecycle gaps:
   - where repeated direct workflows are still implicit
   - where factor/refactor reuse is real but underexposed
   - where docs and examples still over-center the one-shot path
3. Identify the strongest design constraints coming from compatibility and
   mutable-state reality.
4. Record the smallest credible “public direct lifecycle” exposure that would
   materially improve the system without reopening broad API churn.
5. Write the lifecycle gap-analysis artifact.

### Deliverables
- Direct-solver lifecycle gap analysis
- Ranked high-value gap list
- Minimum-credible public lifecycle exposure notes

### Completion Criteria
- The problem statement is narrowed from “direct solver state is awkward” to a
  small set of named lifecycle gaps
- Compatibility constraints are explicit before API design begins
- The later design work has a bounded target instead of an aspirational one

---

## Day 6: Public Lifecycle API Design Batch I

**Title:** API Design I  
**Theme:** Define the first half of the bounded public direct-solver lifecycle
model and its terminology  
**Time estimate:** 12 hours

### Tasks
1. Decide the intended public lifecycle abstraction shape:
   - extension of `sparse_analysis_t` / `sparse_factors_t`
   - additional opaque handles
   - or a bounded hybrid where needed
2. Define the target public lifecycle stages:
   - initialize / zero
   - analyze / prepare
   - factor / refactor
   - solve / reuse
   - free
3. Decide which major direct-solver families the first public lifecycle model
   must cover explicitly and which can remain compatibility-wrapped in early
   implementation sprints.
4. Define the terminology, naming, and relationship to the existing public
   analysis/factors model.
5. Draft the first half of the API design artifact.

### Deliverables
- Public direct-solver lifecycle abstraction draft
- Lifecycle-stage and naming decisions
- Coverage boundary notes for the first public model

### Completion Criteria
- Sprint 50 has a concrete first-pass lifecycle model rather than only a gap
  list
- Public naming and lifecycle stages are explicit before contract details are
  finalized
- The design remains bounded around the strongest direct-solver workflows

---

## Day 7: Post-Design Audit and Remaining Questions

**Title:** Design Audit  
**Theme:** Audit the first design pass to narrow the remaining contract,
compatibility, and non-goal decisions  
**Time estimate:** 8 hours

### Tasks
1. Review the Day 6 design against the Day 5 gap analysis and the inherited
   Epic 4 compatibility boundary.
2. Identify unresolved questions:
   - what should stay one-shot-first
   - what should stay internal-only
   - where examples and benchmarks should lag or adopt the lifecycle model
3. Separate “must decide in Sprint 50” questions from “implementation detail
   for Sprint 51+” questions.
4. Confirm the bounded target set for the final API contract design day.
5. Write the post-design audit artifact.

### Deliverables
- Post-design audit
- Remaining open-question list
- Bounded target set for contract finalization

### Completion Criteria
- The remaining Sprint 50 design queue is concrete rather than generic
- Implementation-detail questions are separated from true public-contract
  decisions
- The sprint stays bounded away from solving Sprint 51 in advance

---

## Day 8: Public Lifecycle API Design Batch II

**Title:** API Design II  
**Theme:** Finalize the public contract details, compatibility relationship,
and caller-facing lifecycle expectations  
**Time estimate:** 8 hours

### Tasks
1. Finalize the public contract for:
   - initialization and zero-state expectations
   - prepare/analyze semantics
   - solve and refactor expectations
   - reuse and reset behavior
   - teardown/free semantics
2. Define what lifecycle state reuse means and what it explicitly does not
   mean.
3. Decide how result structs, option structs, and designated-initializer norms
   should appear in the final caller story.
4. Record how existing one-shot APIs relate to the lifecycle model:
   - wrapper
   - peer entry point
   - simple/default caller path
5. Complete the public lifecycle API design artifact.

### Deliverables
- Final public direct-solver lifecycle API design
- Caller-facing lifecycle contract
- One-shot compatibility relationship notes

### Completion Criteria
- The design artifact is complete enough to drive Sprint 51 implementation
- Reuse, reset, and wrapper semantics are explicit rather than implied
- The final caller story is documented at a high level

---

## Day 9: Non-Goal and Compatibility Fence

**Title:** Scope Fence  
**Theme:** Record the explicit non-goals and compatibility boundaries that keep
Epic 5 from expanding into a broad solver API rewrite  
**Time estimate:** 8 hours

### Tasks
1. Enumerate the direct-solver lifecycle changes Sprint 50-52 are allowed to
   make.
2. Enumerate the explicit non-goals:
   - broad public factor-container redesign everywhere at once
   - removal of one-shot APIs
   - raw internal storage exposure
   - unrelated solver-family expansion
   - broad benchmark-framework redesign
3. Document the accepted compatibility tradeoffs that remain for Epic 5.
4. Record the boundary between Sprint 50 design work and later implementation
   and documentation adoption work.
5. Write the non-goal / compatibility fence artifact.

### Deliverables
- Explicit non-goal list
- Compatibility fence notes
- Sprint 50-to-51 boundary definition

### Completion Criteria
- Epic 5 scope is bounded explicitly rather than by implication
- One-shot compatibility preservation is documented as a conscious contract
- Later sprints have a written fence against design drift

---

## Day 10: Validation and Landing Plan

**Title:** Landing Design  
**Theme:** Define the validation contract and landing order for the later
public direct-solver lifecycle implementation sprints  
**Time estimate:** 12 hours

### Tasks
1. Define the validation contract for later `*.c` / `*.h` direct-solver
   lifecycle work:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full` for substantial public API batches
2. Decide the likely targeted follow-ons for the later implementation sprints:
   - direct-solver examples
   - factor-many / refactor benchmarks
   - touched regression binaries
3. Bound the intended implementation order:
   - public headers / API surface
   - implementation and wrapper integration
   - docs/examples adoption
   - compatibility sweep
   - final validation
4. Record explicit out-of-scope items for Sprint 50 and the immediate Sprint 51
   landing.
5. Write the landing/validation design artifact.

### Deliverables
- Validation-plan artifact
- Implementation landing order
- Explicit out-of-scope notes

### Completion Criteria
- Later implementation sprints have a clear validation contract before code
  edits begin
- The direct-solver lifecycle landing order is grounded in the live seam map
- Scope boundaries are explicit before Sprint 50 closes

---

## Day 11: Documentation and Caller-Surface Audit

**Title:** Caller Audit  
**Theme:** Re-audit README, tutorial, examples, and benchmark docs from the
perspective of the newly designed direct-solver lifecycle model  
**Time estimate:** 8 hours

### Tasks
1. Identify which caller-facing docs currently over-center the one-shot path
   versus which already hint at analysis/refactor and repeated direct
   workflows.
2. Identify the smallest high-signal docs/example surfaces later sprints will
   need to update if the designed lifecycle model lands.
3. Separate:
   - docs that should stay one-shot-first
   - docs that should explain the advanced lifecycle path
   - docs that should only cross-reference
4. Record any naming or terminology drift between the design artifact and the
   current public docs.
5. Write the caller-surface audit artifact.

### Deliverables
- Caller-surface audit
- Later docs/example adoption target list
- Terminology alignment notes

### Completion Criteria
- Later docs/example work is bounded before implementation starts
- The designed lifecycle model is checked against the live caller-facing docs
- No hidden documentation contradiction remains unrecorded

---

## Day 12: Sprint 50 Summary and Handoff Draft

**Title:** Summary Draft  
**Theme:** Assemble the Sprint 50 design outputs into a coherent handoff set
for Sprint 51  
**Time estimate:** 8 hours

### Tasks
1. Consolidate the baseline, inventory, gap-analysis, design, non-goal, and
   validation outputs.
2. Extract the strongest explicit Day 14 handoff points:
   - the public lifecycle target
   - preserved compatibility rules
   - first implementation order
   - major non-goals
3. Record any residual questions intentionally deferred to Sprint 51.
4. Draft the Sprint 50 summary / handoff artifact structure.
5. Update working notes with the pre-closeout synthesis.

### Deliverables
- Sprint 50 summary-draft notes
- Sprint 51 handoff outline
- Consolidated pre-closeout working notes

### Completion Criteria
- Sprint 50 outputs are coherent as one package rather than scattered notes
- The next sprint has a clear implementation starting point
- Remaining open questions are explicitly bounded

---

## Day 13: Sprint 50 Documentation and Sanity Sweep

**Title:** Sanity Sweep  
**Theme:** Run the final design-sprint sanity checks and make sure the Sprint 50
artifacts are internally consistent  
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 50 artifacts for internal consistency:
   - lifecycle terminology
   - compatibility wording
   - non-goal language
   - validation wording
2. Reconfirm that the daily time budget and total sprint estimate stay aligned
   to the project plan.
3. Reconfirm that no Sprint 50 artifact accidentally commits to Sprint 51-52
   implementation details as settled fact where only a design fence exists.
4. Run targeted repo sanity checks for the touched planning/docs references.
5. Record the Day 13 validation/sanity-sweep artifact.

### Deliverables
- Design-sprint sanity-sweep notes
- Internal-consistency corrections if needed
- Final pre-closeout artifact check

### Completion Criteria
- Sprint 50 artifacts are internally consistent and implementation-ready
- The project-plan budget and daily plan still match
- No accidental overcommitment to later implementation details remains

---

## Day 14: Sprint 50 Closeout and Handoff

**Title:** Closeout  
**Theme:** Close Sprint 50 with a bounded direct-solver lifecycle design and a
clean implementation handoff into Sprint 51  
**Time estimate:** 12 hours

### Tasks
1. Write the Sprint 50 closeout and handoff artifact.
2. Finalize working notes with the direct-solver lifecycle design summary and
   the explicit Sprint 51 implementation boundary.
3. Confirm whether Sprint 50 changed any project-plan assumptions that require
   follow-on planning updates.
4. Record the final Sprint 50 state:
   - baseline preserved
   - surface inventory complete
   - design artifact complete
   - non-goal fence complete
   - validation/landing plan complete
5. Ensure the sprint closes from a docs/design-only state without claiming
   implementation or validation work that did not occur.

### Deliverables
- Sprint 50 closeout and handoff artifact
- Final working-notes synthesis
- Explicit Sprint 51 starting boundary

### Completion Criteria
- Sprint 50 closes with a coherent direct-solver lifecycle design package
- Sprint 51 can start from explicit implementation guidance rather than
  re-discovery
- The sprint ends with a truthful docs/design-only closeout state
