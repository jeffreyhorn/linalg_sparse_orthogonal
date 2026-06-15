# EPIC 7 Remediation Todo

**Date:** 2026-06-15  
**Reviewer:** Codex  
**Purpose:** Step-by-step plan to address the gaps identified in
`review-codex-2026-06-15.md`.

## Goal

Use Epic 7 to move the repository from a well-productized, engineering-grade
sparse linear algebra library toward a more state-of-the-art library surface.

The main themes are:

- reduce the structural cost of the current linked-list-first product model
- finish the advanced configuration convergence story
- deepen backend/performance architecture and performance governance
- converge packaging, release, and platform confidence further
- shrink the remaining large source/test review surfaces
- simplify public/reference documentation
- and start capability expansion on the biggest library-ceiling constraints

## Steps

### 1. Freeze the post-Epic-6 baseline and define the Epic 7 state-of-the-art target

Work:

1. Reconfirm the post-Epic-6 validation and reviewed-baseline anchors.
2. Re-read the Epic 6 retrospective and residual queue.
3. Decide which remaining gaps are true Epic 7 goals versus explicit non-goals:
   - core matrix/product-model convergence
   - configuration phase 2
   - backend/performance architecture phase 2
   - platform/release convergence
   - maintainability/test-surface cleanup
   - capability-surface expansion
4. Freeze the non-goal fence before implementation begins.

Exit criteria:

- one written Epic 7 architecture/productization contract
- one explicit list of real state-of-the-art goals
- one explicit non-goal and deferral fence

### 2. Re-tier the core matrix/product model

Work:

1. Audit where `SparseMatrix` remains the right owner and where it is only a
   compatibility-facing shell around compressed working formats.
2. Decide the next explicit product boundary between:
   - construction/editing
   - compressed working formats
   - factor/workspace ownership
3. Reduce unnecessary round-trips through linked-list state on the
   highest-value solver workflows.
4. Tighten caller-visible semantics so advanced workflows need less hidden
   matrix-state discipline.

Exit criteria:

- cleaner public/product relationship between mutable matrix state and numeric
  working state
- smaller conversion burden on the hottest maintained solver lanes
- clearer caller rules for when mutation, copying, and factor ownership apply

### 3. Finish the advanced-configuration modernization story

Work:

1. Inventory the remaining `SPARSE_FM_*`, debug/profile, and residual
   compatibility env-var surfaces.
2. Decide which controls become:
   - public typed options
   - internal typed policy
   - compatibility-only overrides
   - or retired/developer-only debug knobs
3. Land the next typed surfaces where justified.
4. Update precedence, docs, and regression support together.

Exit criteria:

- the remaining env-var surface is materially smaller
- the typed configuration story is coherent beyond the first ND/reorder slice
- debug/profile controls no longer leak into the product story unnecessarily

### 4. De-chronologize the public surface

Work:

1. Audit public headers, README/tutorial/examples/benchmarks/INSTALL for
   sprint-history, ABI-history, and planning-history overexposure.
2. Move durable user-facing contract language into concise product/reference
   form.
3. Move deeper rationale and chronology back into planning docs where possible.
4. Keep maintainer-policy detail in `docs/maintainer_guide.md`, not in every
   user-facing surface.

Exit criteria:

- public docs read more like product/reference docs than delivery archives
- public headers keep API-local caveats but shed avoidable history
- planning chronology lives primarily in `docs/planning/`

### 5. Deepen the backend/performance architecture

Work:

1. Audit the next performance-critical dense-kernel and compressed-format
   paths after Sprint 64’s first Cholesky lane.
2. Decide what a broader backend layer should own:
   - dense kernels
   - optional external math backends
   - runtime/threading policy
   - backend/callback parity
3. Land the next bounded backend-aware solver lanes.
4. Preserve the default self-contained build as an authoritative path.

Exit criteria:

- backend-aware acceleration is no longer confined to one narrow first lane
- the repo has a clearer long-term performance architecture
- fallback correctness and default self-contained behavior remain explicit

### 6. Strengthen benchmark/performance governance phase 2

Work:

1. Re-audit which benchmark surfaces are truly regression-sensitive.
2. Add stronger machine-readable manifests and longitudinal reporting support.
3. Decide where thresholding is justified and where artifact-only comparison is
   still the right policy.
4. Align CI/runtime reporting with that narrower, more defensible model.

Exit criteria:

- performance claims become easier to compare across branches and time
- the canonical benchmark surface supports more than local snapshot capture
- the repo gains a better answer to “what performance regressions matter?”

### 7. Converge packaging, release, and platform quality further

Work:

1. Reassess the static-first release contract and decide whether it should stay
   bounded or widen.
2. Revisit Windows reviewed-surface exclusions, macOS dead-code staging, and
   install-validation asymmetry.
3. Tighten release/install/package verification where the product contract
   still feels narrower than it needs to be.
4. Align docs, workflows, and maintainer policy with the resulting contract.

Exit criteria:

- platform/release asymmetries are either reduced or more intentionally bounded
- packaging/install/release claims read more like a mature distribution story
- reviewed platform evidence is clearer and more stable

### 8. Reduce the remaining large implementation and giant-test hotspots

Work:

1. Re-rank the remaining biggest source files by ownership pain.
2. Re-rank the remaining biggest tests by proof-maintenance pain.
3. Continue bounded extraction only where ownership is genuinely clearer after
   the split.
4. Remove stale sprint-history commentary from touched permanent code and
   permanent tests while keeping useful algorithm explanations.

Exit criteria:

- remaining top hotspots are smaller or more clearly partitioned
- proof files are easier to review by behavior family
- permanent code and tests contain less sprint-local archaeology

### 9. Expand assurance where the current reviewed/test surface is still weaker than the product claim

Work:

1. Add stronger external-oracle or differential proof where practical.
2. Improve the highest-value platform coverage gaps.
3. Strengthen proof around hard lifecycle, CSC, and backend-aware paths.
4. Reduce dependence on giant chronology-style tests for confidence.

Exit criteria:

- the hardest maintained solver paths have stronger second-layer assurance
- reviewed platform evidence is stronger on the highest-value exclusions
- proof becomes easier to trust without depending on giant monolithic files

### 10. Productize advanced workflow adoption

Work:

1. Add small runnable examples for advanced repeated-run workflows where they
   are part of the supported product story.
2. Improve discoverability of:
   - repeated iterative handles
   - repeated eigensolver handles
   - advanced typed configuration
   - compressed/backend-aware direct paths where caller-relevant
3. Tighten callback-parity gaps where the public contract still reads
   inconsistently.

Exit criteria:

- advanced supported workflows are discoverable without reading tests or giant
  headers
- example surfaces are better aligned with the final supported product model
- callback/cancellation behavior is more uniform where it matters

### 11. Start capability-surface expansion on the most limiting ceilings

Work:

1. Define the bounded Epic 7 capability target for:
   - 64-bit index support
   - scalar-type breadth
   - later algorithm-family expansion
2. Land the first end-to-end abstraction or build-surface support where
   justified.
3. Reconcile packaging, docs, and tests with that capability work.

Exit criteria:

- the repo no longer treats 32-bit real-only as a static permanent ceiling
- at least one material capability ceiling has a shipped modernization path
- the next epic does not have to restart capability planning from zero

### 12. Run a final integration sweep and close Epic 7 from a measured baseline

Work:

1. Re-run the full maintained reviewed baseline.
2. Reconcile the final public story across:
   - docs
   - examples
   - headers
   - benchmarks
   - install/package surfaces
   - tests
3. Record final measured outcomes and residual limits.
4. Close the epic with a retrospective and handoff package.

Exit criteria:

- validated final close state
- explicit residual limits
- no hidden ambiguity about whether Epic 7 actually closed the identified gaps

## Expected Outcome

If Epic 7 completes this plan successfully, the project should end up with:

- a more modern core product model
- a much smaller and clearer env-var compatibility surface
- a deeper and more credible backend/performance architecture
- a stronger benchmark/performance-governance story
- a more mature release/platform/install surface
- smaller remaining source and test hotspots
- cleaner public/reference docs
- and a more credible path toward state-of-the-art sparse-library capability
  breadth

That would move the repository from “excellent self-contained sparse C
library” much closer to “serious state-of-the-art sparse linear algebra
library.”
