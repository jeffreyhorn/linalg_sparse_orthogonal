# EPIC 6 Remediation Todo

**Date:** 2026-06-08  
**Reviewer:** Codex  
**Purpose:** Step-by-step plan to address the gaps identified in
`review-codex-2026-06-08.md`.

## Goal

Use Epic 6 to move the repository from a strong post-Epic-5 engineering library
toward a more state-of-the-art sparse linear algebra product surface.

The main themes are:

- converge the direct-solver usability model
- replace process-global tuning with clearer typed configuration
- modernize the performance/backend architecture selectively
- improve benchmark and platform quality governance
- reduce the hardest remaining maintainability hotspots
- strengthen high-value numerical assurance
- and close the gap between “excellent codebase” and “best-in-class library”

## Steps

### 1. Freeze the Epic 6 baseline and define the remaining product-shape goals

Work:

1. Reconfirm the post-Epic-5 validation and reviewed-baseline anchors.
2. Re-read the Epic 5 retrospective and residual queue.
3. Decide which “state-of-the-art” gaps are real Epic 6 goals versus explicit
   non-goals:
   - direct lifecycle polish
   - configuration-surface modernization
   - backend/performance architecture
   - platform/packaging convergence
   - maintainability/test/doc cleanup
4. Write the explicit Epic 6 non-goal fence before code starts.

Exit criteria:

- one written Epic 6 architecture/productization contract
- one explicit list of state-of-the-art goals
- one explicit list of non-goals and deferred ambitions

### 2. Replace the strongest process-global environment-variable controls with typed per-call option surfaces

Work:

1. Inventory the live `SPARSE_ND_*`, `SPARSE_FM_*`, and related advanced
   controls.
2. Decide which controls become:
   - public typed options
   - internal typed options
   - or remain legacy env-var overrides only
3. Define precedence rules between typed options and env vars.
4. Land the first migrated option surfaces and their tests/docs.

Exit criteria:

- the highest-value advanced controls no longer require global env vars for
  routine use
- precedence and compatibility rules are explicit
- the repo has a clearer typed tuning story

### 3. Harden the direct-solver public usability model

Work:

1. Re-audit the one-shot direct APIs for matrix-mutation and cancellation
   surprises.
2. Decide what additional helpers, wrappers, or state-hardening rules are
   needed to reduce caller discipline requirements.
3. Improve the relationship between:
   - one-shot direct APIs
   - `sparse_analysis_t`
   - `sparse_factors_t`
4. Add direct regression proof for the tightened usability contract.

Exit criteria:

- the direct-solver story is easier to teach and use safely
- mutable-matrix caveats are narrower and better structured
- the explicit lifecycle path and the compatibility path feel more coherent

### 4. Deepen internal direct-lifecycle uniformity where the public model still rests on heterogeneous paths

Work:

1. Audit which direct families still delegate through uneven one-shot or
   backend-specific paths.
2. Prioritize the highest-value remaining heterogeneity:
   - LU lifecycle consistency
   - CSC direct repeated-run uniformity
   - solve/refactor semantics alignment
3. Land bounded internal follow-through without reopening the public model.
4. Refresh the benchmark and regression surfaces that prove the path.

Exit criteria:

- fewer “special-case” direct repeated-run paths remain
- the performance and correctness story is more uniform across direct families

### 5. Introduce a bounded performance/backend architecture layer

Work:

1. Audit the highest-value dense-kernel and supernodal hotspots.
2. Decide what a minimal backend/performance abstraction should own:
   - dense kernels
   - optional BLAS/LAPACK integration
   - threading policy hooks
   - future accelerator-extensibility seams
3. Land the first backend-capable layer on selected kernels only.
4. Keep the default self-contained build path intact.

Exit criteria:

- the repo has a real performance/backend seam, not only hardcoded kernels
- optional acceleration can compose with the existing self-contained build
- the project is closer to a state-of-the-art performance architecture

### 6. Turn the benchmark surface into a clearer performance-governance surface

Work:

1. Re-rank the benchmark drivers by long-term product value.
2. Define stable categories:
   - correctness-adjacent proof benches
   - regression-sensitive performance benches
   - exploratory/developer benches
3. Normalize machine-readable output and documentation conventions.
4. Add a bounded local/CI-friendly regression-check layer where justified.

Exit criteria:

- benchmark roles are explicit
- the project has a cleaner top-level performance characterization story
- performance claims become easier to review and maintain

### 7. Converge the remaining platform, packaging, and release-shape gaps

Work:

1. Reassess:
   - macOS dead-code staging
   - Windows reviewed-wrapper/dead-code gaps
   - serialized dead-code topology
2. Improve the packaging/release surface:
   - shared-library strategy or explicit static-only rationale
   - ABI/versioning/deprecation guidance
   - install/package verification where missing
3. Align docs and CI with the resulting contract.

Exit criteria:

- each major platform/release gap has a fresh explicit disposition
- the packaging story is more product-like and less “developer install only”
- the platform contract is closer to best-in-class honesty and completeness

### 8. Reduce the remaining large implementation hotspots

Work:

1. Re-rank the largest remaining source files by ownership pain.
2. Continue decomposition only where there is a real seam:
   - CSC direct-solver residuals
   - iterative residuals
   - graph/reorder residuals
3. Remove stale sprint-history commentary from permanent code while preserving
   durable algorithm explanations.

Exit criteria:

- the remaining hotspot files are smaller or more clearly owned
- permanent code contains less sprint-local narrative
- maintainers can change the hardest code with less review friction

### 9. Reduce the remaining giant-test and numerical-assurance gaps

Work:

1. Continue giant-test helper extraction or splitting where it improves review
   clarity.
2. Add stronger second-layer assurance:
   - differential/oracle comparisons
   - property tests
   - expanded fuzz coverage where practical
   - harder lifecycle and CSC path stress
3. Improve coverage on the reduced or excluded platform paths where justified.

Exit criteria:

- the biggest test surfaces are easier to review
- the hardest solver/workflow paths have stronger assurance than
  self-consistency alone

### 10. Re-tier the docs/examples/reference story around clearer adoption paths

Work:

1. Decide what belongs in:
   - README
   - tutorial
   - examples
   - benchmark docs
   - public headers
   - maintainer guide
2. Reduce the remaining dense workflow duplication.
3. Add or revise a small number of high-signal examples for advanced repeated
   workflows where they are part of the supported product story.
4. Keep planning history in `docs/planning/`.

Exit criteria:

- docs read more like product docs than project archaeology
- advanced workflows are easier to discover from runnable examples
- the public story is easier to scan by audience and task

### 11. Run a final integration sweep across API, docs, examples, benchmarks, and platforms

Work:

1. Validate the one-shot and explicit lifecycle stories side-by-side.
2. Validate the typed-configuration story versus legacy overrides.
3. Validate the benchmark/performance governance story.
4. Reconcile package/install/platform claims against CI and local maintained
   surfaces.

Exit criteria:

- no major caller or maintainer surface contradicts another
- the final Epic 6 story is visible in code, docs, examples, benchmarks, and
  validation surfaces

### 12. Close Epic 6 from a measured baseline

Work:

1. Run the full maintained quality gates.
2. Reconfirm reviewed truthfulness anchors.
3. Record final measured outcomes and residual limits.
4. Close the epic with a retrospective and handoff package.

Exit criteria:

- validated final close state
- explicit residual limits
- no hidden “cleanup epic later” ambiguity

## Expected Outcome

If Epic 6 completes this plan successfully, the project should end up with:

- a more coherent and easier-to-use direct-solver model
- typed advanced configuration instead of env-var-first tuning
- a more modern performance/backend architecture
- a cleaner benchmark/performance governance story
- narrower platform and packaging gaps
- smaller remaining maintainability hotspots
- stronger numerical assurance on the hardest paths
- and a more credible claim to be a state-of-the-art sparse linear algebra
  library rather than only a very strong engineering codebase
