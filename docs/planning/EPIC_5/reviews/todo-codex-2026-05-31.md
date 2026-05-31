# EPIC 5 Remediation Todo

**Date:** 2026-05-31  
**Reviewer:** Codex  
**Purpose:** Step-by-step plan to address the gaps identified in
`review-codex-2026-05-31.md`.

## Goal

Use Epic 5 to move the repository from a structurally strong but still
compatibility-constrained sparse linear algebra system toward a more explicit,
uniform, and product-level lifecycle model.

The main themes are:

- explicit direct-solver lifecycle exposure
- deeper analyze/factor/refactor integration
- remaining maintainability hotspot reduction
- documentation and benchmark simplification
- and final quality/platform convergence

## Steps

### 1. Freeze the Epic 5 baseline and define the public direct-solver lifecycle model

Work:

1. Reconfirm the post-Epic-4 validation and reviewed-baseline anchors.
2. Inventory the current direct-solver public surface:
   - one-shot factor/solve APIs
   - `sparse_analysis_t`
   - `sparse_factors_t`
   - refactor flows
   - examples and benchmarks
3. Decide the steady-state public model:
   - whether Epic 5 extends `sparse_analysis_t` / `sparse_factors_t`
   - or introduces additional opaque direct-solver lifecycle handles
4. Write down the non-goals before code starts.

Exit criteria:

- one written direct-solver lifecycle contract
- one explicit “supported repeated direct workflow” story
- one explicit list of non-goals

### 2. Make the direct-solver repeated workflow explicit and first-class

Work:

1. Expose the chosen lifecycle API for the main direct-solver families.
2. Preserve current one-shot APIs as compatibility wrappers.
3. Ensure prepare / factor / solve / refactor / free semantics are explicit.
4. Keep public state opaque rather than leaking internal CSC or linked-list
   layouts.

Exit criteria:

- public lifecycle API lands for the main direct-solver path
- one-shot APIs still work unchanged
- callers no longer need hidden matrix-state knowledge for the main repeated
  direct workflow

### 3. Deepen the analyze/factor/refactor integration behind that public model

Work:

1. Remove or reduce unnecessary fallback to underlying one-shot symbolic work.
2. Tighten the relationship between analysis, numeric factor, refactor, and
   solve.
3. Make the factor-many path measurably real, not just documented.
4. Verify that analysis-driven paths are the default performance story for
   stable-pattern direct solves.

Exit criteria:

- stronger analysis-driven numeric reuse
- updated direct-solver docs and benchmarks proving the path
- reduced gap between “public lifecycle exists” and “public lifecycle pays off”

### 4. Close the remaining deferred CSC direct-solver gaps

Work:

1. Audit, validate, and if needed extend the existing
   `ldlt_csc_from_sparse_with_analysis` path for the full-pattern indefinite
   workflow.
2. Revisit transparent LDL^T dispatch and analysis-aware CSC routing.
3. Validate indefinite supernodal / factor-many behavior on the intended
   workloads.
4. Reconcile remaining CSC direct-solver deferred comments and docs.

Exit criteria:

- indefinite CSC factor-many path is no longer explicitly incomplete
- LDL^T dispatch story is cleaner and more uniform with Cholesky
- old Epic 2 deferred CSC follow-ons are either closed or explicitly bounded

### 5. Decide whether public repeated-run solver lifecycle should expand or be intentionally bounded

Work:

1. Review the public repeated-run surfaces for:
   - MINRES
   - BiCGSTAB
   - block iterative solvers
   - any remaining eigensolver lifecycle asymmetries
2. Either extend the public lifecycle model or tighten the supported public
   boundary so the asymmetry is clearly intentional.
3. Align benchmarks, docs, and tests with that decision.

Exit criteria:

- no ambiguous “partial by accident” public repeated-run surface remains
- explicit rationale for what is publicly supported and what stays one-shot

### 6. Decompose the remaining large implementation hotspots

Work:

1. Split `src/sparse_eigs.c` by stable ownership seams.
2. Split `src/sparse_iterative.c` further where lifecycle wiring and solver
   logic still compete in one file.
3. Revisit `src/sparse_ldlt_csc.c`, `src/sparse_chol_csc.c`, and `src/sparse_svd.c`
   for the next bounded extraction seams.
4. Remove stale sprint-history blocks from permanent implementation files while
   preserving useful algorithm commentary.

Exit criteria:

- hotspot file sizes reduced
- ownership boundaries clearer
- no major solver file still acts as an all-purpose algorithm host without a
  good reason

### 7. Reduce giant-test maintenance cost without losing behavior-level confidence

Work:

1. Split or helper-extract the largest test binaries:
   - `test_chol_csc`
   - `test_svd`
   - `test_ldlt_csc`
   - `test_qr`
   - `test_etree`
   - `test_iterative`
2. Add direct lifecycle coverage for the final public direct-solver model.
3. Keep benchmark and example parity checks where they prove caller stories.

Exit criteria:

- largest test files are materially easier to review
- lifecycle workflows are directly exercised
- coverage stays behavior-level, not helper-driven

### 8. Simplify the public documentation story

Work:

1. Remove stale sprint-history framing from permanent public headers and README
   sections.
2. Keep planning chronology in `docs/planning/` instead of public API surfaces.
3. Normalize the lifecycle guidance across:
   - README
   - tutorial
   - examples
   - benchmark docs
   - public headers
4. Keep the one-shot-first story where appropriate, but make the advanced
   lifecycle story equally clear.

Exit criteria:

- public docs read like product docs rather than sprint archives
- lifecycle guidance is consistent across user-facing surfaces
- the README becomes easier to scan without losing high-value entry points

### 9. Modernize the example and benchmark characterization story

Work:

1. Decide which examples should stay intentionally one-shot.
2. Add at least one explicit lifecycle/analyze-factor repeated-run example if
   the public surface makes that a supported workflow.
3. Reorganize benchmark documentation around stable workflow categories:
   - one-shot solve
   - analyze/factor/refactor
   - repeated-run iterative reuse
   - repeated-run eigensolver reuse
4. Keep the work bounded; do not turn it into a benchmark framework rewrite.

Exit criteria:

- examples and benchmarks reflect the final public lifecycle story
- benchmark docs are easier to read by workflow rather than sprint history

### 10. Revisit the remaining quality/platform follow-ons from Epic 3

Work:

1. Re-evaluate serialized dead-code execution.
2. Reassess macOS dead-code staging.
3. Reassess Windows reviewed-wrapper parity and dead-code exclusion.
4. Reassess whether the current coverage contract should remain unchanged.

Exit criteria:

- each staged or excluded quality surface has a fresh disposition:
  - fixed
  - still intentionally staged
  - or explicitly deferred again with current rationale

### 11. Run a final integration sweep across API, docs, examples, benchmarks, and tests

Work:

1. Validate the one-shot compatibility paths.
2. Validate the explicit lifecycle paths.
3. Validate analysis/refactor performance claims.
4. Reconcile documentation, examples, benchmarks, and regression naming.

Exit criteria:

- no major caller surface contradicts another
- the final public usage story is visible in code, tests, docs, and benchmarks

### 12. Close Epic 5 from a measured baseline

Work:

1. Run the full maintained quality gates.
2. Reconfirm truthfulness anchors.
3. Record final measured outcomes and residual limits.
4. Close the epic with a retrospective and handoff package.

Exit criteria:

- validated final close state
- explicit residual limits
- no hidden “cleanup sprint later” ambiguity

## Expected Outcome

If Epic 5 completes this plan successfully, the project should end up with:

- a clearer public direct-solver lifecycle model
- a stronger and more uniform repeated-run story
- smaller and easier-to-maintain implementation/test hotspots
- less historical narrative in public docs
- and a tighter fit between product behavior, benchmark evidence, and quality
  policy

That is the right next step for a mature sparse linear algebra library.
