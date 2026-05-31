# Code Review

**Date:** 2026-05-31  
**Reviewer:** Codex  
**Scope:** Full-project source review of the current `linalg_sparse_orthogonal`
tree after Epic 4 closeout, with emphasis on usability, maintainability,
correctness, efficiency, documentation, testing, and overall fitness as a
high-quality sparse linear algebra system.

## Executive Summary

The repository is now a serious sparse linear algebra system rather than a
collection of point features:

- the algorithm surface is broad,
- the validation discipline is real,
- the public headers are heavily documented,
- the benchmark and example surfaces are maintained,
- and Epic 4 closed the largest structural backlog from the prior review.

I did **not** find an immediate release-blocking numerical defect during this
review. The strongest remaining issues are no longer missing infrastructure.
They are integration and product-shape issues:

- the direct-solver lifecycle is still too implicit and mutable,
- public repeated-run support is still uneven across solver families,
- the remaining largest translation units and test files are now the main
  maintainability hotspots,
- user-facing docs still carry too much historical and sprint-local narrative,
- and the quality / platform contract is honest but not yet fully converged.

As a high-quality sparse linear algebra system, the project now looks
architecturally credible, but it is not yet as explicit, uniform, or polished
as it should be at the direct-solver lifecycle boundary.

## Findings

### 1. High: direct-solver lifecycle is still too implicit, and mutable matrix state remains the main usability and correctness tradeoff

Epic 4 intentionally accepted the compatibility-facing `SparseMatrix`
state model as a bounded tradeoff. That tradeoff is still the largest remaining
usability and correctness risk in the codebase.

Representative references:

- `README.md`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `include/sparse_analysis.h`
- `examples/example_analysis.c`

Why this matters:

- The same public matrix object still plays multiple roles:
  - original input matrix
  - reordered matrix
  - in-place direct-factor container
  - object carrying hidden factor/permutation state
- Correct use still depends heavily on caller discipline:
  - when to `sparse_copy()`
  - when identity permutations are required
  - when cancellation may already have mutated state
  - when an “original matrix view” must be preserved for later work
- Epic 4 improved the repeated-run story for iterative solvers and
  eigensolvers, but the direct-solver side remains comparatively implicit.

Primary improvement:

- move the direct-solver public story toward explicit lifecycle objects built
  on the existing `sparse_analysis_t` / `sparse_factors_t` precedent instead of
  continuing to rely on mutable `SparseMatrix` state plus documentation.

### 2. High: the analyze/factor/refactor bridge is real, but it is still heterogeneous and leaves both usability and efficiency on the table

The repository already has a public analyze-once / factor-many precedent, but
it is not yet the dominant direct-solver workflow and is still incomplete as a
unified lifecycle surface.

Representative references:

- `include/sparse_analysis.h`
- `README.md`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `docs/planning/EPIC_2/SPRINT_19/RETROSPECTIVE.md`

Why this matters:

- The public analysis surface is one of the best architectural pieces in the
  repo, but it is still not the default mental model for direct-solver callers.
- `include/sparse_analysis.h` explicitly documents that current numeric
  factorization still delegates to underlying one-shot factorization routines
  instead of fully bypassing internal symbolic work.
- Older deferred CSC follow-ons are still visible:
  - `ldlt_csc_from_sparse_with_analysis`
  - more transparent LDL^T dispatch
  - stronger indefinite supernodal factor-many integration
- As a result, the project has the shape of an explicit lifecycle system
  without yet delivering that lifecycle uniformly across its strongest direct
  solver paths.

Primary improvement:

- make the analysis/factor/refactor path the first-class direct-solver repeated
  workflow, then close the remaining CSC and dispatch gaps around it.

### 3. Medium: the largest remaining implementation hotspots are now concentrated in a few files, and those files are large enough to resist safe change

Epic 4 successfully removed the graph monolith as the dominant structural risk.
The remaining code hotspot list is now much clearer.

Representative references:

- `src/sparse_eigs.c` (`3233` lines)
- `src/sparse_ldlt_csc.c` (`2723` lines)
- `src/sparse_iterative.c` (`2361` lines)
- `src/sparse_chol_csc.c` (`2194` lines)
- `src/sparse_svd.c` (`1728` lines)

Why this matters:

- These files now carry the same class of risk the pre-Epic-4 graph subsystem
  used to carry:
  - algorithm logic
  - lifecycle wiring
  - compatibility wrappers
  - and historical commentary
  - all in one place
- `src/sparse_eigs.c` and `src/sparse_chol_csc.c` in particular still contain
  extensive sprint-history narrative directly inside permanent code.
- The remaining decomposition opportunity is no longer ambiguous. The review can
  point to a concrete file list.

Primary improvement:

- split the remaining largest translation units by solver-family ownership and
  by helper-vs-orchestration role, following the same bounded-decomposition
  style that worked for Epic 4 graph code.

### 4. Medium: the test surface is broad and valuable, but the biggest test files are now the dominant maintainability hotspot in the repository

The project is well tested. It is not test-light. The risk is now test
legibility and change cost, not missing coverage in the abstract.

Representative references:

- `tests/test_chol_csc.c` (`4643` lines)
- `tests/test_svd.c` (`3746` lines)
- `tests/test_ldlt_csc.c` (`3637` lines)
- `tests/test_qr.c` (`3197` lines)
- `tests/test_etree.c` (`2962` lines)
- `tests/test_graph.c` (`2900` lines)
- `tests/test_iterative.c` (`2865` lines)

Why this matters:

- Several core test binaries are now larger than most production files.
- Large tests are harder to review for drift, overlap, and accidental blind
  spots.
- The project now has enough lifecycle surfaces that public-handle,
  analysis/factor, direct factor, and one-shot regression cases should be
  easier to audit by intent than they currently are.

Primary improvement:

- continue the Epic 4 large-test maintainability pattern:
  - extract local helpers
  - split by behavior family
  - keep behavior-level coverage
  - and reduce giant single-file binaries where possible

### 5. Medium: public repeated-run support is now real, but it is still uneven across solver families and underrepresented in examples

Epic 4 landed explicit public repeated-run handles, but the public lifecycle
story is still partial rather than truly uniform.

Representative references:

- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `examples/README.md`
- `README.md`

Why this matters:

- Public repeated-run iterative handles currently cover CG and GMRES, not the
  full solver family surface.
- Public eigensolver handles exist, but the examples intentionally remain
  one-shot-first.
- The repeated-run benchmark proof exists, but it is concentrated in dedicated
  reuse benchmarks rather than reflected more broadly in the example and caller
  story.
- This leaves the project with a correct but uneven message:
  repeated-run support is real, but advanced lifecycle usage still looks more
  like a specialized feature than a fully integrated public model.

Primary improvement:

- either extend the public lifecycle model more uniformly across the remaining
  solver families, or tighten and simplify the supported public boundary so it
  is obviously intentional rather than merely partial.

### 6. Medium: documentation quality is high, but public docs and headers still carry too much sprint-history narrative and tuning-local commentary

The documentation is rich, but some of the public surface is still speaking in
the language of sprint delivery history instead of stable long-term contracts.

Representative references:

- `README.md`
- `include/sparse_eigs.h`
- `src/sparse_eigs.c`
- `src/sparse_chol_csc.c`
- `benchmarks/README.md`
- `examples/README.md`

Why this matters:

- `include/sparse_eigs.h` still contains stale “planned for Sprint 21” and
  other sprint-history framing in a permanent public header.
- `README.md` remains strong, but it is still large and highly detailed even
  after Sprint 48 reduction.
- `benchmarks/README.md` and parts of the main README still describe benchmark
  surfaces in sprint-local terms rather than product-level terms.
- Historical rationale belongs in `docs/planning/`, not in the steady-state
  public contract unless it is directly needed to use the API correctly.

Primary improvement:

- reduce public-facing sprint chronology and migrate stable contract language
  toward:
  - concise user guidance in README/tutorial/examples
  - concise API-local caveats in headers
  - deeper historical rationale in planning artifacts

### 7. Medium: the benchmark surface is useful but fragmented, and the system still lacks a cleaner top-level performance characterization story

The repository now has a strong set of benchmark tools, but they still read
more like individually maintained instruments than one coherent performance
story.

Representative references:

- `benchmarks/bench_main.c`
- `benchmarks/bench_eigs.c`
- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `benchmarks/bench_refactor.c`
- `benchmarks/README.md`

Why this matters:

- There are now separate harnesses for:
  - main factor/solve paths
  - eigensolver sweeps
  - repeated-run iterative reuse
  - repeated-run eigensolver reuse
  - and factor-many refactor workflows
- This is good internal evidence, but it is still harder than it should be to
  answer “how should I characterize the system’s major performance modes?”
- A high-quality sparse numerical system benefits from a clearer benchmark
  taxonomy:
  - one-shot solve
  - analyze/factor/refactor
  - repeated-run solver reuse
  - eigensolver backend comparison

Primary improvement:

- rationalize the benchmark surface and its documentation around a smaller set
  of clearly named workflow categories without turning the sprint into a broad
  benchmark-framework rewrite.

### 8. Medium: the quality contract is honest and strong, but platform/dead-code follow-ons remain intentionally staged and should now be revisited

Epic 3 and Epic 4 did the right thing by naming staged and excluded quality
surfaces explicitly. That honesty was good engineering. It also leaves a clear
EPIC 5 follow-up queue.

Representative references:

- `docs/planning/EPIC_3/EPIC_3_RETROSPECTIVE.md`
- `docs/planning/EPIC_4/EPIC_4_RETROSPECTIVE.md`
- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`

Why this matters:

- Dead-code execution remains serialized.
- macOS dead-code remains staged.
- Windows local reviewed-wrapper parity remains staged.
- Windows dead-code remains excluded.
- Coverage remains calibrated to the current 80% enforced reality rather than a
  stronger long-term instrumentation story.

Primary improvement:

- treat the remaining quality-platform gaps as a bounded productization pass:
  not to re-litigate the Epic 3 contract, but to close the pieces that are now
  mature enough to converge.

## Category Assessment

### Usability

Strong overall, but still uneven at the lifecycle boundary.

The basic one-shot paths are easy to find and well documented. The harder parts
are:

- knowing when mutation matters,
- understanding analyze/factor/refactor vs one-shot workflows,
- and knowing when explicit repeated-run handles are worth using.

### Maintainability

Good architecture direction, but the remaining hotspot list is now very clear:

- a handful of large implementation files
- a handful of giant tests
- and public docs/headers that still carry too much historical narrative

This is now the dominant engineering-risk category.

### Correctness

I did not find a clear numerical release blocker in this review pass.

The main correctness risks are structural:

- mutable matrix state
- lifecycle ambiguity
- and uneven direct-solver explicitness

Those are the kinds of issues that create misuse bugs and future regressions
more than obvious present-day algorithm failures.

### Efficiency

Strong compared with the pre-Epic-4 state.

The main remaining efficiency opportunities are:

- deeper analysis/factor/refactor reuse
- direct-solver lifecycle exposure that makes reuse more natural
- remaining CSC deferred items
- and clearer factor-many / repeated-run benchmark and example adoption

### Documentation

Much improved after Sprint 48, but still too dense and too historical in some
permanent public surfaces.

The best next move is not “write more docs.” It is:

- reduce sprint narrative,
- normalize workflow guidance,
- and make the stable product story easier to read.

### Testing

Broad and credible. Coverage by surface is not the main problem.

The real test-side issue is maintainability:

- several core test files are now too large,
- and the final lifecycle workflow story should be easier to see directly in
  the tests.

## Bottom Line

`linalg_sparse_orthogonal` now qualifies as a high-quality sparse linear
algebra system in breadth, validation discipline, and engineering honesty.

The next improvement wave should not be a generic feature sprint. It should be
an integration-and-polish epic focused on:

- explicit direct-solver lifecycle design
- deeper factor-many integration
- remaining large-file decomposition
- public documentation and benchmark simplification
- and final quality-platform convergence

That is the shortest path from “strong engineering project” to a more uniform,
easier-to-use, easier-to-maintain numerical library.
