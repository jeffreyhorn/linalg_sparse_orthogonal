# Sprint 70 Day 1: Scope and Architecture Baseline

Date: 2026-06-15
Branch: `sprint-70`

## Purpose

Freeze the Sprint 70 starting point before implementation work begins by
reconfirming the inherited Epic 6 closeout contract, the preserved reviewed
baseline, the strongest live product-model and capability hotspots, and the
most important public, implementation, proof, and project-level surfaces that
Epic 7 will touch next.

## Authoritative Inputs

- `docs/planning/EPIC_7/PROJECT_PLAN.md`
- `docs/planning/EPIC_7/SPRINT_70/PLAN.md`
- `docs/planning/EPIC_6/EPIC_6_RETROSPECTIVE.md`
- `docs/planning/EPIC_7/reviews/review-codex-2026-06-15.md`
- `docs/planning/EPIC_7/reviews/todo-codex-2026-06-15.md`
- current reviewed baseline surfaces:
  - `make -n quality-review-full`
  - `make quality-review-cmake-compile`
  - `ctest -N --test-dir build/quality-review-cmake`
- current live public/product/proof/planning surfaces measured directly from
  the repo

## Day 1 Baseline Conclusions

### 1. Sprint 70 starts from a preserved Epic 6 validated close plus an already-grounded Epic 7 review package

Epic 6 already closed the highest-value productization, benchmark-governance,
packaging/platform, maintainability, and assurance gaps that previously kept
the repo from feeling partially finished.

That means Sprint 70 is not rediscovering the library. It is taking the live
post-Epic-6 tree plus the Epic 7 review package and freezing the exact
starting contract for the next epic.

The opening Epic 7 work should therefore be treated as:

- baseline recheck
- product-model gap inventory
- capability ceiling audit
- public/proof-surface contradiction audit
- validation and platform contract freeze
- closeout and handoff

### 2. The strongest local reviewed baseline remains the authoritative Sprint 70 truth surface

The maintained local truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The Day 1 pass re-materialized the reviewed CMake parity tree through
`make quality-review-cmake-compile`, so Sprint 70 starts from a live reviewed
compile/build/test-registration surface rather than from a stale tree.

### 3. The highest-value Epic 7 problem is now structural, not foundational

The live repo still looks like a serious engineering-grade sparse linear
algebra library. The strongest remaining gaps are no longer absent core solver
families or absent quality discipline.

The real Day 1 pressure is concentrated in:

1. core product-model convergence
2. capability-surface expansion on the most limiting ceilings
3. residual configuration modernization
4. backend/performance architecture deepening
5. packaging/platform/install convergence beyond the current asymmetric truth
   surface
6. residual large-source, giant-test, and permanent-surface cleanup

Sprint 70 therefore should not widen into feature work. It should sharpen the
exact architecture and non-goal fence around those six lanes.

### 4. The strongest live Sprint 70 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- maintained public product and policy surfaces:
  - `README.md` = `1034`
  - `docs/maintainer_guide.md` = `578`
  - `INSTALL.md` = `237`
  - `include/sparse_matrix.h` = `583`
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_cholesky.h` = `232`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
  - `include/sparse_types.h` = `233`
- core product-model and numeric-path implementation seams:
  - `src/sparse_matrix.c` = `1052`
  - `src/sparse_chol_csc.c` = `1536`
  - `src/sparse_ldlt_csc.c` = `2130`
  - `src/sparse_lu_csr.c` = `1665`
  - `src/sparse_qr.c` = `1563`
  - `src/sparse_iterative.c` = `1985`
  - `src/sparse_eigs.c` = `1534`
  - `src/sparse_graph.c` = `821`
  - `src/sparse_reorder_nd.c` = `739`
  - `src/sparse_graph_coarsen.c` = `641`
  - `src/sparse_graph_refine.c` = `629`
  - `src/sparse_reorder_amd_qg.c` = `611`
- strongest proof/benchmark/reporting support surfaces:
  - `tests/test_integration.c` = `2411`
  - `tests/test_chol_csc.c` = `4608`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_reorder_nd.c` = `2262`
  - `tests/test_graph.c` = `2900`
  - `tests/test_iterative.c` = `2802`
  - `tests/test_eigs.c` = `1522`
  - `tests/test_qr.c` = `3197`
  - `tests/test_svd.c` = `2766`
  - `tests/test_fuzz.c` = `651`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `407`
  - `benchmarks/bench_iterative_reuse.c` = `395`
  - `benchmarks/bench_eigs_reuse.c` = `278`
- project-level Epic 7 planning and review surfaces:
  - `docs/planning/EPIC_7/PROJECT_PLAN.md` = `356`
  - `docs/planning/EPIC_7/reviews/review-codex-2026-06-15.md` = `542`

These are not all immediate edit targets, but they are the real Day 1 map for
where Epic 7 pressure and architecture risk now live.

### 5. The current tree still confirms the strongest Epic 7 ceiling categories from the review

The Day 1 pass directly re-confirmed the strongest review-backed ceiling
categories:

- linked-list-first public product model versus compressed numeric paths:
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_lu_csr.c`
  - `src/sparse_qr.c`
- narrow capability surface:
  - `README.md` still documents `idx_t` as `int32_t`
  - `README.md` still documents real-only scalar support
  - `include/sparse_types.h` still fixes `idx_t` to `int32_t`
- residual configuration/env-var surface:
  - `src/sparse_svd.c` still carries `SPARSE_SVD_LOWRANK_OUTER`
  - graph/reorder and analysis surfaces still carry a wider typed-policy and
    residual compatibility/debug surface that Epic 7 must rerank
- asymmetric platform/package truth surface:
  - `CMakeLists.txt` still preserves a static-first package shape
  - macOS and Windows workflow/docs surfaces still carry explicit staged or
    narrower reviewed lanes

So the Epic 7 review remains directionally correct against the live repo. Day
1 did not uncover a different first-order problem than the review already
named.

## Preserved Day 1 Non-Goal Fence

Sprint 70 Day 1 confirms the following non-goals before deeper work begins:

- no fake “state-of-the-art” claims not grounded in the live repo
- no broad implementation work disguised as baseline setup
- no reopening solved Epic 6 seams unless a touched Sprint 70 audit seam truly
  requires it
- no weakening of the reviewed truthfulness contract
- no broad cleanup wave disconnected from actual product-model, capability,
  performance, or platform ceilings
- no capability or backend promises before the exact first modernization fences
  are fixed in writing

## Day 1 Exit State

Sprint 70 now starts from one explicit Epic 7 baseline:

- the Epic 6 validated close and Epic 7 review package are both active and
  unchanged
- the strongest local reviewed baseline remains unchanged
- the reviewed CMake parity anchor has been re-established locally at `53`
- the broad Epic 7 goal has already narrowed to product-model convergence,
  capability expansion, configuration modernization, backend/performance
  maturity, packaging/platform convergence, and residual source/test/surface
  cleanup
- the next step is to recheck the live validation contract precisely before
  the deeper product-model and capability audits begin
