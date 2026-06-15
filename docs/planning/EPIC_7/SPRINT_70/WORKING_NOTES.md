# Sprint 70 Working Notes

## Day 1 - Scope Audit & Baseline Setup

### Goal

Freeze the Sprint 70 starting point before implementation work begins by
reconfirming the inherited Epic 6 closeout contract, the preserved reviewed
baseline, the strongest live product-model and capability hotspots, and the
most important public, implementation, proof, and project-level surfaces that
Epic 7 will touch next.

### Actions

1. Re-read the Sprint 70 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`,
   the Epic 6 retrospective, the Epic 7 review, and the Epic 7 todo.
2. Re-read the landed Sprint 70 plan and fixed the bounded workstreams that
   the sprint should actually carry:
   - baseline recheck
   - product-model gap inventory
   - capability ceiling audit
   - public-surface and proof-surface audit
   - validation and platform contract freeze
   - closeout and handoff
3. Reconfirmed the strongest reviewed baseline surfaces:
   - `make quality-review-full`
   - `make -n quality-review-full`
4. Re-materialized the reviewed CMake parity tree and rechecked the parity
   anchor:
   - `make quality-review-cmake-compile`
   - `ctest -N --test-dir build/quality-review-cmake`
5. Measured the strongest likely Sprint 70 touch surfaces directly from the
   live tree across:
   - maintained public product and policy surfaces
   - core product-model and numeric-path implementation seams
   - strongest proof/benchmark/reporting support surfaces
   - project-level Epic 7 review and planning surfaces

### Findings

#### 1. Sprint 70 starts from the Epic 6 validated close and the Epic 7 review package, not from a blank-slate architecture exercise

Epic 6 already closed the highest-value foundational gaps that previously kept
the repo from reading like a real product:

- typed high-value analysis and reorder options
- clearer direct repeated-run lifecycle ownership
- canonical benchmark governance and threshold-free reporting
- truthful packaging/install/platform wording
- stronger large-source, giant-test, and assurance follow-through
- a final coherent public product story across README, tutorial, examples,
  benchmarks, tests, and maintainer surfaces

That means Sprint 70 is not inventing Epic 7 from scratch. It is fixing the
starting contract for the work that the Epic 7 review already identified:

- product-model convergence
- capability-surface expansion
- configuration modernization phase 2
- backend/performance maturity phase 2
- packaging/platform convergence phase 2
- residual source/test/public-surface cleanup

Interpretation:

- Sprint 70 is a bounded baseline and architecture-contract sprint, not a
  disguised implementation sprint
- the Epic 7 review and todo are already real inputs, so Day 1 should sharpen
  them against the live tree rather than restate them abstractly

#### 2. The strongest local reviewed baseline remains the authoritative Sprint 70 starting point

The maintained Day 1 truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

The reviewed CMake parity tree was re-materialized locally through:

- `make quality-review-cmake-compile`

Interpretation:

- Sprint 70 inherits the exact same reviewed baseline story as the Epic 6
  close
- even though Day 1 is docs-only planning work, it still starts from the
  strongest reviewed truth surface rather than a lighter planning-only proxy

#### 3. The highest-value Epic 7 pressure is now structural, not foundational

The live repo still reads as a serious engineering-grade sparse linear algebra
library. The remaining pressure is concentrated in ceilings rather than missing
core solver families or missing quality discipline.

The strongest current Day 1 Epic 7 pressure reduces to:

1. core product-model convergence
2. capability-surface expansion on the biggest ceilings
3. residual configuration modernization
4. backend/performance architecture deepening
5. packaging/platform/install convergence beyond the current asymmetric truth
   surface
6. residual large-source, giant-test, and permanent-surface cleanup

Interpretation:

- Sprint 70 should not pretend the repo still needs broad “stabilization”
- the real Day 1 task is freezing the exact target and non-goal fence for
  those structural ceilings before later sprints start landing code

#### 4. The strongest live Sprint 70 touch surfaces are already identifiable from the current tree

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

Interpretation:

- the strongest remaining Epic 7 pressure is concentrated in a finite set of
  permanent public, implementation, proof, and planning surfaces
- Sprint 70 should start by reranking those seams, not by widening into new
  solver or packaging implementation work

#### 5. The Day 1 non-goal fence is now explicit before deeper audit begins

Sprint 70 Day 1 confirms the following non-goals:

- no fake “state-of-the-art” claims not grounded in the live repo
- no broad implementation work disguised as baseline or audit setup
- no reopening solved Epic 6 seams unless a touched Sprint 70 audit seam truly
  forces it
- no weakening of the reviewed truthfulness contract
- no broad cleanup wave disconnected from real product-model, capability,
  performance, or platform ceilings
- no capability or backend promises before the exact first modernization fences
  are frozen

### Day 1 Close

Sprint 70 now starts from one explicit Epic 7 baseline:

- the Epic 6 validated close and the Epic 7 review package are both active and
  unchanged
- the strongest local reviewed baseline remains unchanged
- the reviewed CMake parity anchor has been re-established locally at `53`
- the broad Epic 7 goal has already narrowed to product-model convergence,
  capability expansion, configuration modernization, backend/performance
  maturity, packaging/platform convergence, and residual source/test/surface
  cleanup
- the next step is to recheck the exact validation and rerun contract before
  the deeper product-model and capability audits begin
