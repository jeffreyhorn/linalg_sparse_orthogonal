# Sprint 60 Day 1: Scope and Productization Baseline

Date: 2026-06-08
Branch: `sprint-60`


## Purpose

Freeze the starting point for Epic 6 before any implementation work begins by
reconfirming the inherited reviewed baseline, the post-Epic-5 compatibility
fence, the strongest remaining productization themes, and the most important
live repo surfaces the sprint will audit next.

## Authoritative Inputs

- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- `docs/planning/EPIC_6/SPRINT_60/PLAN.md`
- `docs/planning/EPIC_6/reviews/review-codex-2026-06-08.md`
- `docs/planning/EPIC_6/reviews/todo-codex-2026-06-08.md`
- `docs/planning/EPIC_5/EPIC_5_RETROSPECTIVE.md`
- current reviewed baseline surfaces:
  - `ctest -N --test-dir build/quality-review-cmake`
  - `make -n quality-review-full`
- current live hotspot surfaces measured directly from the repo

## Day 1 Baseline Conclusions

### 1. Sprint 60 starts from a closed Epic 5 baseline, not from unresolved lifecycle or feature debt

Epic 5 already closed the major public repeated-run, CSC, maintainability,
docs, and quality/platform backlog. That means Sprint 60 is not recovering
hidden Epic 5 defects. It is setting the contract for a new kind of work:
productization, configuration modernization, performance architecture, platform
convergence, and stronger assurance.

### 2. The strongest local reviewed baseline remains the authoritative Epic 6 starting point

The maintained local truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

That means Sprint 60 should not invent a new baseline. It should freeze the
existing one and make later Epic 6 implementation work obey it.

### 3. The strongest remaining Epic 6 gaps are no longer “missing algorithms”

The inherited review/todo queue is now concentrated in:

- direct-solver usability convergence
- typed configuration replacing global env-var tuning where justified
- backend/performance architecture modernization
- benchmark/performance governance
- packaging/platform maturity
- residual hotspot and test-surface maintainability
- stronger oracle/property/differential assurance

This is an architecture and product-surface problem, not a core-capability
problem.

### 4. Sprint 60 reduces cleanly to six bounded workstreams

The project-plan items collapse to:

1. baseline recheck
2. productization gap inventory
3. state-of-the-art target definition
4. configuration/performance surface audit
5. validation/platform contract freeze
6. sprint closeout package

This is the right shape for the opening Epic 6 sprint because it turns a large
review into a smaller, enforceable execution contract.

### 5. The strongest live Sprint 60 surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- quality/platform contract surfaces:
  - `README.md` = `982`
  - `docs/tutorial.md` = `454`
  - `docs/maintainer_guide.md` = `315`
  - `Makefile` = `881`
  - `.github/workflows/ci.yml` = `221`
  - `.github/workflows/macos-ci.yml` = `111`
  - `.github/workflows/windows-ci.yml` = `54`
- public product/control surfaces:
  - `include/sparse_analysis.h` = `375`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
- architecture-sensitive implementation surfaces:
  - `src/sparse_analysis.c` = `780`
  - `src/sparse_graph.c` = `801`
  - `src/sparse_reorder_nd.c` = `642`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt_csc.c` = `2127`
  - `src/sparse_eigs.c` = `1534`
  - `src/sparse_iterative.c` = `1985`
- performance-story surfaces:
  - `benchmarks/README.md` = `246`
  - `benchmarks/bench_refactor.c` = `303`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_iterative_reuse.c` = `370`
  - `benchmarks/bench_eigs_reuse.c` = `253`
- example and assurance surfaces:
  - `examples/README.md` = `134`
  - `examples/example_analysis.c` = `210`
  - `examples/example_iterative.c` = `144`
  - `examples/example_eigs.c` = `287`
  - `tests/test_integration.c` = `1976`
  - `tests/test_iterative.c` = `2802`
  - `tests/test_eigs.c` = `1522`
  - `tests/test_chol_csc.c` = `4552`
  - `tests/test_ldlt_csc.c` = `3680`

These are not all immediate edit targets, but they are the real Day 1 map for
where productization, architecture, and assurance pressure still lives.

## Preserved Day 1 Non-Goal Fence

Sprint 60 Day 1 confirms the following non-goals before deeper work begins:

- no fake “state of the art” claim before the target is written
- no reopened Epic 5 lifecycle or CSC redesign
- no premature backend, packaging, or platform implementation batch
- no broad feature work
- no hidden expansion of iterative/eigensolver repeated-run support

## Day 1 Exit State

Sprint 60 now starts from one explicit baseline:

- the post-Epic-5 repo is validated and coherent
- the strongest local reviewed baseline remains unchanged
- the Epic 6 problem has already narrowed from “what should we do?” to “which
  productization and architecture gaps are real, and which are non-goals?”
- the next step is to freeze the validation/truthfulness contract before
  starting the deeper productization inventory
