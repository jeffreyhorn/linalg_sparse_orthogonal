# Sprint 69 Day 1: Scope and Public Surface Baseline

Date: 2026-06-15
Branch: `sprint-69`

## Purpose

Freeze the Sprint 69 starting point before implementation work begins by
reconfirming the inherited Sprint 68 contract, the preserved reviewed
baseline, the strongest live public-surface and Epic-closeout hotspots, and
the most important docs/header/example/benchmark/project surfaces the sprint
will touch next.

## Authoritative Inputs

- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- `docs/planning/EPIC_6/SPRINT_69/PLAN.md`
- `docs/planning/EPIC_6/SPRINT_68/RETROSPECTIVE.md`
- `docs/planning/EPIC_6/SPRINT_68/artifacts/day14-closeout-and-handoff.md`
- current reviewed baseline surfaces:
  - `make -n quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake`
- current live public/truth/proof surfaces measured directly from the repo

## Day 1 Baseline Conclusions

### 1. Sprint 69 starts from a preserved Sprint 68 validated close, not from renewed subsystem work

Sprint 68 already landed the last bounded giant-test and second-layer
assurance package that Epic 6 still needed:

- `tests/test_chol_csc.c` maintainability relief
- stronger large-`n` CSC-backed Cholesky public-path oracle coverage
- bounded seeded lifecycle property follow-through
- aligned tests/examples/benchmarks ownership wording
- tighter platform-confidence wording for the reduced Windows subset

That means Sprint 69 is not reopening another isolated feature or architecture
lane. Its center is the final integrated public product story and epic-level
closeout.

### 2. The strongest local reviewed baseline remains the authoritative Sprint 69 starting point

The maintained local truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Sprint 69 should inherit that exact validation story. It should not invent a
lighter closeout-only truth surface disconnected from the reviewed baseline.

### 3. The highest-value Sprint 69 problem is concentrated in final public-surface reconciliation, not in more implementation churn

The live repo shows a clear concentration:

- strongest public product and adoption surfaces:
  - `README.md`
  - `docs/tutorial.md`
  - `examples/README.md`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- strongest public reference and interpretation surfaces:
  - `include/sparse_analysis.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`
- strongest proof/adoption/reporting support surfaces:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_fuzz.c`
  - `tests/test_framework_optin.c`
  - `examples/example_analysis.c`
  - `examples/example_basic_solve.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`

So the opening Sprint 69 batch should not pretend every Epic 6 surface is an
equal target. The highest-value work is hotspot ranking plus bounded
product-story simplification and final cross-surface reconciliation on the
surfaces where user-facing story, proof ownership, and project-level closeout
still intersect.

### 4. Sprint 69 reduces cleanly to six bounded workstreams

The project-plan scope collapses to:

1. public surface audit
2. docs/examples productization
3. cross-surface compatibility sweep
4. full validation
5. Epic 6 summary and handoff
6. project-level residual finalization

This is the right Day 1 shape because it turns a broad Epic 6 closeout goal
into a smaller implementation contract.

### 5. The strongest live Sprint 69 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- maintained public product surfaces:
  - `README.md` = `1034`
  - `docs/tutorial.md` = `477`
  - `examples/README.md` = `161`
  - `benchmarks/README.md` = `356`
  - `docs/maintainer_guide.md` = `578`
- likely public header/reference surfaces:
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_cholesky.h` = `232`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
- strongest proof/adoption/reporting surfaces:
  - `tests/test_integration.c` = `2411`
  - `tests/test_chol_csc.c` = `4608`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_reorder_nd.c` = `2262`
  - `tests/test_fuzz.c` = `651`
  - `tests/test_framework_optin.c` = `85`
  - `examples/example_analysis.c` = `210`
  - `examples/example_basic_solve.c` = `110`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `407`
  - `benchmarks/bench_iterative_reuse.c` = `395`
  - `benchmarks/bench_eigs_reuse.c` = `278`
- project-level closeout surface:
  - `docs/planning/EPIC_6/PROJECT_PLAN.md` = `344`

These are not all immediate edit targets, but they are the real Day 1 map for
where final public-story pressure and closeout risk now live.

## Preserved Day 1 Non-Goal Fence

Sprint 69 Day 1 confirms the following non-goals before deeper work begins:

- no fake product simplification that weakens the maintained truthfulness
  contract
- no broad implementation work disguised as public-surface cleanup
- no inflated cross-platform confidence claims beyond reviewed evidence
- no reopening settled Sprint 60-68 seams unless a touched public surface
  truly requires it
- no broad style-only cleanup wave disconnected from actual product-story
  contradictions
- no fake Epic 6 closeout that skips the measured validation baseline

## Day 1 Exit State

Sprint 69 now starts from one explicit public-surface and Epic-closeout
baseline:

- the Sprint 68 giant-test and assurance close is still active and unchanged
- the strongest local reviewed baseline remains unchanged
- the reviewed CMake parity anchor has been re-established locally at `53`
- the broad Epic 6 closeout claim has already narrowed to public audit,
  docs/examples productization, compatibility sweep, validation, Epic 6
  handoff, and project-level residual finalization
- the next step is to recheck the live validation contract precisely before
  the final public-surface audit begins
