# Sprint 68 Day 1: Scope and Giant-Test Baseline

Date: 2026-06-13
Branch: `sprint-68`

## Purpose

Freeze the Sprint 68 starting point before implementation work begins by
reconfirming the inherited Sprint 67 contract, the preserved reviewed
baseline, the strongest live giant-test and assurance hotspots, and the most
important docs/proof/support surfaces the sprint will touch next.

## Authoritative Inputs

- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- `docs/planning/EPIC_6/SPRINT_68/PLAN.md`
- `docs/planning/EPIC_6/SPRINT_67/RETROSPECTIVE.md`
- `docs/planning/EPIC_6/SPRINT_67/artifacts/day14-closeout-and-handoff.md`
- current reviewed baseline surfaces:
  - `make -n quality-review-full`
  - `ctest -N --test-dir build/quality-review-cmake`
- current live giant-test/truth/proof surfaces measured directly from the repo

## Day 1 Baseline Conclusions

### 1. Sprint 68 starts from a preserved Sprint 67 validated close, not from renewed implementation-boundary work

Sprint 67 already landed the bounded large-source maintainability package that
Epic 6 still needed:

- graph/reorder ownership extraction
- shared ND compatibility/default-policy convergence
- large-`n` Cholesky analysis-to-CSC handoff convergence
- proof-surface ownership alignment

That means Sprint 68 is not reopening broad source-boundary work as its main
story. Sprint 68 is the first bounded Epic 6 sprint after that close whose
center is giant-test maintenance cost and second-layer assurance again.

### 2. The strongest local reviewed baseline remains the authoritative Sprint 68 starting point

The maintained local truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Sprint 68 should inherit that exact validation story. It should not invent a
lighter test-only truth surface disconnected from the reviewed baseline.

### 3. The highest-value Sprint 68 problem is concentrated in giant tests and assurance lanes, not in generic cleanup

The live repo shows a clear concentration:

- strongest giant direct-family seams:
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_qr.c`
  - `tests/test_ldlt.c`
  - `tests/test_svd.c`
- strongest giant orchestration/cross-family seams:
  - `tests/test_graph.c`
  - `tests/test_integration.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
- bounded assurance support or secondary proof seams:
  - `tests/test_sparse_lu.c`
  - `tests/test_fuzz.c`
  - `tests/test_suitesparse.c`
  - `tests/test_framework_optin.c`

So the opening Sprint 68 batch should not pretend every test surface is an
equal target. The highest-value work is hotspot ranking plus bounded refactor
and assurance expansion on the files where maintenance load and proof burden
are still colliding.

### 4. Sprint 68 reduces cleanly to six bounded workstreams

The project-plan scope collapses to:

1. giant-test residual audit
2. giant-test refactor batch
3. differential/oracle coverage
4. property/fuzz expansion
5. platform-test follow-through
6. validation and closeout

This is the right Day 1 shape because it turns a broad Epic 6 test-assurance
goal into a smaller implementation contract.

### 5. The strongest live Sprint 68 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- caller-facing docs and maintained truth surfaces:
  - `README.md` = `1025`
  - `docs/maintainer_guide.md` = `561`
  - `benchmarks/README.md` = `351`
- strongest giant-test seams:
  - `tests/test_chol_csc.c` = `4751`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_qr.c` = `3197`
  - `tests/test_graph.c` = `2900`
  - `tests/test_iterative.c` = `2802`
  - `tests/test_ldlt.c` = `2798`
  - `tests/test_svd.c` = `2766`
  - `tests/test_integration.c` = `2371`
  - `tests/test_reorder_nd.c` = `2262`
  - `tests/test_eigs.c` = `1522`
- giant-test support or smaller assurance surfaces:
  - `tests/test_sparse_lu.c` = `908`
  - `tests/test_fuzz.c` = `497`
  - `tests/test_suitesparse.c` = `288`
  - `tests/test_framework_optin.c` = `85`
- strongest proof/adoption/reporting surfaces likely to matter in Sprint 68:
  - `examples/example_analysis.c` = `210`
  - `examples/example_basic_solve.c` = `110`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `407`
  - `benchmarks/bench_iterative_reuse.c` = `395`
  - `benchmarks/bench_eigs_reuse.c` = `278`

These are not all immediate edit targets, but they are the real Day 1 map for
where giant-test maintenance pressure and assurance opportunity now live.

## Preserved Day 1 Non-Goal Fence

Sprint 68 Day 1 confirms the following non-goals before deeper work begins:

- no fake assurance wins that only add brittle fixed outputs
- no broad solver-feature work disguised as test refactoring
- no reopening Sprint 67 implementation-boundary work unless a touched test
  seam truly requires it
- no weakening of the reviewed truthfulness contract
- no broad style-only cleanup wave disconnected from actual giant-test
  ownership seams
- no inflated platform-confidence story beyond reviewed evidence

## Day 1 Exit State

Sprint 68 now starts from one explicit giant-test and assurance baseline:

- the Sprint 67 maintainability close is still active and unchanged
- the strongest local reviewed baseline remains unchanged
- the reviewed CMake parity anchor has been re-established locally at `53`
- the broad Epic 6 giant-test claim has already narrowed to residual audit,
  giant-test refactor, oracle coverage, property/fuzz expansion,
  platform-test follow-through, and closeout
- the next step is to rank those live giant-test seams precisely before
  writing the bounded validation and audit follow-through
