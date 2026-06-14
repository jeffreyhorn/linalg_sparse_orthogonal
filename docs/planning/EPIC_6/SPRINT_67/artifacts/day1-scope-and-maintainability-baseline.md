# Sprint 67 Day 1: Scope and Maintainability Baseline

Date: 2026-06-13
Branch: `sprint-67`

## Purpose

Freeze the Sprint 67 starting point before implementation work begins by
reconfirming the inherited Sprint 66 contract, the preserved reviewed
baseline, the strongest live large-source maintainability hotspots, and the
most important docs/build/implementation/proof surfaces the sprint will touch
next.

## Authoritative Inputs

- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- `docs/planning/EPIC_6/SPRINT_67/PLAN.md`
- `docs/planning/EPIC_6/SPRINT_66/RETROSPECTIVE.md`
- `docs/planning/EPIC_6/SPRINT_66/artifacts/day14-closeout-and-handoff.md`
- current reviewed baseline surfaces:
  - `make -n quality-review-full`
  - `make quality-review-cmake-compile`
  - `ctest -N --test-dir build/quality-review-cmake`
- current live maintainability/build/truth/proof surfaces measured directly
  from the repo

## Day 1 Baseline Conclusions

### 1. Sprint 67 starts from a preserved Sprint 66 validated close, not from renewed packaging or platform work

Sprint 66 already landed the bounded packaging, ABI, install/export, and
workflow/CI contract reconciliation package. That means Sprint 67 is not
reopening the static-first package story, the narrow ABI/version contract, or
the staged platform-residual interpretation except where a touched ownership
seam proves it is truly required. Sprint 67 is the first bounded Epic 6 sprint
after that close whose center is source ownership and large-file
maintainability again.

### 2. The strongest local reviewed baseline remains the authoritative Sprint 67 starting point

The maintained local truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Sprint 67 should inherit that exact validation story. It should not invent a
lighter maintainability-only truth surface disconnected from the reviewed
baseline.

### 3. The highest-value Sprint 67 problem is concentrated in graph/reorder, CSC, iterative, and chronology seams

The live repo shows a clear concentration:

- strongest coordination-heavy graph/reorder seams:
  - `src/sparse_graph.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder_amd_qg.c`
- strongest CSC/direct/analysis residual seams:
  - `src/sparse_analysis.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc.c`
- strongest iterative/eigensolver residual seams:
  - `src/sparse_iterative.c`
  - `src/sparse_eigs.c`
- strongest proof/adoption surfaces likely to move with decomposition:
  - `tests/test_graph.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `tests/test_integration.c`
  - `examples/example_analysis.c`

So the opening Sprint 67 batch should not pretend every oversized file is an
equal target. The highest-value work is hotspot ranking plus bounded ownership
extraction on the files where orchestration, family-local logic, and stale
chronology are still colliding.

### 4. Sprint 67 reduces cleanly to six bounded workstreams

The project-plan scope collapses to:

1. residual hotspot audit
2. graph/reorder decomposition
3. CSC/iterative residual decomposition
4. comment/chronology cleanup
5. build/regression alignment
6. validation and closeout

This is the right Day 1 shape because it turns a broad Epic 6 maintainability
goal into a smaller implementation contract.

### 5. The strongest live Sprint 67 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- caller-facing docs and maintained truth surfaces:
  - `README.md` = `1020`
  - `docs/maintainer_guide.md` = `548`
  - `CMakeLists.txt` = `413`
  - `Makefile` = `897`
- public coordination headers:
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_reorder.h` = `186`
  - `include/sparse_cholesky.h` = `232`
  - `include/sparse_ldlt.h` = `334`
  - `include/sparse_iterative.h` = `765`
  - `include/sparse_eigs.h` = `650`
- strongest implementation/hotspot seams:
  - `src/sparse_graph.c` = `801`
  - `src/sparse_graph_coarsen.c` = `641`
  - `src/sparse_graph_bisect.c` = `528`
  - `src/sparse_graph_refine.c` = `629`
  - `src/sparse_graph_separator.c` = `297`
  - `src/sparse_reorder_nd.c` = `743`
  - `src/sparse_reorder_amd_qg.c` = `611`
  - `src/sparse_analysis.c` = `1020`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt_csc.c` = `2130`
  - `src/sparse_iterative.c` = `1985`
  - `src/sparse_eigs.c` = `1534`
- strongest proof/adoption surfaces likely to matter in Sprint 67:
  - `tests/test_integration.c` = `2367`
  - `tests/test_graph.c` = `2900`
  - `tests/test_reorder_nd.c` = `2262`
  - `tests/test_chol_csc.c` = `4716`
  - `tests/test_ldlt_csc.c` = `3680`
  - `tests/test_iterative.c` = `2802`
  - `tests/test_eigs.c` = `1522`
  - `examples/example_analysis.c` = `210`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `407`
  - `benchmarks/bench_iterative_reuse.c` = `395`
  - `benchmarks/bench_eigs_reuse.c` = `278`

These are not all immediate edit targets, but they are the real Day 1 map for
where maintainability pressure now lives.

## Preserved Day 1 Non-Goal Fence

Sprint 67 Day 1 confirms the following non-goals before deeper work begins:

- no fake maintainability wins that blur real ownership
- no broad feature work disguised as decomposition
- no reopening packaging/platform/build-surface churn unless a touched seam
  truly requires it
- no weakening of the reviewed truthfulness contract
- no broad style-only cleanup wave disconnected from actual ownership seams
- no chronology scrub that removes durable technical explanations just to erase
  sprint-history traces

## Day 1 Exit State

Sprint 67 now starts from one explicit large-source maintainability baseline:

- the Sprint 66 productization close is still active and unchanged
- the strongest local reviewed baseline remains unchanged
- the reviewed CMake parity anchor has been re-established locally at `53`
- the broad Epic 6 maintainability claim has already narrowed to hotspot
  audit, graph/reorder decomposition, CSC/iterative residual decomposition,
  chronology cleanup, build/regression alignment, and closeout
- the next step is to rank those live hotspot seams precisely before writing
  the bounded validation and audit follow-through
