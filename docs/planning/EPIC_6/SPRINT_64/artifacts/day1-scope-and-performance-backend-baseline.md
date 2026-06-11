# Sprint 64 Day 1: Scope and Performance Backend Baseline

Date: 2026-06-11
Branch: `sprint-64`

## Purpose

Freeze the Sprint 64 starting point before implementation work begins by
reconfirming the inherited Sprint 63 contract, the preserved reviewed
baseline, the strongest live dense-kernel and supernodal hotspots, and the
most important docs/build/implementation/proof surfaces the sprint will touch
next.

## Authoritative Inputs

- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- `docs/planning/EPIC_6/SPRINT_64/PLAN.md`
- `docs/planning/EPIC_6/SPRINT_63/RETROSPECTIVE.md`
- `docs/planning/EPIC_6/SPRINT_63/artifacts/day14-closeout-and-handoff.md`
- current reviewed baseline surfaces:
  - `ctest -N --test-dir build/quality-review-cmake`
  - `make -n quality-review-full`
- current live backend/build/benchmark/proof surfaces measured directly from
  the repo

## Day 1 Baseline Conclusions

### 1. Sprint 64 starts from a frozen Sprint 63 close, not from renewed lifecycle or configuration work

Sprint 63 already landed the bounded direct-lifecycle follow-through package
and closed with the repeated-run direct model intact. That means Sprint 64 is
not reopening one-shot direct semantics, repeated-run ownership, or Phase 1
typed configuration choices. It is the first bounded Epic 6 sprint centered on
backend architecture, selected hot kernels, build/options wiring, fallback
correctness, and benchmark proof.

### 2. The strongest local reviewed baseline remains the authoritative Sprint 64 starting point

The maintained local truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Sprint 64 should not invent a new validation story. It should inherit the
existing one and make later backend, build-option, and kernel edits obey it.

### 3. The highest-value Sprint 64 problem is concentrated in dense-kernel, supernodal, build-option, and benchmark-proof seams

The live repo shows a clear concentration:

- strongest maintained truth surfaces:
  - `README.md`
  - `docs/tutorial.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
- strongest build/config surfaces:
  - `CMakeLists.txt`
  - `Makefile`
- strongest implementation seams:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_ldlt_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_qr.c`
  - `src/sparse_svd.c`
- strongest proof seams:
  - `benchmarks/bench_refactor.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_ldlt_csc.c`
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `examples/example_analysis.c`

So the opening Epic 6 backend batch should not pretend the entire repo is
equally affected. The highest-value work is dense-kernel plus CSC/supernodal
follow-through, with explicit build/option and benchmark proof support.

### 4. Sprint 64 reduces cleanly to seven bounded workstreams

The project-plan scope collapses to:

1. hotspot audit
2. backend abstraction design
3. kernel integration batch 1
4. build and option surface
5. benchmark proof refresh
6. regression and safety checks
7. validation and closeout

This is the right Day 1 shape because it turns a broad Epic 6 backend goal
into a smaller implementation contract.

### 5. The strongest live Sprint 64 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- caller-facing docs and maintained truth surfaces:
  - `README.md` = `988`
  - `docs/tutorial.md` = `469`
  - `docs/maintainer_guide.md` = `398`
  - `benchmarks/README.md` = `249`
- build and option surfaces:
  - `CMakeLists.txt` = `397`
  - `Makefile` = `881`
- public lifecycle/backend-adjacent headers:
  - `include/sparse_analysis.h` = `498`
  - `include/sparse_cholesky.h` = `226`
  - `include/sparse_ldlt.h` = `334`
- strongest implementation/kernel seams:
  - `src/sparse_dense.c` = `506`
  - `src/sparse_chol_csc_supernodal.c` = `556`
  - `src/sparse_ldlt_csc_supernodal.c` = `392`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt_csc.c` = `2127`
  - `src/sparse_qr.c` = `1563`
  - `src/sparse_svd.c` = `1319`
- strongest benchmark/example/proof surfaces likely to matter in Sprint 64:
  - `benchmarks/bench_refactor.c` = `303`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `393`
  - `benchmarks/bench_ldlt_csc.c` = `516`
  - `tests/test_integration.c` = `2367`
  - `tests/test_chol_csc.c` = `4617`
  - `tests/test_ldlt_csc.c` = `3680`
  - `examples/example_analysis.c` = `210`

These are not all immediate edit targets, but they are the real Day 1 map for
where backend-architecture pressure now lives.

## Preserved Day 1 Non-Goal Fence

Sprint 64 Day 1 confirms the following non-goals before deeper work begins:

- no broad framework rewrite
- no fake platform closure beyond reviewed evidence
- no backend widening that weakens the self-contained default build
- no benchmark-governance sprawl disguised as kernel work
- no packaging/platform expansion unless a selected kernel landing proves it is
  a real blocker to the bounded backend path

## Day 1 Exit State

Sprint 64 now starts from one explicit backend-architecture implementation
baseline:

- the Sprint 63 direct-lifecycle close is still active and unchanged
- the strongest local reviewed baseline remains unchanged
- the broad Epic 6 backend-architecture claim has already narrowed to
  dense-kernel, supernodal, build-option, and benchmark-proof seams
- the next step is to rank those live hotspots precisely before writing the
  bounded backend abstraction design
