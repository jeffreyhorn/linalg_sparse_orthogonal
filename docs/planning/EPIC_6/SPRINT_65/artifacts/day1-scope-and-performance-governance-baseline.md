# Sprint 65 Day 1: Scope and Performance Governance Baseline

Date: 2026-06-11
Branch: `sprint-65`

## Purpose

Freeze the Sprint 65 starting point before implementation work begins by
reconfirming the inherited Sprint 64 contract, the preserved reviewed
baseline, the strongest live benchmark/performance-governance hotspots, and
the most important docs/build/implementation/proof surfaces the sprint will
touch next.

## Authoritative Inputs

- `docs/planning/EPIC_6/PROJECT_PLAN.md`
- `docs/planning/EPIC_6/SPRINT_65/PLAN.md`
- `docs/planning/EPIC_6/SPRINT_64/RETROSPECTIVE.md`
- `docs/planning/EPIC_6/SPRINT_64/artifacts/day14-closeout-and-handoff.md`
- current reviewed baseline surfaces:
  - `ctest -N --test-dir build/quality-review-cmake`
  - `make -n quality-review-full`
- current live benchmark/build/truth/solver/proof surfaces measured directly
  from the repo

## Day 1 Baseline Conclusions

### 1. Sprint 65 starts from a frozen Sprint 64 close, not from renewed backend-abstraction-first work

Sprint 64 already landed the bounded backend-aware Cholesky CSC supernodal
package and closed with the default self-contained build still authoritative.
That means Sprint 65 is not reopening the first backend-aware lane, the
bounded dense-kernel descriptor decision, or the public meaning of
`SPARSE_ERR_BACKEND_CONTRACT`. It is the first bounded Epic 6 sprint centered
on performance governance, benchmark consolidation, smaller canonical
baselines, and solver-efficiency follow-through selected from real benchmark
evidence.

### 2. The strongest local reviewed baseline remains the authoritative Sprint 65 starting point

The maintained local truth surfaces are still:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

Sprint 65 should inherit that exact validation story. It should not invent a
new performance-only truth surface disconnected from the reviewed baseline.

### 3. The highest-value Sprint 65 problem is concentrated in benchmark-role, output, canonical-baseline, and solver-follow-through seams

The live repo shows a clear concentration:

- strongest maintained truth surfaces:
  - `README.md`
  - `docs/tutorial.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
  - `Makefile`
- strongest benchmark proof surfaces:
  - `benchmarks/bench_refactor.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_ldlt_csc.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
- strongest likely solver/hotspot follow-through seams:
  - `src/sparse_dense.c`
  - `src/sparse_chol_csc_supernodal.c`
  - `src/sparse_ldlt_csc_supernodal.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_iterative.c`
  - `src/sparse_eigs.c`
- strongest proof/adoption seams:
  - `tests/test_integration.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `examples/example_analysis.c`

So the opening Sprint 65 batch should not pretend every benchmark driver is
equally authoritative. The highest-value work is benchmark-role classification
plus output/taxonomy normalization on the maintained proof surfaces, followed
by bounded solver follow-through chosen from that sharper map.

### 4. Sprint 65 reduces cleanly to seven bounded workstreams

The project-plan scope collapses to:

1. benchmark-role audit
2. output and taxonomy normalization
3. canonical performance surface selection
4. solver-efficiency follow-through
5. local and CI-friendly regression/reporting checks
6. docs and example alignment
7. validation and closeout

This is the right Day 1 shape because it turns a broad Epic 6 performance
goal into a smaller implementation contract.

### 5. The strongest live Sprint 65 touch surfaces are already identifiable from the current tree

The highest-value current Day 1 hotspots are:

- caller-facing docs and maintained truth surfaces:
  - `README.md` = `997`
  - `docs/tutorial.md` = `469`
  - `docs/maintainer_guide.md` = `442`
  - `benchmarks/README.md` = `268`
- build and validation truth surfaces:
  - `Makefile` = `881`
  - `CMakeLists.txt` = `397`
- strongest benchmark binaries likely to matter in the first governance pass:
  - `benchmarks/bench_refactor.c` = `303`
  - `benchmarks/bench_refactor_csc.c` = `611`
  - `benchmarks/bench_chol_csc.c` = `406`
  - `benchmarks/bench_ldlt_csc.c` = `516`
  - `benchmarks/bench_iterative_reuse.c` = `370`
  - `benchmarks/bench_eigs_reuse.c` = `253`
- strongest implementation/hotspot seams likely to be influenced by the audit:
  - `src/sparse_dense.c` = `597`
  - `src/sparse_chol_csc_supernodal.c` = `500`
  - `src/sparse_ldlt_csc_supernodal.c` = `392`
  - `src/sparse_chol_csc.c` = `1532`
  - `src/sparse_ldlt_csc.c` = `2127`
  - `src/sparse_iterative.c` = `1985`
  - `src/sparse_eigs.c` = `1534`
- strongest proof/adoption surfaces likely to matter in Sprint 65:
  - `tests/test_integration.c` = `2367`
  - `tests/test_chol_csc.c` = `4716`
  - `tests/test_ldlt_csc.c` = `3680`
  - `examples/example_analysis.c` = `210`

These are not all immediate edit targets, but they are the real Day 1 map for
where performance-governance pressure now lives.

## Preserved Day 1 Non-Goal Fence

Sprint 65 Day 1 confirms the following non-goals before deeper work begins:

- no fake performance claims beyond reviewed evidence
- no benchmark-governance sprawl disconnected from real proof surfaces
- no broad backend/platform rewrite disguised as efficiency work
- no widening that weakens the self-contained default build or truthfulness
  contract
- no fragile pseudo-regression gates that treat noisy local timings as stable
  authoritative signals

## Day 1 Exit State

Sprint 65 now starts from one explicit performance-governance implementation
baseline:

- the Sprint 64 backend-aware close is still active and unchanged
- the strongest local reviewed baseline remains unchanged
- the broad Epic 6 performance-governance claim has already narrowed to
  benchmark-role, output, canonical-baseline, solver-follow-through, and
  regression-reporting seams
- the next step is to rank those live benchmark surfaces precisely before
  writing the bounded normalization and canonical-baseline design
