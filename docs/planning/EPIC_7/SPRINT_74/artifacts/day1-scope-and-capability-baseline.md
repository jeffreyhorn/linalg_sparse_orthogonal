# Sprint 74 Day 1: Scope and Capability Baseline

Date: 2026-06-16
Branch: `sprint-74`

## Purpose

Turn the Sprint 74 project-plan scope plus the Sprint 70, Sprint 72, and
Sprint 73 handoff into one bounded capability-modernization sprint, with the
strongest live capability ceilings and non-goal fence fixed before deeper
audit begins.

## Main Result

Sprint 74 now starts from a precise capability queue, not from another
planning reset and not from another product-model or public-surface cleanup
wave.

The strongest next Epic 7 queue is explicitly:

- capability ceiling rerank
- bounded index/scalar architecture design
- first end-to-end index-width modernization seam
- scalar-surface preparation only where later widening truly needs it
- proof and docs follow-through only where the landed capability work truly
  moves the maintained contract

## Preserved Fence

The Sprint 70 capability and truthfulness fence remains explicit:

- no one-shot full type-generic conversion
- no fake complex-readiness story without end-to-end proof
- no broad capability widening disguised as local cleanup
- no widened reviewed/platform/install claims detached from shipped evidence

## Live Baseline Anchors

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

## Strongest Likely Sprint 74 Touch Surfaces

Raw Day 1 `wc -l` counts from the live tree:

### Maintained public and policy surfaces

- `README.md` = `1037`
- `INSTALL.md` = `237`
- `docs/maintainer_guide.md` = `621`
- `include/sparse_types.h` = `233`
- `include/sparse_matrix.h` = `604`
- `include/sparse_analysis.h` = `499`
- `include/sparse_iterative.h` = `765`
- `include/sparse_eigs.h` = `650`

### Capability-modernization implementation seams

- `src/sparse_types.c` = `50`
- `src/sparse_matrix.c` = `1073`
- `src/sparse_iterative.c` = `1985`
- `src/sparse_eigs.c` = `1534`
- `src/sparse_svd.c` = `1319`

### Strongest proof and adoption/reporting surfaces

- `tests/test_sparse_matrix.c` = `1054`
- `tests/test_integration.c` = `2448`
- `tests/test_iterative.c` = `2802`
- `tests/test_eigs.c` = `1522`
- `examples/example_analysis.c` = `210`
- `examples/example_basic_solve.c` = `110`
- `benchmarks/bench_refactor_csc.c` = `611`
- `benchmarks/bench_chol_csc.c` = `407`

## Capability Ceiling Families

The live capability ceiling remains concentrated in these families:

### Index-width path

Centered on `include/sparse_types.h` and the matrix/public product shell:

- `idx_t` still aliases `int32_t`
- `IDX_MAX` still aliases `INT32_MAX`
- the public migration story is still a documented recompile path rather than
  a shipped broader-width product surface

### Scalar-surface breadth

Centered on iterative/eigs interfaces and their implementation lanes:

- callback signatures in `include/sparse_iterative.h` still expose
  `const double *` and `double *`
- dense working arrays and eigensolver kernels in `src/sparse_iterative.c`,
  `src/sparse_eigs.c`, and `src/sparse_svd.c` remain real-only `double`
  surfaces

### Later algorithm-breadth ceiling

Still centered on the narrower public eigensolver story:

- symmetric sparse eigensolver strength is real
- broader algorithm-family widening remains later than the first width and
  scalar-preparation lane

## Interpretation

The live tree says Sprint 74 should start here:

- index width remains the strongest first modernization lane
- scalar-surface preparation remains the strongest second lane
- broader algorithm-surface widening should remain out of scope
- proof cost remains concentrated in permanent matrix, integration,
  iterative, and eigensolver owners rather than in new generic proof surfaces

## Exit State

Sprint 74 Day 1 closes with:

1. one capability-modernization starting queue
2. one explicit non-goal fence
3. one live reviewed baseline anchor
4. one ranked live capability hotspot map
