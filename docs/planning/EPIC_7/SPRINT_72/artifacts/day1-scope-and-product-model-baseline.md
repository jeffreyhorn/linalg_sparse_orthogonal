# Sprint 72 Day 1: Scope and Product-Model Baseline

Date: 2026-06-16
Branch: `sprint-72`

## Purpose

Turn the Sprint 72 project-plan scope plus the Sprint 70-71 handoff into one
bounded first-phase product-model convergence sprint, with the strongest live
touch surfaces and non-goal fence fixed before deeper audit begins.

## Main Result

Sprint 72 now starts from a precise implementation-facing queue, not from
another planning reset and not from another public-surface cleanup wave.

The strongest next Epic 7 queue is explicitly:

- product-model convergence from the public direct-workflow seam
- bounded ownership cleanup between `SparseMatrix`, compressed working paths,
  and factor/workspace state
- proof and docs follow-through only where the landed implementation truly
  moves the contract

## Preserved Fence

The Sprint 70 architecture and non-goal fence remains explicit:

- no broad `SparseMatrix` rewrite
- no capability widening disguised as ownership cleanup
- no platform/install/package claim widening
- no fake generic abstraction-layer campaign detached from real workflow pain

## Live Baseline Anchors

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity anchor:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

## Strongest Likely Sprint 72 Touch Surfaces

Raw Day 1 `wc -l` counts from the live tree:

### Maintained public/product surfaces

- `README.md` = `1037`
- `docs/maintainer_guide.md` = `578`
- `INSTALL.md` = `237`
- `include/sparse_matrix.h` = `583`
- `include/sparse_analysis.h` = `498`
- `include/sparse_iterative.h` = `765`
- `include/sparse_eigs.h` = `650`

### Product-model / numeric-path seams

- `src/sparse_matrix.c` = `1052`
- `src/sparse_ldlt_csc.c` = `2130`
- `src/sparse_iterative.c` = `1985`
- `src/sparse_lu_csr.c` = `1665`
- `src/sparse_chol_csc.c` = `1536`
- `src/sparse_qr.c` = `1563`
- `src/sparse_eigs.c` = `1534`

### Direct-workflow public-boundary support

- `include/sparse_lu.h` = `362`
- `include/sparse_cholesky.h` = `215`
- `include/sparse_ldlt.h` = `334`

### Strongest proof/adoption surfaces

- `tests/test_chol_csc.c` = `4608`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_qr.c` = `3197`
- `tests/test_graph.c` = `2900`
- `tests/test_iterative.c` = `2802`
- `tests/test_svd.c` = `2766`
- `tests/test_integration.c` = `2411`
- `tests/test_sparse_matrix.c` = `1054`
- `examples/example_analysis.c` = `210`
- `examples/example_basic_solve.c` = `110`

## Interpretation

The live tree still says the same thing the Sprint 70 audit said:

- the strongest product-model pressure is not a broad matrix-model rewrite
- it is the mixed ownership burden at the public direct-workflow seam
- the strongest compressed-path support seams remain CSC/CSR-backed direct
  paths, not generic iterative or eigensolver widening
- the strongest proof cost remains concentrated in the existing permanent test
  owners rather than in new proof surfaces

## Exit State

Sprint 72 Day 1 closes with:

1. one implementation-facing starting queue
2. one explicit non-goal fence
3. one live reviewed baseline anchor
4. one ranked live touch-surface map
