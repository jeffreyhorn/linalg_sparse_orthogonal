# Sprint 67 Day 2: Validation Baseline and Touched-Surface Recheck

Date: 2026-06-13
Branch: `sprint-67`

## Purpose

Reconfirm the reviewed baseline and the targeted rerun set that Sprint 67
decomposition work must preserve before any implementation work lands.

## Reviewed Validation Contract

The strongest local reviewed baseline remains:

- `make quality-review-full`

The reviewed CMake parity anchor remains exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

The authoritative Sprint 67 split is:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial decomposition, ownership-boundary, or
  build/regression-alignment work:
  - `make quality-review-full`
- docs-only days:
  - targeted sanity checks only

## Targeted Sprint 67 Rerun Set

The high-signal Sprint 67 rerun set is:

- cross-family and orchestration proof surfaces:
  - `./build/test_integration`
- graph/reorder family proofs:
  - `./build/test_graph`
  - `./build/test_reorder_nd`
- CSC direct-family proofs:
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
- iterative and eigensolver residual proofs:
  - `./build/test_iterative`
  - `./build/test_eigs`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_basic_solve`
- maintained benchmark/reporting surfaces:
  - `./build/bench_refactor_csc`
  - `./build/bench_chol_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

All of those surfaces were present in the current `build/` tree at Day 2.

## Touched-Surface Recheck

The highest-signal likely Sprint 67 touch surfaces at Day 2 are:

- implementation hotspots:
  - `src/sparse_graph.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_bisect.c`
  - `src/sparse_graph_refine.c`
  - `src/sparse_reorder_nd.c`
  - `src/sparse_reorder_amd_qg.c`
  - `src/sparse_analysis.c`
  - `src/sparse_chol_csc.c`
  - `src/sparse_ldlt_csc.c`
  - `src/sparse_iterative.c`
  - `src/sparse_eigs.c`
- proof/support surfaces:
  - `tests/test_graph.c`
  - `tests/test_reorder_nd.c`
  - `tests/test_chol_csc.c`
  - `tests/test_ldlt_csc.c`
  - `tests/test_iterative.c`
  - `tests/test_eigs.c`
  - `tests/test_integration.c`
- likely narrow coordination headers only if the audit proves they need moving:
  - `include/sparse_analysis.h`
  - `include/sparse_reorder.h`
  - `include/sparse_iterative.h`
  - `include/sparse_eigs.h`

Measured hotspot sizes remain anchored to the Day 1 baseline:

- `src/sparse_graph.c` = `801`
- `src/sparse_graph_coarsen.c` = `641`
- `src/sparse_graph_bisect.c` = `528`
- `src/sparse_graph_refine.c` = `629`
- `src/sparse_reorder_nd.c` = `743`
- `src/sparse_reorder_amd_qg.c` = `611`
- `src/sparse_analysis.c` = `1020`
- `src/sparse_chol_csc.c` = `1532`
- `src/sparse_ldlt_csc.c` = `2130`
- `src/sparse_iterative.c` = `1985`
- `src/sparse_eigs.c` = `1534`
- `tests/test_graph.c` = `2900`
- `tests/test_reorder_nd.c` = `2262`
- `tests/test_chol_csc.c` = `4716`
- `tests/test_ldlt_csc.c` = `3680`
- `tests/test_iterative.c` = `2802`
- `tests/test_eigs.c` = `1522`

## Exit State

Sprint 67 Day 2 closes with:

- the same reviewed truthfulness baseline as the Sprint 66 close
- one explicit validation split for docs-only versus bounded code-touching
  versus substantial decomposition/build-alignment work
- one fixed rerun set centered on the actual decomposition-risk proof surface
- one clear starting point for the Day 3 residual hotspot audit
