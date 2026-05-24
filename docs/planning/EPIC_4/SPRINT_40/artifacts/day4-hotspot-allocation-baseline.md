# Sprint 40 Day 4 Artifact: Hotspot & Allocation-Density Baseline

## Purpose

Measure the structural hotspots the later Epic 4 refactors are intended to
target.

## Largest File Concentrations

### `src/`

- `src/sparse_graph.c` = `3555` lines
- `src/sparse_eigs.c` = `3143` lines
- `src/sparse_ldlt_csc.c` = `2723` lines
- `src/sparse_iterative.c` = `2349` lines
- `src/sparse_chol_csc.c` = `2208` lines
- `src/sparse_svd.c` = `1726` lines
- `src/sparse_lu_csr.c` = `1666` lines
- `src/sparse_qr.c` = `1577` lines

### `tests/`

- `tests/test_chol_csc.c` = `4643` lines
- `tests/test_svd.c` = `3712` lines
- `tests/test_ldlt_csc.c` = `3637` lines
- `tests/test_qr.c` = `3259` lines
- `tests/test_etree.c` = `2890` lines
- `tests/test_iterative.c` = `2795` lines
- `tests/test_ldlt.c` = `2774` lines
- `tests/test_graph.c` = `2628` lines

### `benchmarks/`

- `benchmarks/bench_eigs.c` = `958` lines
- `benchmarks/bench_main.c` = `774` lines
- `benchmarks/bench_ldlt_csc.c` = `516` lines
- `benchmarks/bench_convergence.c` = `421` lines
- `benchmarks/bench_chol_csc.c` = `393` lines

### `scripts/`

- `scripts/deadcode_report.py` = `523` lines
- `scripts/epic3_warning_workflow.sh` = `215` lines
- `scripts/deadcode_workflow.sh` = `189` lines
- `scripts/wall_check.sh` = `162` lines

## Allocation-Density Signal

This pass used raw `malloc` / `calloc` / `realloc` / `free` occurrence counts
as a simple baseline signal.

### `src/`

- `src/sparse_graph.c` = `208`
- `src/sparse_svd.c` = `182`
- `src/sparse_ldlt_csc.c` = `156`
- `src/sparse_iterative.c` = `156`
- `src/sparse_ldlt.c` = `148`
- `src/sparse_etree.c` = `144`
- `src/sparse_eigs.c` = `141`
- `src/sparse_qr.c` = `139`

### `tests/`

- `tests/test_chol_csc.c` = `451`
- `tests/test_qr.c` = `424`
- `tests/test_ilu.c` = `398`
- `tests/test_iterative.c` = `387`
- `tests/test_etree.c` = `360`
- `tests/test_svd.c` = `339`
- `tests/test_ldlt_csc.c` = `269`
- `tests/test_minres.c` = `266`

### `benchmarks/`

- `benchmarks/bench_main.c` = `75`
- `benchmarks/bench_convergence.c` = `50`
- `benchmarks/bench_refactor_csc.c` = `44`
- `benchmarks/bench_ldlt_csc.c` = `44`

## Day 4 Interpretation

### First-tier architecture hotspots

The strongest measured architectural hotspots are:

- `src/sparse_graph.c`
- `src/sparse_eigs.c`
- `src/sparse_iterative.c`
- `src/sparse_ldlt_csc.c`
- `src/sparse_svd.c`

### Large test files: real risk, but not all immediate split targets

The test surface clearly has large helper-dense files, but many of the biggest
tests are still coherent feature-owner binaries. The best near-term
maintainability strategy is helper/fixture consolidation first, not automatic
whole-file splitting.

### Benchmark hotspots

The strongest current benchmark maintainability targets are:

- `benchmarks/bench_main.c`
- `benchmarks/bench_eigs.c`

Sprint 37’s helper extraction appears to have successfully reduced the relative
urgency of the backend-comparison pair.

### Script hotspots

The strongest support-script maintainability targets are:

- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`

## Day 4 Conclusion

Day 4 confirms that Epic 4’s later refactor queue is structurally grounded:

- graph decomposition is the top source-level hotspot
- repeated-workspace numeric internals are the next major cluster
- large tests need helper extraction more than blind splitting
- benchmarks are concentrated in `bench_main.c` and `bench_eigs.c`
- dead-code support scripts remain real maintainability surfaces
