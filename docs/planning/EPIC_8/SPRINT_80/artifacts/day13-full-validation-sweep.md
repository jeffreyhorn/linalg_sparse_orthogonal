# Sprint 80 Day 13: Full Validation Sweep

## Purpose

Execute the full Sprint 80 validation queue and retain one explicit measured
baseline for the Epic 8 review-and-contract package.

## Full Validation Baseline

The code-day validation baseline completed cleanly:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 642.39 sec`

## Focused Follow-On Proof

The highest-signal reviewed proof-owner reruns also all passed:

- `./build/quality-review-cmake/test_chol_csc` -> `147 / 147`
- `./build/quality-review-cmake/test_ldlt_csc` -> `96 / 96`
- `./build/quality-review-cmake/test_ldlt` -> `84 / 84`
- `./build/quality-review-cmake/test_iterative` -> `80 / 80`
- `./build/quality-review-cmake/test_qr` -> `72 / 72`
- `./build/quality-review-cmake/test_integration` -> `51 / 51`
- `./build/quality-review-cmake/test_reorder_nd` -> `35 / 35`
- `./build/quality-review-cmake/test_fuzz` -> `26 / 26`
- `./build/quality-review-cmake/test_eigs` -> `31 / 31`

Representative retained outputs stayed stable:

- `test_fuzz` retained `large-n LDLT CSC lifecycle property: 3/3 passed`
- `test_chol_csc` retained
  `tests/data/suitesparse/bcsstk14.mtx: n=1806, rel_residual=1.080e-15`
- `test_reorder_nd` retained `Pres_Poisson ND/AMD = 0.923`
- `test_reorder_nd` retained `bcsstk14 ND/AMD = 1.124`
- `example_analysis` residual stayed `4.44e-16`
- `example_basic_solve` residual stayed `0.00e+00`

## Canonical Benchmark / Reporting Follow-On

The maintained canonical report command completed successfully on the final
rerun:

- `make bench-canonical-report`

The bundle wrote:

- `bench_refactor_csc.csv`
- `bench_chol_csc.csv`
- `bench_iterative_reuse.csv`
- `bench_eigs_reuse.csv`
- `index.tsv`
- `manifest.txt`

Representative retained rows:

- `bench_refactor_csc nos4` retained `speedup_refactor = 1.71`
- `bench_chol_csc nos4` retained
  `csc_supernodal_panel_solver = batched_panel`
- `bench_iterative_reuse` retained `cg 1.00x`, `gmres 1.05x`, `minres 1.06x`
- `bench_eigs_reuse` retained `growm 1.00x`, `thick_restart 1.07x`,
  `lobpcg 1.01x`

One transient note is preserved explicitly:

- the first standalone `make bench-canonical-report` rerun hit a
  non-reproducing missing-output error at the `bench_eigs_reuse.csv` write
  step
- an immediate clean rerun from the same tree succeeded without source edits
- the successful rerun is therefore the authoritative Day 13 close state

## Install / Export Proof

The maintained install/export regressions also passed:

- `bash tests/test_install.sh` -> `11 / 11`
- `bash tests/test_cmake_install.sh` -> `13 / 13`

Retained install/package anchor:

- installed `pkg-config` version remained `2.2.0`

## Runtime Note

One non-blocking runtime note remains explicit:

- reviewed CMake `test_reorder_nd` still dominated runtime at `486.38 sec`
  out of the `642.39 sec` total

This does not block Sprint 80 closeout because the full reviewed path
completed cleanly and all maintained parity anchors stayed exact.

## Day 13 Exit State

Sprint 80 now closes from one explicit validated baseline across reviewed
tests, focused proof-owner reruns, canonical benchmark reporting, and
install/export proof.
