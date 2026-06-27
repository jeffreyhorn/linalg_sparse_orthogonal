# Sprint 93 Day 13: Full Validation Sweep

## Purpose

Validate the full Sprint 93 runtime/threading/ND package from the live branch
state after the Day 12 evidence batch, then freeze one exact close baseline
before Sprint 93 closeout.

## Main Result

The full Day 13 queue passed cleanly from the live branch state:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- `./build/quality-review-cmake/test_reorder_nd`
- `./build/quality-review-cmake/test_graph`
- `./build/quality-review-cmake/test_threads`
- `./build/quality-review-cmake/test_omp`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/bench_reorder --sprint86-slice --skip-factor`
- `./build/bench_reorder --sprint86-slice --skip-factor --reorder-via-analyze`
- `make bench-canonical-report`

## Reviewed Anchors

The reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- reviewed CMake `Total Test time (real)` = `286.93 sec`

## Focused Reruns

The touched reviewed-runtime and adjacent proof surfaces also passed cleanly:

- `test_reorder_nd`
  - `35 / 35`
  - `1` skip
  - `Time = 175.541 s`
- `test_graph`
  - `61 / 61`
  - `Time = 7.368 s`
- `test_threads`
  - `8 / 8`
  - `Time = 0.089 s`
- `test_omp`
  - `12 / 12`
  - `Time = 0.012 s`

Representative example reruns also stayed clean:

- `example_analysis`
  - residual = `4.44e-16`
- `example_basic_solve`
  - residual = `0.00e+00`

## Runtime Evidence

The bounded Sprint 93 runtime-evidence surfaces also passed cleanly:

- `./build/bench_reorder --sprint86-slice --skip-factor`
  - representative ND rows:
    - `bcsstk14,1806,nd,132634,422.7,skip,direct,sprint86,160`
    - `Pres_Poisson,14822,nd,2474435,5165.8,skip,direct,sprint86,160`
- `./build/bench_reorder --sprint86-slice --skip-factor --reorder-via-analyze`
  - representative ND rows:
    - `bcsstk14,1806,nd,132634,449.6,skip,analyze,sprint86,160`
    - `Pres_Poisson,14822,nd,2474435,5589.6,skip,analyze,sprint86,160`

Canonical reporting also passed:

- `make bench-canonical-report`
  - wrote the canonical bundle under `build/bench-reports/canonical`

## Residual Notes

The residual non-blocking runtime note stays explicit:

- reviewed `test_reorder_nd` remained the long pole at `169.17 sec` inside the
  reviewed CMake run and `175.541 s` in the explicit focused rerun
- the bounded Sprint 86 runtime slice remains mixed by matrix and entry path,
  not broad-claim oriented
- OpenMP proof still reads truthfully as the current serial build lane:
  - `test_omp` reported `OpenMP DISABLED (serial build)`

## Exit State

- Sprint 93 now has one validated live-branch close baseline.
- The reviewed runtime, touched proof owners, bounded runtime evidence, and
  canonical reporting surfaces all rechecked cleanly from the post-Day-12
  branch state.
- Day 14 can now close Sprint 93 from one exact validated baseline rather than
  from a partially rerun runtime lane.
