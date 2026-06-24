# Sprint 86 Day 13: Full Validation Sweep

## Purpose

Run the full Sprint 86 validation queue fixed on Day 12 and capture the
measured close baseline from actual execution.

## Main Result

The full Day 13 queue passed cleanly:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

The maintained reviewed anchors stayed exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- reviewed CMake `ctest` = `53 / 53`
- reviewed CMake `Total Test time (real)` = `229.94 sec`

## Focused Reviewed Reruns

The focused reviewed proof owners all passed:

- `test_reorder` = `38 / 38`
- `test_reorder_nd` = `35 / 35` with `1` bounded skip
- `test_reorder_amd_qg` = `7 / 7`
- `test_graph` = `61 / 61`

Representative retained proof outputs:

- `test_reorder`:
  - `west0067`: AMD fill `819` vs natural `928`
  - `nos4`: AMD fill `1174` vs natural `1510`
- `test_reorder_nd`:
  - rerun time = `131.638 sec`
  - `Pres_Poisson`: AMD nnz(L) `2668793`, ND nnz(L) `2474435`
  - `bcsstk14`: AMD nnz(L) `116071`, ND nnz(L) `132634`
- `test_reorder_amd_qg`:
  - `bcsstk14`: wrapper nnz(L) = qg nnz(L) = `116071`
- `test_graph`:
  - `bcsstk14`: separator `97`, smoke time `45.2 ms`
  - `Pres_Poisson`: separator `216`, smoke time `277.4 ms`

## Examples and Benchmark Follow-Ons

Representative examples passed:

- `example_analysis`:
  - solve residual = `4.44e-16`
- `example_basic_solve`:
  - residual `||b - Ax|| = 0.00e+00`

Representative benchmark/reporting follow-ons passed:

- `make bench-reorder-sprint86` emitted:
  - `bcsstk14`: AMD `nnz_L=116071`, `reorder_ms=79.8`; ND `nnz_L=132634`,
    `reorder_ms=297.0`
  - `Pres_Poisson`: AMD `nnz_L=2668793`, `reorder_ms=5311.6`; ND
    `nnz_L=2474435`, `reorder_ms=3675.0`
- `make bench-canonical-report` wrote:
  - `bench_refactor_csc.csv`
  - `bench_chol_csc.csv`
  - `bench_iterative_reuse.csv`
  - `bench_eigs_reuse.csv`
  - `index.tsv`
  - `manifest.txt`

## Runtime Note

One non-blocking runtime note remains explicit:

- reviewed CMake `test_reorder_nd` remained the long tail at `135.01 sec`
  out of `229.94 sec`

The full reviewed path still completed cleanly, so this remains a closeout
runtime note rather than a Sprint 86 blocker.

## Exit State

- Sprint 86 now has a measured validated close baseline.
- The reviewed anchors stayed exact across the full sweep.
- Day 14 can close from execution evidence rather than implementation state.
