# Sprint 87 Day 13: Full Validation Sweep

## Purpose

Run the full Sprint 87 validation queue fixed on Day 12 and capture the
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
- reviewed CMake `Total Test time (real)` = `299.15 sec`

## Focused Reviewed Reruns

The focused reviewed proof owners all passed:

- `test_reorder` = `38 / 38`
- `test_reorder_nd` = `35 / 35` with `1` bounded skip
- `test_reorder_amd_qg` = `7 / 7`
- `test_graph` = `61 / 61`

Representative retained proof outputs:

- `test_reorder_nd`:
  - rerun time = `117.359 sec`
  - `Pres_Poisson`: AMD nnz(L) `2668793`, ND nnz(L) `2474435`
  - `bcsstk14`: AMD nnz(L) `116071`, ND nnz(L) `132634`
- `test_reorder_amd_qg`:
  - `bcsstk14`: wrapper nnz(L) = qg nnz(L) = `116071`

## Examples and Package Proof Follow-Ons

Representative examples passed:

- `example_analysis`:
  - solve residual = `4.44e-16`
- `example_basic_solve`:
  - residual `||b - Ax|| = 0.00e+00`

The maintained package and consumer proof follow-ons also passed:

- `bash tests/test_install.sh`:
  - `13` passed, `0` failed
- `bash tests/test_cmake_install.sh`:
  - `15` passed, `0` failed

The retained maintained reporting follow-on also passed:

- `make bench-canonical-report` wrote:
  - `bench_refactor_csc.csv`
  - `bench_chol_csc.csv`
  - `bench_iterative_reuse.csv`
  - `bench_eigs_reuse.csv`
  - `index.tsv`
  - `manifest.txt`

## Runtime Note

One non-blocking runtime note remains explicit:

- reviewed CMake `test_reorder_nd` remained the long tail at `142.76 sec`
  out of `299.15 sec`

The full reviewed path still completed cleanly, so this remains a Sprint 87
close-baseline runtime note rather than a blocker.

## Exit State

- Sprint 87 now has a measured validated close baseline.
- The reviewed baseline, package proofs, and retained reporting follow-ons all
  stayed clean.
- Day 14 can close from execution evidence rather than implementation state.
