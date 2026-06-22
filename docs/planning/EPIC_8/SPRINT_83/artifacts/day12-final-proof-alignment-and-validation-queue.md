# Sprint 83 Day 12: Final Proof Alignment And Validation Queue

## Goal

Lock the final Sprint 83 proof-owner map and Day 13 validation queue after the
shared matrix/types/QR scalar-owner landings, without implying broader
capability maturity than the sprint actually delivered.

## Final Proof-Owner Map

- `tests/test_sparse_matrix.c`
  - shared matrix-shell scalar seam
  - shared width contract
- `tests/test_iterative.c`
  - iterative public scalar seam
- `tests/test_eigs.c`
  - eigensolver public scalar seam
- `tests/test_qr.c`
  - bounded QR public scalar seam
- `tests/test_svd.c`
  - deferred SVD family-local proof surface, not a Sprint 83 widened-owner
    proof target
- `tests/test_chol_csc.c`
  - direct-family Cholesky proof surface, not a Sprint 83 widened-owner proof
    target
- `tests/test_ldlt.c`
  - direct-family LDL^T proof surface, not a Sprint 83 widened-owner proof
    target
- `tests/test_integration.c`
  - retained cross-feature workflow owner for public direct and repeated-run
    behavior

## Representative Executable Support Map

- reviewed CMake regression owners:
  - `test_sparse_matrix`
  - `test_qr`
  - `test_svd`
  - `test_chol_csc`
  - `test_ldlt`
  - `test_integration`
- representative examples:
  - `example_analysis`
  - `example_basic_solve`
- benchmark/reporting owners:
  - `bench_svd`
  - `bench_refactor_csc`
  - `make bench-canonical-report`

## Support-Surface Outcome

No further support-only movement is needed before the full sweep:

- `README.md` already remains broadly truthful
- `docs/maintainer_guide.md` already matches the widened proof-owner split
- no public header outside the landed matrix/types/QR surfaces still misreads
  the Sprint 83 widened owner story in a way that requires correction now

## Day 13 Validation Queue

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- `./build/quality-review-cmake/test_sparse_matrix`
- `./build/quality-review-cmake/test_qr`
- `./build/quality-review-cmake/test_svd`
- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_ldlt`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_svd tests/data/suitesparse/nos4.mtx`
- `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `make bench-canonical-report`

## Explicit Out-Of-Scope Note

Install/export proof remains out of scope for Day 13 because Sprint 83 did not
move package, install, export, or runtime-package mechanics.
