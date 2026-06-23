# Sprint 84 Day 12: Final Proof Alignment And Validation Queue

## Goal

Lock the final Sprint 84 proof-owner map, CI/policy reading, and Day 13
validation queue after the bounded direct-family differential, seeded-property,
and failure-path assurance landings.

## Final Proof-Owner Map

- `tests/test_chol_csc.c`
  - bounded direct-family maintained external differential lane
  - fixture-backed Cholesky CSC SPD external comparison on `nos4` and
    `bcsstk04`
- `tests/test_fuzz.c`
  - bounded seeded generative lifecycle/property lane
  - large-`n` CSC-backed Cholesky and LDL^T reorder/repeat/refactor/residual
    properties
- `tests/test_integration.c`
  - public lifecycle oracle surface
  - cancellation, preservation, rejection, repeated-run, and retry-after-
    failure guarantees
- `tests/test_ldlt.c`
  - family-local LDL^T direct proof surface, not a Sprint 84 external-
    differential center
- `tests/test_iterative.c`
  - retained iterative proof owner, not a Sprint 84 adopted external-
    differential center
- `tests/test_eigs.c`
  - retained eigensolver proof owner, not a Sprint 84 adopted external-
    differential center

## Representative Executable Support Map

- reviewed CMake proof owners:
  - `test_chol_csc`
  - `test_ldlt`
  - `test_fuzz`
  - `test_integration`
  - `test_iterative`
  - `test_eigs`
- representative examples:
  - `example_analysis`
  - `example_basic_solve`
- benchmark/reporting owners:
  - `bench_refactor_csc`
  - `bench_svd`
  - `make bench-canonical-report`

## Support-Surface Outcome

No further support-only movement is needed before the full sweep:

- `docs/maintainer_guide.md` already remains truthful about the bounded
  direct-family external differential lane, the seeded property lane, and the
  public lifecycle oracle lane
- `README.md` already remains truthful about the same proof-owner split and
  the Windows `test_fuzz` exclusion caveat
- `.github/workflows/windows-ci.yml` already remains truthful that `test_fuzz`
  stays outside the reviewed Windows subset

## CI / Policy Truth Map

- Linux and macOS local/reviewed paths still exercise `test_fuzz`
- Windows still excludes `test_fuzz` from the reviewed CMake subset
- Sprint 84 therefore widens local and Linux/macOS assurance depth without
  creating new reviewed-Windows evidence claims

## Day 13 Validation Queue

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`
- `ctest -N --test-dir build/quality-review-cmake`
- `./build/quality-review-cmake/test_chol_csc`
- `./build/quality-review-cmake/test_ldlt`
- `./build/quality-review-cmake/test_fuzz`
- `./build/quality-review-cmake/test_integration`
- `./build/quality-review-cmake/test_iterative`
- `./build/quality-review-cmake/test_eigs`
- `./build/quality-review-cmake/example_analysis`
- `./build/quality-review-cmake/example_basic_solve`
- `./build/quality-review-cmake/bench_svd tests/data/suitesparse/nos4.mtx`
- `./build/quality-review-cmake/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
- `make bench-canonical-report`

## Explicit Out-Of-Scope Note

Install/export proof remains out of scope for Day 13 because Sprint 84 did not
move package, install, export, or runtime-package mechanics.
