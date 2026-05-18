# Sprint 32 Day 13 Full Validation Sweep

**Date:** 2026-05-18  
**Branch:** `sprint-32`

## Objective

Re-run the authoritative warning reproduction and the full standard validation flow from the current branch state, then record the final Sprint 32 before/after evidence.

## Validation Commands

Authoritative warning reproduction:

- `cmake --build build/sprint32-day1-cmake --parallel 1 --clean-first`

Standard validation flow:

- `make format`
- `make lint`
- `make test`
- `ctest -N --test-dir build/sprint32-day1-cmake`
- `ctest --test-dir build/sprint32-day1-cmake --output-on-failure`

Truthfulness/coverage spot checks:

- grep across:
  - `tests/test_reorder_nd.c`
  - `tests/test_framework_optin.c`
  - `tests/test_eigs.c`
  - `tests/test_svd.c`
  - `tests/test_chol_csc.c`
- explicit active vs commented `RUN_TEST(...)` count in `tests/test_reorder_nd.c`

## Results

### Warning reproduction

The authoritative Apple Clang serialized CMake rebuild passed with:

- total warnings: `0`

That confirms the Day 12 zero-warning state is stable on a fresh rebuild and not an incremental-build artifact.

### Makefile path

All standard Makefile validation steps passed:

- `make format`
- `make lint`
- `make test`

Important Day 13 observations from that path:

- `make lint` still includes the compile-only tooling gate added in Sprint 31
- `make test` still exercises the active suite after the Sprint 32 truthfulness cleanup
- `test_framework_optin` still passes with the expected skip accounting:
  - `8` tests run
  - `0` failed
  - `3` skipped

### CTest path

The CMake/CTest side also remained healthy:

- `ctest -N --test-dir build/sprint32-day1-cmake`
  - `53` registered tests
- `ctest --test-dir build/sprint32-day1-cmake --output-on-failure`
  - `53 / 53` passed
  - total real time: `159.66 sec`

Selected long or high-signal CTest results:

- `test_reorder_nd`
  - passed in `72.65 sec`
- `test_fuzz`
  - passed in `23.89 sec`
- `test_lu_csr`
  - passed in `8.78 sec`
- `test_colamd`
  - passed in `5.37 sec`
- `test_chol_csc`
  - passed in `5.52 sec`
- `test_graph`
  - passed in `5.21 sec`
- `test_sprint19_integration`
  - passed in `5.35 sec`

## Truthfulness / Coverage Preservation

Day 13 explicitly checked that Sprint 32's truthfulness cleanup removed dormant scaffold without silently removing active protection.

### `test_reorder_nd.c`

- active `RUN_TEST(...)` calls: `23`
- commented-out `RUN_TEST(...)` calls: `0`

That matches the post-Day-5 truthful state:

- active protections remain active
- the deleted historical Pres_Poisson target stubs did not reappear

### `test_framework_optin.c`

The opt-in framework coverage remains live and executable:

- `SKIP_TEST(...)`
- `RUN_TEST_SLOW(...)`
- `RUN_TEST_EXPERIMENTAL(...)`

It is still registered in CTest and still passes in the full validation run.

### Broader suite registration

`ctest -N` still reports `53` registered tests, which means Sprint 32 did not accidentally shrink the runnable suite while cleaning up dormant inline scaffold.

## Final Warning Comparison

Relative to the Day 1 baseline:

- full-tree warnings: `98 -> 0`
- `src`: `0 -> 0`
- `tests`: `98 -> 0`
- `benchmarks`: `0 -> 0`
- `examples`: `0 -> 0`

By warning class:

- `-Wmissing-field-initializers`: `62 -> 0`
- `-Wdouble-promotion`: `33 -> 0`
- `-Wunused-function`: `3 -> 0`

Residual warning queue after Day 13:

- none

## Conclusion

Day 13 confirms that Sprint 32's cleanup is not just locally green but end-to-end validated:

- the authoritative warning reproduction stays at zero
- the Makefile validation path passes
- the CTest validation path passes
- active truthfulness/opt-in coverage remains present and tested

That leaves Day 14 with closeout and handoff work only; there is no remaining implicit warning debt or validation uncertainty to carry forward.
