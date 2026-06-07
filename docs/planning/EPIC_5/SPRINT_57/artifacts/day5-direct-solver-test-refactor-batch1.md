# Sprint 57 Day 5 - direct-solver test refactor batch 1

Date: 2026-06-06
Branch: `sprint-57`

## Scope

Land the first bounded direct-solver giant-test maintainability batch by
executing the Day 4 `test_chol_csc.c` helper-header extraction plan without
changing test-binary topology, proof meaning, or fixture coverage.

## Landed changes

### New local helper seam

Created:

- `tests/test_chol_csc_supernodal_helpers.h`

Moved into it:

- `detect_supernodes_alloc(...)`
- `day8_count_supernodes(...)`
- `day9_assert_batched_matches_scalar(...)`
- `day11_build_spd(...)`

These helpers are family-local to the Cholesky CSC supernodal / writeback /
dispatch proof cluster, so moving them out of the giant test file creates a
real ownership seam without pretending they are generic repo-wide test
utilities.

### Retained main-file boundary

Kept in `tests/test_chol_csc.c`:

- all test bodies
- all existing `RUN_TEST(...)` ordering
- `main()`
- the supernodal, writeback, and dispatch proof groups themselves

The only structural additions were:

- a forward declaration for `day8_chol_csc_match(...)`
- the include for the new helper header

This kept the binary shape stable while removing the moved helper bodies from
the main file.

## Explicit non-goals preserved

This batch did not:

- create a new test target
- edit `Makefile`
- edit `CMakeLists.txt`
- widen `tests/test_solver_helpers.h`
- move logic into `tests/test_ldlt_csc.c`
- move logic into `tests/test_integration.c`
- change residual thresholds, fixture coverage, or dispatch/lifecycle proof
  meaning

That matches the Day 4 fence exactly: ownership/readability first, topology
change later only if justified.

## Outcome

Measured touched-file sizes after landing:

- `tests/test_chol_csc.c` = `4552` lines
- `tests/test_chol_csc_supernodal_helpers.h` = `96` lines

Maintainability result:

- the largest direct-solver giant test now has a concrete owned helper seam
- the seam is aligned with a real proof-family boundary rather than a
  mechanical extract-small-functions cleanup
- the batch reduces local clutter without scattering the strongest CSC proof
  story across multiple binaries or generic helper layers

## Validation

Required code-day gate:

- `make format`
- `make lint`
- `make test`

All passed.

Stronger reviewed baseline:

- `make quality-review-full`

Passed.

Maintained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 266.18 sec`

Focused touched-surface reruns:

- `./build/test_chol_csc` -> `137 / 137`
- `./build/test_integration` -> `37 / 37`
- `./build/test_cholesky` -> `21 / 21`
- `./build/example_analysis` -> residual `4.44e-16`
- `./build/bench_refactor_csc tests/data/suitesparse/nos4.mtx --repeat 1`
  - `speedup_refactor = 1.11x`
  - `res_public = 8.24e-16`
  - `res_csc = 7.06e-16`

## Conclusion

Day 5 successfully landed the first Sprint 57 direct-solver giant-test batch:

- extracted the selected Cholesky CSC supernodal helper seam
- kept the `test_chol_csc` binary, runner, and proof bodies stable
- preserved all existing direct-solver caller-story and fixture truthfulness
- validated the batch through both the required code-day gate and the stronger
  reviewed baseline

This leaves Sprint 57 with a real first maintainability win and a cleaner
starting point for the next direct-solver or cross-family giant-test step.
