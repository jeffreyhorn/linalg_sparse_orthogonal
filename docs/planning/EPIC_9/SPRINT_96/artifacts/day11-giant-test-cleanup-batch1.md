# Sprint 96 Day 11: Giant-Test Cleanup Batch 1

## Purpose

Day 11 implements the first bounded proof-owner reduction designed on Day 10:
split the largest retained Cholesky CSC proof owner into a core CSC suite and a
supernodal/writeback suite.

## Implementation Summary

Created a new test owner:

- `tests/test_chol_csc_supernodal.c`

Updated existing owners:

- `tests/test_chol_csc.c`
- `tests/test_chol_csc_supernodal_helpers.h`
- `Makefile`
- `CMakeLists.txt`

The new `chol_csc_supernodal` suite owns:

- supernode detection
- supernodal postorder corpus-safety checks
- dense Cholesky primitive checks
- dense backend environment-contract checks
- dense LDLT factor cross-checks housed in the Cholesky CSC proof family
- supernode extract/writeback plumbing
- supernode diagonal-block factor checks
- panel/full batched path integration checks
- parametrised scalar/batched cross-checks
- CSC linked-list writeback roundtrips and rejection tests

The existing `chol_csc` suite retains:

- allocation and growth tests
- conversion roundtrip tests
- permutation-cache tests
- symbolic validation and edge-case tests
- workspace/elimination scaffold tests
- scalar kernel tests
- solve/residual/shim tests
- transparent dispatch tests
- external dense-reference tests

## Boundary Adjustments During Landing

The Day 10 design intentionally kept transparent dispatch in
`tests/test_chol_csc.c`. During the first focused build, one solve-group test
had moved with the supernodal block while its runner remained in the core file:

- `test_factor_with_analysis_large_n_matches_explicit_supernodal_route(...)`

That test was moved back into `tests/test_chol_csc.c`, where the
solve/residual/shim runner still owns it.

The dispatch-only deterministic SPD matrix builder was also kept local to
`tests/test_chol_csc.c`. The family-local helper header now serves only the
new supernodal/writeback proof owner and no longer needs to be included by the
core test file.

## Build Registration

Makefile:

- added `$(TESTDIR)/test_chol_csc_supernodal.c` to `TEST_SRCS` immediately
  after `$(TESTDIR)/test_chol_csc.c`

CMake:

- added `add_sparse_test(test_chol_csc_supernodal)` immediately after
  `add_sparse_test(test_chol_csc)`

No platform gate was added. Existing `_WIN32` behavior remains with the tests
that already owned it.

## Size Impact

Before the split:

- `tests/test_chol_csc.c`: 5029 lines

After the split:

- `tests/test_chol_csc.c`: 2619 lines after formatting
- `tests/test_chol_csc_supernodal.c`: 2493 lines after formatting
- `tests/test_chol_csc_supernodal_helpers.h`: 243 lines after formatting

The largest proof owner is now split into two explicit suites while preserving
the same assertion coverage.

## Validation

Focused checks passed:

```sh
make build/test_chol_csc build/test_chol_csc_supernodal
./build/test_chol_csc
./build/test_chol_csc_supernodal
```

Focused suite results:

- `test_chol_csc`: 92 tests passed
- `test_chol_csc_supernodal`: 60 tests passed

Required full code-day quality chain passed:

```sh
make format && make lint && make test
```

The full test run included the new `test_chol_csc_supernodal` executable and
finished with `All tests passed.`

## Stale-Reference Scans

Post-split scans confirmed:

- supernodal and writeback runner groups live in
  `tests/test_chol_csc_supernodal.c`
- transparent dispatch and external dense-reference tests remain in
  `tests/test_chol_csc.c`
- `Makefile` and `CMakeLists.txt` both register
  `test_chol_csc_supernodal`
- the Cholesky CSC family-local helper header no longer describes itself as
  belonging to the old monolithic `test_chol_csc.c` file

## Residual Queue

Remaining proof-owner cleanup candidates:

- `tests/test_ldlt_csc.c`
- `tests/test_integration.c`
- transparent dispatch in `tests/test_chol_csc.c`, only if a later sprint
  needs another Cholesky CSC proof-owner split

## Day 11 Exit State

The first giant-test cleanup batch is landed and validated. Sprint 96 can move
to the next proof-owner or closeout step with Cholesky CSC supernodal/writeback
coverage in its own executable and the core Cholesky CSC suite still owning
public dispatch behavior.
