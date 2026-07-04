# Sprint 104 Day 5 Backend Descriptor Batch

## Purpose

Day 5 implements the first backend descriptor/status batch from the Day 4
boundary. The implementation stays behavior-preserving: it does not add public
API, does not change enum values or struct layout, and does not make optional
dense backends required. It adds focused tests that make the Day 3/4 fallback
contract explicit for invalid optional dense-backend requests.

## Implemented Scope

| surface | change | reason |
|---|---|---|
| Cholesky CSC dense backend tests | added `test_supernodal_dense_backend_invalid_env_falls_back_to_builtin` | proves unknown `SPARSE_CHOL_DENSE_BACKEND` values fall back to builtin descriptor |
| LDLT dense backend tests | added `test_ldlt_dense_backend_invalid_env_falls_back_to_builtin` | proves unknown `SPARSE_LDLT_DENSE_BACKEND` values report/use builtin |
| test runners | registered both new tests in their existing dense-backend groups | keeps fallback proof close to existing builtin/external/accelerate contract tests |

No public headers, library source behavior, benchmark output columns, or
examples were changed.

## Contract Proven

| contract point | proof owner |
|---|---|
| unknown Cholesky dense-backend env request is not a hard failure | `tests/test_chol_csc_supernodal.c` |
| unknown LDLT dense-backend env request is not a hard failure | `tests/test_ldlt.c` |
| builtin remains selected after invalid request | both new tests |
| env state is cleaned up after focused tests | both new tests call `tf_unsetenv(...)` after descriptor/status read |

## Compatibility Notes

- Existing zero-initialized option structs are unchanged.
- Existing enum values are unchanged.
- Existing ABI/layout is unchanged.
- Existing optional backend env names are unchanged.
- Existing silent fallback behavior is preserved and now explicitly tested for
  invalid env values.
- Windows remains covered by the same builtin fallback expectation because the
  invalid env tests do not require dynamic provider probing.

## Validation Plan

Because Day 5 touched `.c` test files, required validation is:

1. focused build of affected tests:
   - `make build/test_chol_csc_supernodal build/test_ldlt`
2. focused affected test execution:
   - `./build/test_chol_csc_supernodal`
   - `./build/test_ldlt`
3. full required quality chain:
   - `make format && make lint && make test`
4. final hygiene:
   - `git diff --check`
   - `rg -n "[ \t]+$" tests/test_chol_csc_supernodal.c tests/test_ldlt.c docs/planning/EPIC_10/SPRINT_104`

## Non-Claims

This batch does not claim:

- optional acceleration is available;
- optional acceleration is faster than builtin;
- invalid public backend enum handling changed;
- public vendor-backend selection exists;
- benchmark timing is portable;
- graph/ND runtime controls are covered by dense-backend descriptors.

## Day 6 Handoff

Day 6 should audit OpenMP, process-global env controls, thread-local graph/FM
overrides, and nested-parallelism behavior. The Day 5 fallback tests give Day 6
a stable dense-backend baseline: invalid optional dense-backend env values
remain builtin-fallback cases, not hard runtime errors.
