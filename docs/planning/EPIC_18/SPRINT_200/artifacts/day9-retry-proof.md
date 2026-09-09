# Sprint 200 Day 9: Retry Proof

## Scope

Day 9 extends the selected-owner allocation-failure proof for
`sparse_symbolic_lu()` by proving failure-then-success retry behavior. It uses
the same selected symbolic LU failure-site table from Days 7 and 8.

## Regression Coverage

Day 9 adds `test_symbolic_lu_allocation_failures_recover_on_retry` to
`tests/test_etree.c`.

For each selected symbolic LU allocation-failure site, the test:

1. initializes `sym_L` and `sym_U` with nonzero sentinel metadata and null
   buffers;
2. enables deterministic allocation failure at the selected `fail_after`
   position;
3. calls `sparse_symbolic_lu()`;
4. resets the allocation hook before assertions;
5. asserts `SPARSE_ERR_ALLOC`;
6. asserts the caller-owned unsymmetric 3x3 input matrix is unchanged;
7. asserts the caller-owned identity permutation is unchanged for permutation
   cases;
8. asserts failed outputs are empty and repeatedly free-safe;
9. reruns `sparse_symbolic_lu()` without injection using the same output
   objects;
10. asserts successful retry output is fresh, nonempty, monotone, and contains
    the numeric LU structure bound.

## Fresh Output Assertions

`assert_symbolic_lu_retry_output_fresh()` verifies retry publication after a
failed call:

- `sym_L.n` equals the input row count;
- `sym_U.n` equals the input column count;
- both symbolic objects have at least diagonal cardinality;
- `col_ptr` and `row_idx` are non-null for both outputs;
- terminal `col_ptr[n]` equals each output `nnz`;
- both column-pointer arrays are monotone;
- the symbolic bounds contain a numeric partial-pivot LU factorization of the
  same input matrix.

These checks prove retry success does not depend on stale failed-call sentinel
state.

## Input Preservation Assertions

The retry test reuses Day 7's input snapshots:

- matrix dimensions remain 3x3;
- expected fixture values at all inserted positions remain unchanged;
- permutation entries remain `{0, 1, 2}` for permutation-path failures.

The same preservation assertions are run after the failed call and after the
successful retry.

## Registration Guard

`tests/test_symbolic_allocation_failure_gate_registration.py` now requires:

- `RUN_TEST(test_symbolic_lu_allocation_failures_recover_on_retry);`
- `assert_symbolic_lu_retry_output_fresh(A, &sym_L, &sym_U);`

This keeps the Day 9 retry proof in the focused
`make symbolic-allocation-failure-gate` path.

## Validation

Commands run:

```sh
make format
make symbolic-allocation-failure-gate
make lint
make test
git diff --check
```

Results:

- `make format`: PASS.
- `make symbolic-allocation-failure-gate`: PASS; `test_etree` reported 104
  tests, 0 failures, 0 skipped, and 4316 assertions.
- `make lint`: PASS.
- `make test`: PASS.
- `git diff --check`: PASS.

Day 9 modified C test code, so the full C quality gate was run.

## Residual Non-Claims

Day 9 does not claim:

- allocation-failure retry behavior for every public API;
- retry coverage for matrix-construction setup allocations outside the selected
  symbolic LU owner;
- allocator balance accounting or sanitizer leak-detection coverage.
