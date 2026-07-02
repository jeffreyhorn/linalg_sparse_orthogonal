# Sprint 102 Day 11 - General Solver Oracle Expansion Batch

## Scope

Day 11 implements the bounded general direct-solver oracle lane selected on
Day 10:

- solver family: linked-list LU;
- positive fixture: `lu_nonsym_square_5`;
- expected-failure fixture: `lu_singular_square_4`;
- external helper: `tests/lu_external_dense_reference.py`;
- C owner: `tests/test_sparse_lu.c`;
- tolerance: `1e-10` for dense-reference and known-solution comparison;
- pivot mode: `SPARSE_PIVOT_COMPLETE`.

The implementation does not change LU CSR, QR, SVD, direct CSC dispatch,
public headers, library sources, build files, or public documentation.

## Implementation

### External Dense Reference Helper

Added `tests/lu_external_dense_reference.py` with two deterministic fixtures:

- `lu_nonsym_square_5`: a nonsymmetric 5x5 square solve fixture;
- `lu_singular_square_4`: a rank-deficient 4x4 expected-failure fixture.

The helper uses the same output contract consumed by the shared C test helper:

```text
OK n
x[0]
x[1]
...
```

On singular input, the helper emits an `ERROR ...` line and exits nonzero.

### C Test Coverage

Updated `tests/test_sparse_lu.c` to:

- opt into the shared external-reference vector parser from
  `tests/test_solver_helpers.h`;
- build the `lu_nonsym_square_5` fixture in sparse linked-list form;
- build the `lu_singular_square_4` fixture in sparse linked-list form;
- compare linked-list LU against the external dense-reference solution;
- compare the linked-list LU result against the known `x_true = 1..5`
  solution;
- assert deterministic singular detection through `SPARSE_ERR_SINGULAR`.

The positive lane reports:

```text
external LU dense ref lu_nonsym_square_5: n=5, max|x-x_ref|=8.882e-16, residual=3.553e-15
```

The singular Python helper reports:

```text
ERROR matrix is singular to dense reference tolerance
```

## Validation

Focused helper validation:

- `python3 tests/lu_external_dense_reference.py lu_nonsym_square_5`: passed;
  emitted `OK 5` and recovered `x = 1..5` to roundoff.
- `python3 tests/lu_external_dense_reference.py lu_singular_square_4`: passed
  as an expected helper failure; emitted `ERROR matrix is singular to dense
  reference tolerance` and exited with status `1`.

Focused C validation:

- `make build/test_sparse_lu`: passed.
- `./build/test_sparse_lu`: passed; 39 tests, 0 failures, 0 skips, 144
  assertions.

Required code-touch quality gate:

- `make format`: passed.
- `make lint`: passed.
- `make test`: passed; `All tests passed.`

## Evidence Boundaries

Day 11 earns only these claims:

- linked-list LU has one deterministic nonsymmetric external dense-reference
  solve lane for `lu_nonsym_square_5`;
- linked-list LU has one deterministic singular expected-failure fixture for
  `lu_singular_square_4`;
- the shared external-reference parser can be reused by LU without changing
  its family-local fixture, factorization, tolerance, residual, or assertion
  ownership.

Day 11 does not claim:

- LU CSR external dense-reference coverage;
- QR or SVD external oracle coverage;
- broad dense-reference coverage for all LU fixtures;
- direct compressed-format LU APIs;
- portable performance superiority;
- public API or packaging changes.

## Closeout

The Day 10 LU boundary was implemented without tolerance relaxation and
without broadening Sprint 102 claims. The next work item should close out the
LU lane, compare the remaining QR/SVD evidence gaps against sprint capacity,
and decide whether another general direct-solver oracle lane is feasible.
