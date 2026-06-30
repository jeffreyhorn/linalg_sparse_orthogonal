# Sprint 98 Day 5: External Correctness Expansion Batch 1

## Purpose

Day 5 lands the first implementation batch for the Sprint 98 LDLT CSC external
correctness lane frozen on Day 4.

## What Changed

Implemented a bounded external dense-reference solve comparison for
deterministic LDLT CSC KKT fixtures.

New helper:

- `tests/ldlt_external_dense_reference.py`

Updated test harness:

- `tests/test_ldlt_csc.c`

New tests:

- `test_s98_external_dense_reference_kkt_5x5`
- `test_s98_external_dense_reference_kkt_10x10`

## Helper Contract

The helper accepts fixture keys:

- `kkt5`
- `kkt10`

It constructs the dense fixture independently from project C code, builds the
known solution vector `x_true[i] = i + 1`, computes `b = A * x_true`, solves
the dense system with deterministic partial-pivoting Gaussian elimination, and
emits the reference solution.

The helper intentionally does not:

- call project C code
- mirror Bunch-Kaufman pivot internals
- inspect CSC storage
- depend on NumPy, SciPy, LAPACK, SuiteSparse, or platform packages
- emit runtime or fill data

## C Harness Behavior

The C harness:

1. builds the matching deterministic KKT fixture in `tests/test_ldlt_csc.c`
2. builds `x_true[i] = i + 1`
3. computes the original-order right-hand side
4. runs the existing LDLT CSC analysis-aware path through
   `s20_two_pass_indefinite_factor`
5. permutes the right-hand side into the factor's internal order
6. solves with `ldlt_csc_solve`
7. maps the solution back to original fixture order
8. reads the external dense-reference solution
9. checks solution agreement and residual strength

This keeps the maintained comparison on user-visible solve behavior rather
than factor internals, pivot arrays, permutation arrays, or CSC layout.

## Observed Focused Results

Focused helper runs:

```sh
python3 tests/ldlt_external_dense_reference.py kkt5
python3 tests/ldlt_external_dense_reference.py kkt10
python3 tests/ldlt_external_dense_reference.py nope
```

Observed behavior:

- `kkt5` emitted `OK 5` with solution `1, 2, 3, 4, 5`
- `kkt10` emitted `OK 10` with round-off-level solution values near
  `1..10`
- unknown fixture key failed with `ERROR unknown fixture nope`

Focused LDLT CSC run:

```sh
make build/test_ldlt_csc && ./build/test_ldlt_csc
```

Observed new-lane highlights:

- `test_s98_external_dense_reference_kkt_5x5`
  - `max|x-x_ref| = 0.000e+00`
  - `rel_residual = 0.000e+00`
- `test_s98_external_dense_reference_kkt_10x10`
  - `max|x-x_ref| = 3.553e-15`
  - `rel_residual = 2.292e-16`

The focused run passed:

- `98` tests run
- `0` failed
- `0` skipped

## Required Full Validation

Because Day 5 modified a C test file, the required source validation was run:

```sh
make format && make lint && make test
```

Result:

- passed
- final test output ended with `All tests passed.`

## Claim Boundary

Day 5 supports this bounded claim after validation:

- Sprint 98 now has an LDLT CSC external dense-reference solve comparison on
  deterministic KKT fixtures.

Day 5 does not support these claims:

- broad LDLT external proof across all indefinite matrices
- parity with another solver stack
- validation of every Bunch-Kaufman pivot shape
- runtime or fill superiority
- platform parity for the external helper lane

## Residual Risks

- The lane is intentionally small: `kkt5` and `kkt10` are deterministic proof
  fixtures, not a broad corpus.
- Windows keeps external helper skip behavior consistent with the existing
  Cholesky external-helper model.
- Day 6 should tighten names, comments, and maintainer proof-owner wording if
  needed, but should not widen fixtures without a new boundary note.

## Day 5 Result

The selected external correctness lane now has a working maintained comparison
path. LDLT CSC solves for `kkt5` and `kkt10` agree with an independent dense
reference helper at round-off-level tolerance, and the required full validation
chain passed.
