# Sprint 121 Day 9 - Least-Squares and Pseudoinverse Expansion

## Purpose

Day 9 expanded bounded least-squares, minimum-norm, and pseudoinverse
evidence using deterministic fixtures. The changes stay in test code and do
not make external-library parity or broad numerical-optimality claims.

## Code Surfaces

- `tests/test_qr_solve.c`
- `tests/test_svd.c`

No Makefile, CMake, workflow, package, benchmark, public API, or production
source surfaces were changed.

## QR Least-Squares Additions

### Compatible Tall Least-Squares

`test_qr_solve_overdetermined_compatible_tall` uses a deterministic 4x2
full-column-rank matrix and an exact generated right-hand side.

Assertions:

- QR solve returns the known solution `{2.0, -1.0}`.
- Reported residual is below `1e-10`.
- Independent true residual helper reports residual below `1e-10`.

### Incompatible Tall Least-Squares

`test_qr_solve_overdetermined_incompatible_known_residual` uses the same 4x2
matrix but adds an exact vector orthogonal to the column space.

Assertions:

- QR solve preserves the known least-squares solution `{2.0, -1.0}`.
- Reported residual is exactly the known residual norm `sqrt(3)`.
- Reported residual normalized by `||b||_2` matches the independent true
  residual helper.

### Underdetermined Minimum-Norm

`test_qr_solve_minnorm_underdetermined_known_solution` uses a deterministic
2x4 system with two independent row constraints.

Assertions:

- `sparse_qr_solve_minnorm` returns `{0.5, 0.5, 0.5, 0.5}`.
- The returned solution satisfies `A*x = b`.
- The solution norm is `1.0`.

## SVD Pseudoinverse Addition

`test_pinv_underdetermined_minnorm_solution` uses the same 2x4 fixture through
`sparse_pinv`.

Assertions:

- `A*A^+*A ~= A` with max error below `1e-10`.
- `A^+*b` returns `{0.5, 0.5, 0.5, 0.5}`.
- The induced solution satisfies `A*x = b`.
- The induced solution norm is `1.0`.

## Non-Claims

- These tests validate deterministic internal contracts only.
- They do not claim LAPACK/SuiteSparse parity.
- They do not claim global optimality across arbitrary ill-conditioned
  least-squares or pseudoinverse systems.
- They intentionally keep residual and solution expectations fixture-local.

## Focused Validation

Command:

```sh
make build/test_qr_solve build/test_svd && ./build/test_qr_solve && ./build/test_svd
```

Result:

- `test_qr_solve`: 13 tests, 0 failures, 0 skips, 1014 assertions.
- `test_svd`: 101 tests, 0 failures, 0 skips, 1616 assertions.

## Deferred Queue

- Day 10 should expand bounded low-rank and partial-SVD evidence.
- Dense-reference comparison lanes remain queued for Days 11-12.
- Minimum-norm helper extraction from `tests/test_colamd.c` remains deferred
  unless future work moves those ownership tests into a dedicated QR solve
  helper surface.
