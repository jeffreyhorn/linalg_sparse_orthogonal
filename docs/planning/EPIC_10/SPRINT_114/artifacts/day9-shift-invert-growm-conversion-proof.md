# Sprint 114 Day 9: Shift-Invert Grow-M Conversion Proof

## Purpose

Day 9 proves the shift-invert grow-m conversion path before any eigensolver
source ownership changes. The proof targets the public behavior of
`SPARSE_EIGS_NEAREST_SIGMA` when grow-m Lanczos runs on
`(A - sigma I)^{-1}` and publishes original-space eigenpairs through
`lambda = sigma + 1 / theta`.

## Fixture

| Field | Value |
|---|---|
| Matrix | Laplacian tridiagonal |
| Dimension | `n = 24` |
| Requested pairs | `k = 4` |
| Shift | `sigma = 1.37` |
| Backend | forced `SPARSE_EIGS_BACKEND_LANCZOS` |
| Iteration cap | `max_iterations = 24` |
| Tolerance | `tol = 1e-11` |
| Vectors | `compute_vectors = 1`, `reorthogonalize = 1` |

The expected original-space eigenvalues are kept visible in the test through
the closed-form Laplacian spectrum:

```text
lambda_p = 2 - 2 cos(p*pi/(n + 1))
```

For `sigma = 1.37`, the four nearest values are `p = 10, 9, 11, 8` in
descending transformed magnitude `abs(1 / (lambda_p - sigma))`.

## Implemented Proof

| File | Test | Public assertions |
|---|---|---|
| `tests/test_eigs.c` | `test_s114_shift_invert_growm_conversion_nearest_sigma` | `SPARSE_OK`, `n_requested == k`, `n_converged == k`, forced grow-m backend, `peak_basis_size == 24`, `iterations == 24`, residual within tolerance, one Lanczos progress callback, expected original-space values, transformed-theta magnitude order, original-space residuals, and vector orthonormality. |

## Public-Result Invariants

- Shift-invert reports original-space `lambda`, not transformed Ritz `theta`.
- `NEAREST_SIGMA` ordering follows descending `abs(theta)`, equivalent to
  nondecreasing `abs(lambda - sigma)` for nonsingular shifts.
- The forced backend remains grow-m Lanczos, so the proof belongs to the
  source path under consideration for Day 10.
- Vector publication remains original-space: returned vectors satisfy
  `A v = lambda v`, not only the shift-inverted operator equation.
- The grow-m basis and iteration fields remain visible through
  `peak_basis_size` and `iterations`.

## Movement Assessment

No source movement was performed. Day 9 completes the required shift-invert
grow-m conversion proof, but Day 10 still needs to review the combined Days
2-9 evidence before deciding whether one narrow eigensolver movement is safe.

## Validation

Focused validation passed:

```text
make build/test_eigs && ./build/test_eigs
```

The focused run executed `43` tests with `0` failures and `956` assertions.

## Completion Criteria

- Shift-invert grow-m conversion is directly proven through public results.
- Backend, basis, convergence, residual, and vector expectations are asserted.
- No public API, install-header, source-list, helper-target, Make, CMake, or
  reviewed CTest membership changes were introduced.
- Day 10 can make an evidence-backed movement/no-move decision.
