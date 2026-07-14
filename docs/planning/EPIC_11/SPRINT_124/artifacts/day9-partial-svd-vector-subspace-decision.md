# Day 9 Partial-SVD Vector and Subspace Decision

## Decision

Accept one bounded partial-SVD vector-residual lane and defer subspace,
repeated-spectrum, clustered-spectrum, rank-deficient subspace, convergence
budget, SuiteSparse corpus, and low-rank optimality lanes.

The accepted lane is `partial_svd_vector_residual_diag6_k2`. It reuses the
existing pure-Python external dense-reference singular-value fixture
`partial_svd_diag6_k2`, requests partial-SVD vectors from the product
implementation, and validates vector behavior through residuals and
orthogonality rather than raw vector equality.

## Accepted Evidence

| Field | Decision |
| --- | --- |
| Fixture key | `partial_svd_vector_residual_diag6_k2` |
| Dense-reference key | `partial_svd_diag6_k2` |
| Matrix | 6x6 diagonal matrix with values `9.0`, `6.0`, `3.0`, `1.0`, `0.5`, `0.25` |
| `k` | 2 |
| Product owner | `tests/test_svd_partial_helpers.h` |
| External helper owner | `tests/svd_external_dense_reference.py` value output only |
| Test registration owner | `tests/test_svd.c` |
| Metrics | external singular-value agreement, `A v_i - sigma_i u_i`, `A^T u_i - sigma_i v_i`, `U^T U - I`, and `V^T V - I` |
| Tolerances | `1e-8` for singular values, residuals, and orthogonality on this exact diagonal fixture |
| Skip policy | missing `python3` keeps the existing external-helper skip behavior; Windows keeps the existing explicit external-helper skip |
| Failure interpretation | helper `ERROR` is a reference/protocol failure; singular-value mismatch is bounded value regression; residual or orthogonality mismatch is bounded vector-residual regression |

## Why This Lane Is Acceptable

- It uses sign-invariant metrics. A sign flip in either singular-vector family
  does not change `A v_i - sigma_i u_i`, `A^T u_i - sigma_i v_i`, or
  orthogonality pass/fail meaning.
- It does not compare raw vector components against an external basis.
- It keeps external dense-reference scope narrow: the helper still emits only
  singular values, and the product test owns vector residuals.
- It avoids repeated or clustered singular values, so per-triplet residual
  interpretation is stable.
- It does not claim broad vector, subspace, convergence, or low-rank parity.

## Deferred Subspace and Residual Lanes

| Deferred lane | Reason | Future owner and promotion gate |
| --- | --- | --- |
| Repeated-spectrum subspace fixture | Individual vectors are basis-ambiguous inside the repeated leading subspace | Future subspace owner must add projector or principal-angle helper output and define pass/fail tolerances. |
| Clustered-spectrum subspace fixture | Near-tie ordering and convergence interpretation are not yet owned | Future convergence/subspace owner must define spectral gap, iteration budget, and projector tolerance. |
| Rank-deficient subspace fixture | Numerical-rank threshold and zero-space semantics are not externalized | Future rank/subspace owner must define rank threshold, zero tolerance, and left/right projector metrics. |
| Rectangular vector-residual fixture | Square exact lane should land first to prove helper/test protocol | Future vector owner may add tall/wide residual lanes after this fixture is stable. |
| SuiteSparse vector-residual corpus lane | Optional corpus availability and matrix conditioning need separate failure interpretation | Future corpus owner must state skip rules, residual windows, and corpus-specific diagnostics. |
| Low-rank optimality fixture | Reconstruction error and optimality claims are separate from top-k vector residuals | Future low-rank owner must define Frobenius/2-norm metric and sparse-output semantics. |
| Convergence-budget fixture | Timing smoke is not convergence proof | Future convergence owner must define options, iteration cap, residual tolerance, randomization policy if any, and failure meaning. |

## Maintainer Evidence Update

The maintainer guide now names the bounded partial-SVD vector-residual fixture
separately from bounded external singular-value fixtures. The wording preserves
the broad vector/subspace non-claim because this lane proves only one exact
diagonal vector-residual scenario.

## Non-Claim Register

Day 9 preserves the following non-claims:

- no LAPACK, SciPy, NumPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  vendor-backend parity claim;
- no broad SVD or partial-SVD external parity claim;
- no broad singular-vector parity claim beyond the single bounded diagonal
  vector-residual fixture;
- no subspace, repeated-spectrum, clustered-spectrum, or rank-deficient
  subspace parity claim;
- no convergence-budget guarantee;
- no low-rank global optimality claim;
- no package, ABI, platform, performance, scalability, public API, or
  state-of-the-art claim.

## Validation Plan

Because Day 9 changes `.c`/`.h` test files, validation requires:

1. `python3 tests/svd_external_dense_reference.py partial_svd_diag6_k2`
2. `make build/test_svd && ./build/test_svd`
3. `make format && make lint && make test`

All must pass before Day 9 is considered complete.
