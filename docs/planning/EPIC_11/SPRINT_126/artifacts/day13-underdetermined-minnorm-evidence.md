# Sprint 126 Day 13 Underdetermined Minimum-Norm Evidence

## Decision

Day 13 accepts one bounded underdetermined minimum-norm exact-value lane:
`qr_minnorm_5x10_exact_values`.

The accepted lane strengthens the existing `test_minnorm_5x10` owner-local
fixture in `tests/test_colamd.c`. It adds exact solution-value assertions,
an exact norm assertion, and a max-residual diagnostic for the already
registered 5 x 10 underdetermined fixture.

Day 13 does not add a new QR-vs-SVD cross-check. The Sprint 125 2 x 4
`test_minnorm_vs_pinv` lane remains the only accepted QR-vs-SVD
minimum-norm cross-check.

## Accepted Fixture Contract

| Field | Value |
| --- | --- |
| Candidate key | `qr_minnorm_5x10_exact_values` |
| Owner | `tests/test_colamd.c::test_minnorm_5x10` |
| Shape | 5 x 10 |
| Matrix pattern | For each row `i`, `A[i, i] = 2` and `A[i, i + 5] = 1`. |
| RHS | `[1, 2, 3, 4, 5]` |
| Derivation | Each row solves independent `2a + b = rhs_i`; minimum norm is `[a, b] = [2*rhs_i/5, rhs_i/5]`. |
| Expected solution | `[0.4, 0.8, 1.2, 1.6, 2.0, 0.2, 0.4, 0.6, 0.8, 1.0]` |
| Expected norm | `sqrt(11)` |
| Residual tolerance | `1e-10` per row |
| Value tolerance | `1e-10` |
| Norm tolerance | `1e-10` |

## Implemented Evidence

- Added the expected 10-entry solution vector to `test_minnorm_5x10`.
- Preserved the existing `A*x = b` residual assertions.
- Added max residual calculation and diagnostic printing.
- Added per-entry exact solution assertions.
- Added exact norm assertion against `sqrt(11)`.
- Did not add a generic helper, new external-reference fixture, SuiteSparse
  corpus fixture, or QR-vs-SVD comparison.

## Focused Validation

Focused validation passed:

```text
$ make build/test_colamd && ./build/test_colamd
minnorm 5x10: maxerr=8.88e-16, ||x||=3.3166
Tests run:    70
Tests failed: 0
Tests skipped: 0
Assertions:   310
ALL TESTS PASSED
```

## QR-vs-SVD Disposition

Additional QR-vs-SVD minimum-norm cross-checks remain deferred.

| Candidate | Disposition | Reason |
| --- | --- | --- |
| 5 x 10 QR-vs-SVD cross-check | Deferred | Exact closed-form values provide the Day 13 trust value without turning SVD pseudoinverse into a broader oracle. |
| 3 x 6 QR-vs-SVD cross-check | Deferred | Exact-value promotion metadata is not complete for this fixture. |
| SuiteSparse QR-vs-SVD corpus cross-check | Deferred | Day 11 deferred SuiteSparse minimum-norm corpus expansion. |
| Generic QR-vs-SVD helper movement | Deferred | Would hide behavior ownership and tolerance policy. |

## Non-Claims Preserved

- No new QR-vs-SVD cross-check is accepted in Day 13.
- No SVD pseudoinverse as a global QR oracle.
- No broad QR minimum-norm optimality claim.
- No SuiteSparse, optional-large, rank-deficient corpus, platform, or
  performance claim.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, external package, COLAMD,
  fallback, refinement, nullspace, Q-basis, economy, sparse-mode, reorder,
  backend, package, ABI, public API, CI, CMake, CTest, scalability, memory, or
  state-of-the-art parity claim.

## Validation

Day 13 changed C test code. Required validation:

```text
make build/test_colamd && ./build/test_colamd
make format && make lint && make test
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_126 tests/test_colamd.c
```

Full quality gate completed:

```text
$ make format && make lint && make test
All tests passed.
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 7 is complete or explicitly deferred. | Complete by bounded evidence plus deferral | Accepted 5 x 10 exact-value evidence; additional QR-vs-SVD cross-checks deferred. |
| Accepted cross-checks are fixture-keyed and bounded. | Complete | No new cross-check accepted; existing 2 x 4 baseline remains bounded. |
| Broad SVD-pseudoinverse, dense-library, and external parity claims remain absent. | Complete | See QR-vs-SVD disposition and non-claims. |
