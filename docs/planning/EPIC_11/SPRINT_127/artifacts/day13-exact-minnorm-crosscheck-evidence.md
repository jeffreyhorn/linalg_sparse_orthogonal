# Sprint 127 Day 13 Exact Minimum-Norm And Cross-Check Evidence

## Decision

Day 13 accepts one bounded underdetermined minimum-norm exact-value lane:
`qr_minnorm_3x6_exact_values`.

The accepted lane strengthens the existing `test_minnorm_3x6` owner-local
fixture in `tests/test_colamd.c`. It adds exact solution-value assertions, an
exact norm assertion, and a max-residual diagnostic for the already registered
3 x 6 underdetermined fixture.

Day 13 does not add a new QR-vs-SVD cross-check. The Sprint 125 2 x 4
`test_minnorm_vs_pinv` lane remains the only accepted QR-vs-SVD minimum-norm
cross-check.

## Accepted Fixture Contract

| Field | Value |
| --- | --- |
| Candidate key | `qr_minnorm_3x6_exact_values` |
| Owner | `tests/test_colamd.c::test_minnorm_3x6` |
| Shape | 3 x 6 |
| Matrix pattern | Row 0 has `A[0,0] = 2`, `A[0,3] = 1`; row 1 has `A[1,1] = 3`, `A[1,4] = 1`; row 2 has `A[2,2] = 1`, `A[2,5] = 2`. |
| RHS | `[3, 4, 5]` |
| Derivation | Each row solves independent `ca + db = rhs`; the minimum-norm pair is `rhs * [c, d] / (c^2 + d^2)`. |
| Expected solution | `[1.2, 1.2, 1.0, 0.6, 0.4, 2.0]` |
| Expected norm | `sqrt(8.4)` |
| Residual tolerance | `1e-10` per row |
| Value tolerance | `1e-10` |
| Norm tolerance | `1e-10` |

## Implemented Evidence

- Added the expected 6-entry solution vector to `test_minnorm_3x6`.
- Preserved the existing `A*x = b` residual assertions.
- Added max residual calculation and diagnostic printing.
- Added per-entry exact solution assertions.
- Added exact norm assertion against `sqrt(8.4)`.
- Did not add a generic helper, new external-reference fixture, SuiteSparse
  corpus fixture, optional-large fixture, or QR-vs-SVD comparison.

## Focused Validation

Focused validation passed:

```text
$ make build/test_colamd && ./build/test_colamd
minnorm 3x6: maxerr=1.78e-15, ||x||=2.8983
Tests run:    70
Tests failed: 0
Tests skipped: 0
Assertions:   317
Time:         4.333 s
ALL TESTS PASSED
```

## QR-vs-SVD Disposition

Additional QR-vs-SVD minimum-norm cross-checks remain deferred.

| Candidate | Disposition | Reason |
| --- | --- | --- |
| 3 x 6 QR-vs-SVD cross-check | Deferred | Exact closed-form values provide the Day 13 trust value without turning SVD pseudoinverse into a broader oracle. |
| 5 x 10 QR-vs-SVD cross-check | Deferred | Sprint 126 exact-value evidence already covers this fixture without adding SVD oracle language. |
| SuiteSparse QR-vs-SVD corpus cross-check | Deferred | Day 11 deferred SuiteSparse minimum-norm corpus expansion. |
| Generic QR-vs-SVD helper movement | Deferred | Would hide behavior ownership and tolerance policy. |

## Helper Movement Decision

Day 13 does not move helper ownership. Exact-value assertions remain at the
owner-local COLAMD minimum-norm call site so the fixture key, residual
tolerance, value tolerance, norm tolerance, and non-claim boundary stay
visible.

Future helper movement remains gated behind behavior-specific names and
focused validation for every owner executable whose behavior wording changes.

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
find docs/planning/EPIC_11/SPRINT_127 -type f -name '*.md' -print0 | \
  xargs -0 awk '(/[ \t]$/){print FILENAME ":" FNR ": trailing whitespace"; bad=1} END{exit bad}'
```

Full quality gate completed:

```text
$ make format && make lint && make test
All tests passed.
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 7 is complete or explicitly deferred. | Complete by bounded evidence plus deferral | Accepted 3 x 6 exact-value evidence; additional QR-vs-SVD cross-checks deferred. |
| Accepted evidence remains behavior-specific and bounded. | Complete | Assertions live in `test_minnorm_3x6` with fixture-local values, residuals, norm, and tolerances. |
| Broad SVD-pseudoinverse, helper API, and parity claims remain absent. | Complete | No new cross-check or helper was added; non-claims remain explicit. |
