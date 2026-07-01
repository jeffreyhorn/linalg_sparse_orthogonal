# Sprint 102 Day 13 - Validation and Evidence Reconciliation

## Scope

Day 13 reconciles Sprint 102 implementation, documentation, and validation
evidence before closeout. Because Sprint 102 modified `.c` and `.h` files,
the required full quality chain was:

```sh
make format && make lint && make test
```

No CMake, workflow, packaging, or generated API HTML surfaces were changed in
Sprint 102, so no additional CMake-specific validation lane was required for
Day 13.

## Focused Validation

| command | result | evidence |
|---|---|---|
| `python3 tests/ldlt_external_dense_reference.py ldlt_kkt_scaled_10` | passed | emitted `OK 10` and recovered `x = 1..10` to roundoff |
| `python3 tests/lu_external_dense_reference.py lu_nonsym_square_5` | passed | emitted `OK 5` and recovered `x = 1..5` to roundoff |
| `python3 tests/lu_external_dense_reference.py lu_singular_square_4` | passed as expected failure | emitted `ERROR matrix is singular to dense reference tolerance` and exited nonzero |
| `make build/test_ldlt_csc build/test_sparse_lu` | passed | focused binaries were up to date |
| `./build/test_ldlt_csc` | passed | 99 tests, 0 failures, 0 skips, 2318 assertions |
| `./build/test_sparse_lu` | passed | 39 tests, 0 failures, 0 skips, 144 assertions |

Focused direct-solver metrics:

- `ldlt_kkt_scaled_10`: `max|x - x_ref| = 8.882e-15`,
  `rel_residual = 1.692e-17`.
- `lu_nonsym_square_5`: `max|x - x_ref| = 8.882e-16`,
  `residual = 3.553e-15`.
- `lu_singular_square_4`: `SPARSE_ERR_SINGULAR` from the C LU factorization
  path and nonzero `ERROR ...` status from the helper path.

## Full Quality Gate

| command | result |
|---|---|
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed; final output: `All tests passed.` |

The full suite also re-exercised the touched direct-solver lanes:

- `test_sparse_lu`: 39 tests, 0 failures, 0 skips, 144 assertions.
- `test_chol_csc`: 92 tests, 0 failures, 0 skips, 20844 assertions.
- `test_ldlt_csc`: 99 tests, 0 failures, 0 skips, 2318 assertions.

## Evidence Reconciliation Against Day 3 Taxonomy

| fixture | taxonomy class | owner | expected state | observed state |
|---|---|---|---|---|
| `ldlt_kkt_scaled_10` | `indef-kkt-scaled` | `tests/test_ldlt_csc.c` plus `tests/ldlt_external_dense_reference.py` | LDLT CSC success with external dense-reference solve | passed at roundoff-level error and residual |
| `lu_nonsym_square_5` | `nonsym-square-small` | `tests/test_sparse_lu.c` plus `tests/lu_external_dense_reference.py` | linked-list LU success with external dense-reference solve | passed at roundoff-level error and residual |
| `lu_singular_square_4` | `square-rank-def` | `tests/test_sparse_lu.c` plus `tests/lu_external_dense_reference.py` | linked-list LU expected singular failure | passed with `SPARSE_ERR_SINGULAR` and helper `ERROR` status |

All new fixtures are deterministic, cheap, family-local, and have declared
success or expected-failure semantics. No Sprint 102 fixture uses random input,
local timing, or benchmark output as correctness evidence.

## Evidence Reconciliation Against Sprint 100 Template

| field | LDLT CSC scaled KKT lane | linked-list LU lane |
|---|---|---|
| comparison family | direct solver | direct solver |
| solver or algorithm path | LDLT CSC solve | linked-list LU factor/solve |
| artifact owner | Sprint 102 Day 8 artifact | Sprint 102 Day 11 artifact |
| implementation owner | `tests/test_ldlt_csc.c`; `tests/ldlt_external_dense_reference.py` | `tests/test_sparse_lu.c`; `tests/lu_external_dense_reference.py` |
| test owner | `tests/test_ldlt_csc.c` | `tests/test_sparse_lu.c` |
| external oracle owner | `tests/ldlt_external_dense_reference.py` | `tests/lu_external_dense_reference.py` |
| benchmark owner | none | none |
| validation command | helper command, focused binary, full quality gate | helper commands, focused binary, full quality gate |
| claim state after work | earned for named deterministic fixture only | earned for named deterministic solve and singular expected failure only |

Bounded claims evaluated:

- LDLT CSC solves the deterministic `ldlt_kkt_scaled_10` fixture consistently
  with the external dense-reference helper at the declared tolerance.
- Linked-list LU solves the deterministic `lu_nonsym_square_5` fixture
  consistently with the external dense-reference helper at the declared
  tolerance.
- Linked-list LU reports deterministic singular failure on
  `lu_singular_square_4`.

Disallowed broader claims:

- full external oracle coverage for every direct solver family;
- LU CSR external oracle coverage;
- QR or SVD external oracle coverage;
- direct public CSR/CSC solver APIs;
- portable performance superiority;
- broad ecosystem parity with mature sparse linear algebra packages.

## Earned, Deferred, and Non-Claim State

| state | item | evidence or reason |
|---|---|---|
| earned | shared external-reference vector parser | `tests/test_solver_helpers.h` reused by Cholesky CSC, LDLT CSC, and linked-list LU focused lanes |
| earned | LDLT CSC scaled KKT external-reference coverage | `ldlt_kkt_scaled_10` helper and C test pass with `1e-10` tolerance retained |
| earned | linked-list LU nonsymmetric external-reference coverage | `lu_nonsym_square_5` helper and C test pass with `1e-10` tolerance retained |
| earned | linked-list LU singular expected-failure coverage | `lu_singular_square_4` returns `SPARSE_ERR_SINGULAR` in C and helper `ERROR` status |
| earned | public direct-solver guidance refresh | README, tutorial, and maintainer guide updated with bounded trust wording |
| deferred | QR external dense least-squares or rank oracle lane | ranked after LU; no Sprint 102 implementation capacity used on QR |
| deferred | SVD external dense oracle lane | lower priority and heavier oracle design; no Sprint 102 implementation |
| deferred | LU CSR external dense-reference coverage | intentionally excluded from first LU oracle lane |
| non-claim | direct compressed solver APIs | Sprint 102 did not add public direct CSR/CSC solve APIs |
| non-claim | broad solver superiority or ecosystem parity | no benchmark sentinel or external ecosystem comparison was added |
| non-claim | complete direct-family external oracle coverage | Cholesky, LDLT, and LU lanes remain bounded to named fixtures |

## Sprint 103 Dependency Notes

Sprint 103 should treat these as explicit inputs:

- Use the maintainer-guide Sprint 102 trust-boundary table before writing any
  public comparison or capability claim.
- If QR is selected next, define the exact dense least-squares or rank oracle
  before implementation; likely fixture classes are `tall-full-rank` and
  `rect-rank-def`.
- If SVD is selected next, define whether the oracle checks singular values,
  rank, reconstruction, or pseudoinverse behavior before implementation.
- If LU CSR is selected next, keep it separate from the linked-list LU lane and
  name CSR-specific fixture, tolerance, and helper behavior.
- Do not promote helper availability skips, local benchmark output, or examples
  into correctness evidence.

## Closeout

All required Day 13 checks passed. Sprint 102 direct-solver claims are now tied
to named tests, fixtures, helpers, and validation commands, and Sprint 103
dependencies are explicit.
