# Sprint 123 Day 7 QR Minimum-Norm and Q/Economy Evidence Decision

## Purpose

Day 7 decides whether Sprint 123 should implement underdetermined
minimum-norm or Q/economy external QR evidence alongside the Day 6 compatible
least-squares fixture. The decision must preserve QR, COLAMD,
SVD-pseudoinverse, refinement, fallback, SuiteSparse, basis, sign, and economy
shape ownership instead of hiding those semantics behind a generic external
reference helper.

This is a decision artifact only. No C source, header, Python helper, build,
CMake, CTest, workflow, public API, or public wording changes are made by Day
7.

## Inputs Reviewed

| Input | Decision Use |
| --- | --- |
| Sprint 123 Plan Day 7 | Requires minimum-norm and Q/economy decisions, sign/basis semantics, helper ownership constraints, and residual handoff. |
| Sprint 123 Day 5 QR requirements | Defines underdetermined/minimum-norm and Q/economy candidates plus basis and ownership rules. |
| Sprint 123 Day 6 QR decision | Accepts `qr_overdetermined_compatible_5x3` for Day 8 and defers rank-deficient QR external evidence. |
| Sprint 122 Day 9 minimum-norm helper ownership review | Defers minimum-norm helper migration and defines future helper naming and ownership rules. |
| Sprint 121 Day 5 helper extraction plan | Identifies minimum-norm helper movement as deferred and keeps tolerances at scenario owners. |
| `include/sparse_qr.h` | Defines QR economy mode, `sparse_qr_form_q`, `sparse_qr_apply_q`, and `sparse_qr_solve_minnorm` semantics. |
| `tests/test_qr.c` | Owns Q application, Q orthogonality, economy shape, economy solve, sparse-mode, and reconstruction evidence. |
| `tests/test_qr_solve.c` | Owns one visible 2x4 QR minimum-norm solve scenario. |
| `tests/test_colamd.c` | Owns broad minimum-norm, COLAMD, fallback, refinement, rank-deficient, and SuiteSparse submatrix scenarios. |
| `tests/test_svd.c` | Owns SVD pseudoinverse and Moore-Penrose minimum-norm cross-check evidence. |

## Decision Summary

No Day 7 minimum-norm or Q/economy external QR evidence is accepted for Day 8
implementation.

Day 8 should implement only the Day 6 accepted compatible tall
least-squares fixture:

`qr_overdetermined_compatible_5x3`

Underdetermined minimum-norm and Q/economy external evidence are explicitly
deferred because each requires a behavior-specific policy that is not safe to
bolt onto the current full-rank least-squares helper.

## Minimum-Norm Decision

Minimum-norm external QR evidence is deferred.

| Candidate | Disposition | Future Owner | Promotion Gate |
| --- | --- | --- | --- |
| `qr_underdetermined_minnorm_3x5` | Deferred | Future QR solve / minimum-norm oracle owner | Define expected solution, norm comparator, residual metric, fallback behavior, and whether COLAMD/reorder options are in scope. |
| external 2x4 known minimum-norm fixture | Deferred | Future minimum-norm helper owner | Avoid duplicating the existing deterministic 2x4 QR/SVD/COLAMD evidence unless the external reference adds a distinct trust boundary. |
| QR-vs-SVD-pseudoinverse external minnorm check | Deferred | Future QR/SVD cross-solver owner | Define whether SVD pseudoinverse is an oracle, cross-check, or independent behavior owner. |
| rank-deficient minimum-norm external lane | Deferred | Future rank-deficient/minimum-norm owner | Separate residual-only, rank-threshold, nullspace, pseudoinverse, and minimum-norm claims before implementation. |

The current minimum-norm evidence is intentionally distributed:

- `tests/test_qr_solve.c` owns a visible 2x4 QR solve scenario;
- `tests/test_colamd.c` owns broad minimum-norm, COLAMD, fallback, refinement,
  rank-deficient, and SuiteSparse submatrix scenarios;
- `tests/test_svd.c` owns pseudoinverse and Moore-Penrose behavior.

A new external helper would need to preserve all of those owner boundaries.
Day 7 therefore defers implementation rather than hiding ownership behind a
generic `minnorm` fixture.

## Q/Economy Decision

Q/economy external evidence is deferred.

| Candidate | Disposition | Future Owner | Promotion Gate |
| --- | --- | --- | --- |
| `qr_economy_q_shape_5x3` | Deferred | Future QR basis/economy owner | Define whether evidence is shape-only, projection residual, orthogonality, or basis comparison. |
| external Q column comparison | Rejected for Sprint 123 Day 8 | Future QR basis owner | Requires sign, orientation, column ordering, and repeated/degenerate basis policy. |
| economy projection external check | Deferred | Future QR basis/economy owner | Define projection metric such as `||Q Q^T b - projection_ref||` without comparing raw basis columns. |
| economy/full solve external check | Deferred | Future QR economy owner | Existing deterministic full-vs-economy solve checks already cover the local behavior; external evidence must add a distinct reference. |

The current `tests/test_qr.c` coverage already owns:

- full Q round-trip and application;
- tall and wide Q orthogonality;
- economy solve equivalence;
- thin-Q orthogonality;
- economy R shape;
- square, wide, rank-deficient, 1x1, and SuiteSparse economy behavior;
- sparse-mode dense-vs-sparse agreement.

Externalizing raw Q basis comparisons would be brittle because Householder QR
bases can differ by sign and, in rank-deficient or repeated subspace cases, by
valid basis rotation. Day 7 therefore defers Q/economy external evidence until
a future owner defines projection or subspace metrics instead of raw column
equality.

## Required Basis and Shape Policy for Future Work

| Topic | Required Future Rule |
| --- | --- |
| Sign | If raw vectors are compared, each column must have an explicit sign-normalization policy. |
| Orientation | If comparing Q columns, define whether columns are ordered and unique for the fixture. |
| Degeneracy | Repeated, rank-deficient, or near-rank-deficient subspaces must use subspace/projection metrics, not raw basis equality. |
| Economy shape | State whether Q is full `m x m`, thin `m x n`, or wide-case full `m x m`, and how `sparse_qr_form_q` lays it out. |
| Projection metric | Prefer projection or orthogonality residuals over column equality when basis is not unique. |
| Backend boundary | Sparse-mode or dense-mode comparisons must remain backend-specific and must not imply performance or implementation parity. |

## Helper Ownership Constraints

| Surface | Constraint |
| --- | --- |
| `tests/test_qr_solve.c` | May own focused QR solve external fixtures, including Day 8 compatible tall evidence. |
| `tests/test_qr.c` | Continues owning Q basis, Q application, economy, sparse-mode, rank, nullspace, and reconstruction behavior. |
| `tests/test_colamd.c` | Continues owning broad minimum-norm, COLAMD/reorder, fallback, refinement, and SuiteSparse submatrix behavior. |
| `tests/test_svd.c` | Continues owning SVD pseudoinverse and Moore-Penrose behavior. |
| `tests/test_qr_helpers.h` | May provide fixture builders or measurements only when helper names preserve behavior semantics and callers keep tolerances visible. |
| `tests/qr_external_dense_reference.py` | Should remain a bounded least-squares helper unless a future owner designs a separate basis/minimum-norm protocol. |

## Day 8 Scope

Day 8 should:

1. Implement `qr_overdetermined_compatible_5x3` from Day 6, or publish a
   deferral if implementation uncovers a fixture/reference issue.
2. Not implement minimum-norm external evidence.
3. Not implement Q/economy external evidence.
4. Preserve `test_qr` and `test_colamd` ownership boundaries.
5. Run focused helper and `test_qr_solve` checks plus the full C quality gate
   if `.c`, `.h`, or script files change.

## Non-Claim Register

Day 7 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, or broad
  external dense-library parity;
- broad QR factorization or least-squares parity;
- direct-solver parity;
- rank-deficient QR external parity;
- underdetermined or minimum-norm global optimality;
- QR/SVD pseudoinverse equivalence as an oracle;
- Q-basis, Q-sign, Q-orientation, economy-mode, sparse-mode, reorder, or
  backend parity;
- package, ABI, platform, public API, CMake, Makefile, CI, or CTest expansion;
- performance, scalability, memory behavior, or state-of-the-art behavior.

## Validation Notes

Day 7 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_123`

The branch already contains Day 4 `.c` and Python helper changes; Day 4 ran
the full `make format && make lint && make test` gate after those changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 2 is complete or explicitly deferred. | Complete for Day 7 decision scope | Minimum-norm and Q/economy external lanes are explicitly deferred; Day 8 scope is limited to Day 6 compatible QR evidence. |
| Minimum-norm and Q/economy semantics are not hidden in generic helpers. | Complete | Helper ownership constraints preserve QR, COLAMD, SVD, refinement, fallback, SuiteSparse, basis, and economy owners. |
| Future implementation has behavior-specific proof gates. | Complete | Minimum-norm and Q/economy deferral tables define future owners and promotion gates. |
