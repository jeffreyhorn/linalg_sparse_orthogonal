# Sprint 122 Day 9 Minimum-Norm Helper Ownership Review

## Purpose

Day 9 revisits minimum-norm helper ownership after Sprint 121 identified the
area as useful but too semantic to migrate casually. The review covers QR solve,
SVD pseudoinverse, COLAMD/reordering, refinement, and SuiteSparse submatrix
minimum-norm proof owners.

## Decision

Do not migrate minimum-norm helpers in Sprint 122.

The accepted decision is an explicit deferral to a future QR solve /
minimum-norm consolidation owner. The current tests encode distinct behavior
contracts that would be easy to hide behind generic helpers: QR minimum-norm
solve semantics, SVD pseudoinverse equivalence, COLAMD/reordering behavior,
refinement residual behavior, rank-deficient consistency, fallback to regular
least-squares for `m >= n`, and SuiteSparse submatrix smoke coverage.

## Current Ownership Inventory

| Surface | Current Owner | Behavior Owned |
| --- | --- | --- |
| `test_qr_solve_minnorm_underdetermined_known_solution` | `tests/test_qr_solve.c` | QR solve executable owns one visible 2x4 underdetermined minimum-norm scenario and exact solution/norm checks. |
| `test_minnorm_*` family | `tests/test_colamd.c` | Broad minimum-norm solve, COLAMD option, rank-deficient, square/fallback, zero-row, and refinement behavior. |
| `test_minnorm_vs_pinv` | `tests/test_colamd.c` | Cross-checks QR minimum-norm output against `sparse_pinv` on a bounded fixture. |
| `test_minnorm_ss_submatrix` | `tests/test_colamd.c` | Optional SuiteSparse underdetermined submatrix smoke with norm comparison against a known feasible vector. |
| `test_pinv_underdetermined_minnorm_solution` | `tests/test_svd.c` | SVD pseudoinverse owns underdetermined minimum-norm solution behavior independently of QR. |
| `tf_svd_pinv_first_moore_penrose_error` | `tests/test_svd_helpers.h` | SVD helper owns Moore-Penrose reconstruction measurement, not QR minimum-norm semantics. |
| `tf_qr_*` helpers | `tests/test_qr_helpers.h` | QR helper layer owns reusable QR fixtures, exact RHS, reconstruction, and residual utilities, but not a minimum-norm semantic wrapper. |

## Duplicate and Risk Table

| Pattern | Duplication | Migration Risk |
| --- | --- | --- |
| 2x4 underdetermined fixture | Appears in QR solve, COLAMD minimum-norm, and SVD pseudoinverse tests with small pattern differences. | A shared builder is plausible, but moving it now could hide which executable owns QR solve versus SVD pseudoinverse behavior. |
| Norm calculation | Local `vec_norm2` calls appear across QR, COLAMD, and SVD tests. | Low mechanical risk, but it does not by itself solve ownership; extracting only this helper adds little value. |
| `A*x ~= b` residual checks | Repeated in minimum-norm tests with fixture-local tolerances. | Medium risk: exact, rank-deficient, SuiteSparse, and refinement cases use different tolerance meanings. |
| Minimum-norm comparison against alternate feasible solution | Used in `test_minnorm_is_minimal` and SuiteSparse submatrix proof. | Medium risk: helper must preserve proof text explaining why the alternate vector is feasible. |
| QR-vs-pseudoinverse equivalence | Encoded in `test_minnorm_vs_pinv`; related SVD proof lives in `test_svd.c`. | High risk: a shared helper could imply QR and SVD are one oracle surface rather than a bounded cross-solver check. |
| COLAMD/reorder option behavior | `test_minnorm_with_colamd` keeps option-specific ownership visible. | High risk: a generic minimum-norm helper would obscure the reorder-path claim. |
| Refinement residual non-increase | `test_refine_minnorm` and null-arg refinement checks own refinement-specific semantics. | High risk: not a pure solve helper; it has iteration and residual-after behavior. |

## Migration Boundary if Accepted Later

A future consolidation owner can safely migrate only after it writes a smaller
scope contract first:

1. Create a QR-owned minimum-norm fixture header only if every helper name
   includes the behavior being asserted, such as `tf_qr_minnorm_make_2x4_split`
   or `tf_qr_minnorm_assert_exact_split_solution`.
2. Keep SVD pseudoinverse helpers under the SVD helper owner unless the helper
   name explicitly says it is a cross-check fixture, not an oracle merger.
3. Keep COLAMD/reorder, rank-deficient, refinement, and SuiteSparse smoke tests
   as scenario owners even if they reuse a fixture builder.
4. Do not introduce a generic `assert_minnorm_solution` helper unless callers
   still pass tolerance, residual interpretation, expected norm, expected
   vector, option path, and diagnostic label explicitly.
5. Validate any accepted migration with `make format && make lint && make test`
   because it will touch `.c` and likely `.h` files.

## Deferral Rationale

Minimum-norm proof ownership is not a single helper problem. It is a set of
behavior claims spread across QR, COLAMD/reordering, SVD pseudoinverse,
refinement, and optional corpus coverage. Sprint 122 has already added bounded
external-reference lanes for SVD, QR least-squares, and partial SVD. Moving
minimum-norm helpers today would create churn without adding a new oracle lane
or reducing a concrete bug risk.

The correct Day 9 outcome is therefore to preserve current scenario ownership,
record the migration rules above, and hand the work to a future QR solve /
minimum-norm consolidation owner.

## Affected Surfaces

| Surface | Day 9 Action |
| --- | --- |
| `tests/test_qr_solve.c` | No code movement; remains QR solve owner for the visible 2x4 minimum-norm scenario. |
| `tests/test_colamd.c` | No code movement; remains broad minimum-norm, COLAMD, refinement, and SuiteSparse owner. |
| `tests/test_svd.c` | No code movement; remains SVD pseudoinverse owner for underdetermined minimum-norm behavior. |
| `tests/test_qr_helpers.h` | No new helper until a future migration names QR minimum-norm semantics explicitly. |
| `tests/test_svd_helpers.h` | No ownership change; pseudoinverse helpers remain SVD-scoped. |
| Makefile / CMake / CTest | No change. |
| Production source | No change. |
| Public docs / API | No change. |

## Residual Owner

Future owner: QR solve / minimum-norm consolidation.

Required inputs before implementation:

- fixture list separating exact, rank-deficient, reorder, fallback,
  refinement, and SuiteSparse cases;
- helper naming that preserves scenario ownership;
- tolerance table for exact residual, rank-deficient residual, norm comparison,
  cross-solver comparison, and refinement non-increase;
- validation plan that includes the affected QR, COLAMD, SVD, and full quality
  checks.

## Validation Notes

Day 9 changed documentation only. Required validation is `git diff --check` and
a focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_122`.
The branch passed `make format`, `make lint`, and `make test` after the Day 8
C/header changes; Day 9 does not add new code changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Minimum-norm helper ownership has a concrete decision. | Complete | Migration is explicitly deferred to a future QR solve / minimum-norm consolidation owner. |
| No helper movement hides QR solve or pseudoinverse semantics. | Complete | No helper movement was performed; QR, COLAMD, and SVD owners remain visible. |
| Deferred work has an owner. | Complete | Residual owner and implementation inputs are recorded above. |
