# Day 11 Minimum-Norm Helper Migration Decision

## Purpose

Decide whether Sprint 123 should migrate minimum-norm test helpers out of scenario owners into shared helpers. The decision must preserve visible behavior ownership for QR solve, COLAMD/reordering, SVD pseudoinverse, refinement, fallback, rank-deficient, and SuiteSparse submatrix cases.

## Decision Summary

Minimum-norm helper migration is explicitly deferred.

The current duplicate code is not the hard part. The risky part is that the assertions around each minimum-norm scenario carry different behavior claims:

- direct QR minimum-norm solution shape and exact known values;
- COLAMD/reordering behavior;
- overdetermined fallback to ordinary least squares;
- rank-deficient and zero-row semantics;
- refinement residual behavior;
- SuiteSparse submatrix smoke behavior;
- QR-vs-SVD pseudoinverse cross-check behavior;
- SVD Moore-Penrose and pseudoinverse-owned minimum-norm behavior.

Moving those assertions into generic helper wrappers today would make the tests shorter but would hide the behavior-specific tolerance and ownership boundaries that Sprint 121 and Sprint 122 deliberately kept visible.

## Ownership Inventory

| Surface | Owned scenarios | Current ownership should remain |
| --- | --- | --- |
| `tests/test_qr_solve.c` | Focused 2x4 QR minimum-norm solve with exact solution, residual, and norm checks | Yes. This is the visible QR solve scenario owner. |
| `tests/test_colamd.c` | Null args, known 2x4, minimality comparison, 3x6, overdetermined fallback, COLAMD reorder, 5x10, rank-deficient, square fallback, 1xn, refinement, zero-row, QR-vs-pinv, refine-null, west0067 submatrix | Yes. This is the broad minimum-norm, reorder, fallback, refinement, and SuiteSparse owner. |
| `tests/test_svd.c` | `sparse_pinv` Moore-Penrose checks, rectangular pseudoinverse, underdetermined pseudoinverse minimum-norm solution | Yes. This is the SVD pseudoinverse and Moore-Penrose owner. |
| `tests/test_qr_helpers.h` | QR fixture builders and QR measurement helpers | May host future builders or measurements, but should not own minimum-norm assertions yet. |
| `tests/test_svd_helpers.h` | SVD matrix builders and pseudoinverse measurements | May host measurement helpers, but SVD pseudoinverse assertions must remain visible in `test_svd.c`. |
| `tests/test_solver_helpers.h` | External-reference helper plumbing and generic test utilities | Should not absorb minimum-norm behavior semantics. |

## Duplicate Opportunities

| Duplicate | Possible helper name | Safe today? | Reason |
| --- | --- | --- | --- |
| `vec_norm2` appears in multiple test surfaces | `tf_vec_norm2` or owner-specific `tf_qr_minnorm_vec_norm2` | Not in Sprint 123 Day 11 | Generic vector norm extraction is mechanically safe, but the benefit is too small without also moving scenario assertions. |
| Build `Ax` then compare to `b` | `tf_qr_minnorm_residual_inf` | Defer | Residual tolerances differ by scenario and must remain at the caller. |
| 2x4 known minimum-norm matrix builder | `tf_qr_minnorm_make_split_2x4` | Defer | The 2x4 fixture appears in QR, COLAMD, and SVD contexts, but each context proves a different behavior. |
| Compare candidate solution against expected `[0.5, 0.5, 0.5, 0.5]` | `tf_assert_minnorm_split_2x4_solution` | No | Assertion helper would hide whether the owner is QR solve, COLAMD, or SVD pseudoinverse. |
| QR-vs-SVD pseudoinverse multiplication | `tf_minnorm_apply_pinv_to_rhs` | Defer | This is cross-solver evidence and should stay explicit until a dedicated QR/SVD minimum-norm owner exists. |
| SuiteSparse submatrix setup | `tf_minnorm_make_west0067_submatrix` | Defer | Loader, skip, shape, and corpus assumptions are scenario-local. |

## Semantic Risks

| Risk | Impact |
| --- | --- |
| Generic `minnorm` helpers blur QR and SVD ownership | A failure could be harder to assign to QR solve, SVD pseudoinverse, or a shared helper. |
| Assertion wrappers hide tolerances | Tests would lose the visible `1e-12`, `1e-10`, or `1e-8` scenario thresholds that explain what each fixture proves. |
| Reorder and fallback behavior become invisible | COLAMD, square/overdetermined fallback, and refinement cases are not just minimum-norm arithmetic checks. |
| SuiteSparse skip and corpus assumptions become generic | A shared helper could make a corpus smoke look like a broad minimum-norm guarantee. |
| QR-vs-pseudoinverse cross-check becomes an oracle claim | The existing test is a cross-check; moving it into helper language risks implying SVD is the external oracle for QR. |

## Future Helper Naming Policy

If a future dedicated minimum-norm owner migrates helpers, helper names should encode the behavior under test:

- `tf_qr_minnorm_residual_inf`
- `tf_qr_minnorm_solution_norm2`
- `tf_qr_minnorm_make_split_2x4`
- `tf_qr_minnorm_make_rankdef_2x4`
- `tf_qr_minnorm_make_zero_row_2x4`
- `tf_qr_minnorm_apply_pinv_rhs`
- `tf_qr_minnorm_make_west0067_submatrix`

Avoid generic names such as:

- `assert_minnorm`
- `check_minnorm`
- `minnorm_oracle`
- `minnorm_parity`
- `validated_minnorm`

## Required Future Promotion Gates

A future helper migration can proceed only if it satisfies all of these gates:

1. Each migrated helper has a behavior-specific name.
2. Tolerance values remain at the scenario call site or are passed explicitly.
3. Assertions remain scenario-local unless the helper name encodes the exact scenario.
4. QR solve, COLAMD/reordering, SVD pseudoinverse, refinement, fallback, rank-deficient, and SuiteSparse ownership remain separately visible.
5. No helper name or artifact language treats SVD pseudoinverse as an external oracle for QR.
6. Focused validation runs at least:
   - `make build/test_qr_solve && ./build/test_qr_solve`
   - `make build/test_colamd && ./build/test_colamd`
   - `make build/test_svd && ./build/test_svd`
7. Because `.c` or `.h` files would change, final validation must run:
   - `make format && make lint && make test`

## Day 11 Outcome

No code was changed for Day 11. The migration is deferred with a behavior-specific naming policy and promotion gates.

This keeps minimum-norm scenario ownership visible while giving a future QR solve / minimum-norm consolidation owner enough structure to migrate small measurement helpers without turning scenario assertions into generic wrappers.

## Non-Claim Register

Day 11 does not claim:

- broad minimum-norm optimality beyond existing fixtures;
- QR/SVD pseudoinverse oracle parity;
- external dense-library parity for minimum-norm behavior;
- SuiteSparse-wide minimum-norm support;
- reorder, fallback, or refinement superiority;
- package, platform, ABI, public API, performance, or state-of-the-art behavior.

## Validation Notes

Day 11 changed documentation only. Required validation:

```text
git diff --check
rg -n "[ \t]$" docs/planning/EPIC_11/SPRINT_123
```

The branch already passed the full required code gate after Day 10's `.c`, `.h`, and helper-script changes:

```text
make format && make lint && make test
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 4 is complete or explicitly deferred. | Complete | Migration is explicitly deferred. |
| No minimum-norm scenario loses visible behavior ownership. | Complete | No code moved; ownership inventory is recorded. |
| Any helper movement has focused validation evidence. | Not applicable | No helper movement occurred on Day 11. |
