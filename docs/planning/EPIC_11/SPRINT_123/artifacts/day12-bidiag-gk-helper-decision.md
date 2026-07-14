# Day 12 Bidiagonal/Golub-Kahan Helper Decision

## Purpose

Decide whether Sprint 123 should extract Bidiagonal/Golub-Kahan helper code into shared helpers after the sprint added new SVD, QR, and partial-SVD external evidence. The decision must keep specialized wide-transpose, implicit Householder reconstruction, explicit `U`/`V` extraction, and bidiagonal QR iteration semantics visible.

## Decision Summary

Bidiagonal/Golub-Kahan helper extraction is explicitly deferred.

Sprint 123 should not move current Bidiagonal/Golub-Kahan helpers into the general SVD helper layer. Limited future extraction is acceptable only through a dedicated Bidiagonal/Golub-Kahan helper owner, preferably in a dedicated header such as `tests/test_bidiag_helpers.h`, and only when helper names preserve the exact reconstruction or iteration semantics.

No code was changed for Day 12.

## Current Ownership Inventory

| Surface | Current owner | Behavior owned |
| --- | --- | --- |
| `bidiag_reconstruction_error` | `tests/test_bidiag.c` | Reconstructs `A = U B V^T` from implicit Householder sequences and recursively handles internally transposed wide matrices. |
| `test_bidiag_*` family | `tests/test_bidiag.c` | Bidiagonal reduction shape, diagonal/superdiagonal behavior, wide transpose flag, single-row, diagonal, `nos4`, null, and free-zeroed behavior. |
| `gk_reconstruction_error` | `tests/test_svd.c` | Reconstructs from explicit extracted `U`, `V`, `diag`, and `superdiag` arrays for non-transposed GK checks. |
| `test_gk_extract_*` | `tests/test_svd.c` | Explicit `U`/`V` extraction for square and tall fixtures, plus wide-transpose reconstruction skip behavior. |
| `validate_gk` and `test_gk_*` | `tests/test_svd.c` | GK reconstruction and `U`/`V` column orthogonality with transposed-wide guardrails. |
| `test_bidiag_svd_*` | `tests/test_svd.c` | Bidiagonal QR iteration on raw bidiagonal arrays, analytical 2x2 singular values, UV accumulation, zero-superdiag deflation, and k=1 behavior. |
| `tests/test_svd_helpers.h` | Shared SVD measurement helpers | Owns full-SVD reconstruction, orthogonality, pseudoinverse, and low-rank measurement helpers, not Bidiagonal/GK internals. |
| `tests/test_svd_partial_helpers.h` | Partial-SVD scenario helpers/tests | Owns partial-SVD value/vector checks, not GK extraction or bidiagonal QR iteration. |

## Specialized Semantics To Preserve

| Semantic | Required boundary |
| --- | --- |
| Wide internal transpose | Tests must preserve `bd.transposed` interpretation instead of treating all bidiagonalizations as direct `A = U B V^T` reconstructions. |
| Implicit Householder reconstruction | `tests/test_bidiag.c` reconstructs from reflector vectors and beta values, not from dense SVD `U`/`Vt` results. |
| Explicit GK `U`/`V` extraction | `tests/test_svd.c` checks extracted `U` and `V` matrices with column-major layout and scenario-specific dimensions. |
| Wide GK reconstruction skip | Current `validate_gk` intentionally skips direct reconstruction for transposed wide cases and relies on orthogonality plus end-to-end wide SVD evidence. |
| Bidiagonal QR iteration | `bidiag_svd_iterate` tests operate on raw bidiagonal arrays and optional `U`/`V` accumulators, not sparse matrix SVD result objects. |
| Tolerance ownership | Reconstruction, orthogonality, analytical singular-value, `nos4`, and deflation checks use different tolerances that must remain visible at call sites. |

## Extraction Candidate Assessment

| Candidate | Decision | Reason |
| --- | --- | --- |
| Move `bidiag_reconstruction_error` to `tests/test_svd_helpers.h` | Reject for Sprint 123 | It would make a general SVD helper own `sparse_bidiag_t`, Householder vectors, and transposed-wide recursion. |
| Move `gk_reconstruction_error` to `tests/test_svd_helpers.h` | Reject for Sprint 123 | It is not full-SVD reconstruction; it assumes explicit extracted `U`/`V` and upper-bidiagonal arrays. |
| Extract `tests/test_bidiag_helpers.h` now | Defer | This could be valid future work, but it touches `.h`/`.c` files and should be done only with focused SVD/bidiag validation and a dedicated owner. |
| Share small square/tall/wide matrix builders | Defer | Safe only if names encode Bidiagonal/GK use and assertions remain scenario-local. |
| Share column orthogonality measurement | Already acceptable | Generic measurement is fine when callers keep tolerances and labels visible. |
| Move bidiagonal QR iteration assertions into helper wrappers | Reject | The QR iteration tests are algorithm-kernel proof owners; helper wrappers would hide iteration semantics. |
| Create generic `assert_svd_reconstructs` wrappers | Reject | This would hide layout, transpose, and tolerance differences between full SVD, GK extraction, and bidiagonal reduction. |

## Future Dedicated Helper Policy

A future Bidiagonal/GK maintainability owner may extract a dedicated helper header only if helper names encode the specific behavior:

- `tf_bidiag_reconstruction_error_from_reflectors`
- `tf_bidiag_reconstruction_error_with_transpose`
- `tf_gk_reconstruction_error_from_extracted_uv`
- `tf_gk_make_square_3x3_fixture`
- `tf_gk_make_tall_10x5_fixture`
- `tf_gk_make_wide_5x10_fixture`
- `tf_bidiag_qr_reconstruct_from_accumulated_uv`

Avoid generic names:

- `assert_svd_reconstruction`
- `check_bidiag`
- `gk_oracle`
- `bidiag_validated`
- `svd_internal_parity`

## Required Future Promotion Gates

Any future helper extraction must satisfy all of these gates:

1. A dedicated Bidiagonal/GK owner is named.
2. The helper header is Bidiagonal/GK-specific, not a general SVD helper expansion.
3. Tolerances remain at scenario call sites or are passed explicitly.
4. Wide-transpose behavior remains explicit in each caller.
5. `bidiag_svd_iterate` tests retain algorithm-kernel proof ownership.
6. Focused validation runs:
   - `make build/test_bidiag && ./build/test_bidiag`
   - `make build/test_svd && ./build/test_svd`
7. Because `.c` or `.h` files would change, final validation runs:
   - `make format && make lint && make test`
8. If Makefile, CMake, or CTest membership changes, source-list and CTest surface proof must be added before review.

## Residual Handoff

Carry this deferred item to a future Bidiagonal/GK maintainability owner:

> Extract only behavior-named Bidiagonal/GK measurement helpers into a dedicated helper header, preserving wide-transpose, implicit Householder reconstruction, explicit `U`/`V` extraction, bidiagonal QR iteration, and scenario-local tolerances.

## Non-Claim Register

Day 12 does not claim:

- broad SVD helper consolidation;
- external dense-library parity for Bidiagonal/GK internals;
- LAPACK, SciPy, NumPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or vendor-backend parity;
- package, platform, ABI, public API, CMake/CTest, performance, scalability, memory, or state-of-the-art behavior;
- any new test membership or Windows CTest count behavior.

## Validation Notes

Day 12 changed documentation only. Required validation:

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
| Item 5 is complete or explicitly deferred. | Complete | Extraction is explicitly deferred. |
| General SVD helpers do not absorb specialized Bidiagonal/GK semantics. | Complete | No code moved; future policy requires a dedicated helper owner. |
| Any extraction preserves reconstruction and iteration proof meaning. | Not applicable for Day 12 | No extraction occurred; preservation gates are documented for future movement. |
