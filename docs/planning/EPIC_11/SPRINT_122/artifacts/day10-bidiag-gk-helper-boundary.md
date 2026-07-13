# Sprint 122 Day 10 Bidiagonal/Golub-Kahan Helper Boundary Review

## Purpose

Day 10 completes Sprint 122 Item 5 by reviewing whether Bidiagonal and
Golub-Kahan helper extraction should proceed after Sprint 121 deferred it as
too semantic for the first SVD helper batch.

The review covers `tests/test_bidiag.c`, the Golub-Kahan extraction and
validation blocks in `tests/test_svd.c`, and the public/internal boundaries in
`include/sparse_bidiag.h`, `include/sparse_svd.h`, and `src/sparse_svd.c`.

## Decision

Do not consolidate Bidiagonal/Golub-Kahan helpers into the general SVD helper
layer in Sprint 122.

Limited future extraction is acceptable only for behavior-named measurement
helpers, not broad assertion wrappers. The current checks encode specialized
transpose, Householder, upper-bidiagonal, wide-matrix, reconstruction,
orthogonality, and bidiagonal-QR semantics that would be obscured by a generic
SVD helper.

## Current Ownership Inventory

| Surface | Current Owner | Behavior Owned |
| --- | --- | --- |
| `bidiag_reconstruction_error` | `tests/test_bidiag.c` | Reconstructs `A = U*B*V^T` from implicit Householder sequences and recursively handles internally transposed wide matrices. |
| `test_bidiag_*` family | `tests/test_bidiag.c` | Bidiagonal factorization shape, diagonal/superdiagonal behavior, wide transpose flag, single-row, diagonal, `nos4`, null, and free-zeroed behavior. |
| `gk_reconstruction_error` | `tests/test_svd.c` | Reconstructs from explicit extracted `U`, `V`, and bidiagonal arrays for non-transposed Golub-Kahan extraction checks. |
| `test_gk_extract_*` | `tests/test_svd.c` | Extracted `U`/`V` reconstruction for square/tall fixtures and explicit wide-transpose skip behavior. |
| `validate_gk` and `test_gk_*` | `tests/test_svd.c` | Golub-Kahan reconstruction and column-orthogonality checks with special handling for transposed wide bidiagonalizations. |
| `test_bidiag_svd_*` | `tests/test_svd.c` | Implicit QR iteration on bidiagonal arrays, analytical 2x2 singular values, UV accumulation, deflation, and k=1 behavior. |
| `sparse_bidiag_t` / `sparse_bidiag_factor` | `include/sparse_bidiag.h` | Public bidiagonal factorization contract, including internal transpose interpretation for wide matrices. |
| `sparse_svd_extract_uv` / `bidiag_svd_iterate` | `include/sparse_svd.h` / `src/sparse_svd.c` | Explicit `U`/`V` extraction and bidiagonal QR iteration contracts used by SVD. |

## Specialized Semantics

| Semantic | Why General SVD Helpers Must Not Absorb It |
| --- | --- |
| Internal transpose for wide matrices | `sparse_bidiag_factor` factors `A^T` when `m < n`; tests must preserve `transposed` interpretation and U/V swapping rather than pretending every reconstruction is `A = U*B*V^T` with an upper bidiagonal for `A`. |
| Householder reflector application | `tests/test_bidiag.c` reconstructs from implicit reflector sequences, not dense SVD `U`/`V` outputs. |
| Upper-bidiagonal storage | `diag` and `superdiag` represent a compact upper bidiagonal for tall/square cases and `A^T` for wide cases. |
| Explicit extracted `U`/`V` shape | Golub-Kahan extraction returns `m x k` and `n x k` dense matrices with column-major layout; this differs from full SVD `U`, `sigma`, `Vt` result ownership. |
| Wide reconstruction skip | Current GK validation intentionally skips direct reconstruction when `bd.transposed` and relies on end-to-end wide SVD reconstruction elsewhere. A generic helper could erase that guardrail. |
| Bidiagonal QR iteration | `bidiag_svd_iterate` tests operate directly on bidiagonal arrays and optional UV accumulation; they are not sparse-matrix SVD result tests. |
| Tolerance meaning | Reconstruction, orthogonality, analytical singular values, `nos4`, and zero-superdiag deflation use different tolerances and diagnostics. |

## Consolidation Risk Assessment

| Candidate | Disposition | Reason |
| --- | --- | --- |
| Move `bidiag_reconstruction_error` into `tests/test_svd_helpers.h` | Defer | It depends on `sparse_bidiag_t`, implicit Householder vectors, and transpose recursion; a general SVD helper header would inherit bidiagonal internals. |
| Move `gk_reconstruction_error` into `tests/test_svd_helpers.h` | Defer | It is narrower than full SVD reconstruction and assumes explicit extracted `U`/`V` plus upper-bidiagonal arrays. |
| Extract a new `tests/test_bidiag_helpers.h` | Accept only as future maintainability work | This could be safe if helper names remain bidiagonal/GK-specific and both `test_bidiag.c` and `test_svd.c` keep scenario-level assertions visible. |
| Create generic `assert_svd_reconstruction` wrappers | Reject | Would hide transpose handling, U/V/Vt layout differences, and tolerance ownership. |
| Share simple fixture builders | Accept as future limited work | Small dense/tall/wide fixtures could be shared if names encode bidiagonal/GK use and do not move assertions. |
| Share dense column orthogonality helper | Already acceptable | Existing SVD helper measurement can remain shared because callers still own tolerance and scenario labels. |
| Move bidiagonal QR iteration tests into helper assertions | Reject | These are algorithm-kernel proof owners, not reusable SVD result assertions. |

## Future Extraction Boundary

A future extraction may proceed only under a Bidiagonal/Golub-Kahan helper owner
with a dedicated header such as `tests/test_bidiag_helpers.h`. The first safe
batch should be limited to measurement or fixture builders:

1. `tf_bidiag_reconstruction_error_from_reflectors` for implicit
   Householder/bidiagonal reconstruction, including explicit `transposed`
   behavior.
2. `tf_gk_reconstruction_error_from_extracted_uv` for explicit `U`/`V` plus
   bidiagonal arrays.
3. Fixture builders for square, tall, wide, single-row, diagonal, and
   rank-deficient bidiagonal/GK cases.
4. Optional allocation-size checks if future helpers allocate dense buffers.

The following must remain at the test call site:

- fixture-specific tolerance;
- whether wide transpose reconstruction is skipped or asserted;
- printed diagnostic label;
- whether the test owns factorization, extraction, QR iteration, or full SVD;
- any public API error contract.

## Validation Requirements for Future Movement

Any future helper movement will touch `.c` and `.h` files, so the validation
gate must be:

1. `make format`
2. `make build/test_bidiag && ./build/test_bidiag`
3. `make build/test_svd && ./build/test_svd`
4. `make lint`
5. `make test`
6. `git diff --check`
7. focused trailing-whitespace scan over touched files and Sprint docs

If Makefile, CMake, CTest membership, or Windows reviewed-count behavior
changes, the owner must add source-list and CTest surface proof before review.

## Affected Surfaces

| Surface | Day 10 Action |
| --- | --- |
| `tests/test_bidiag.c` | No code movement; remains the owner of implicit Householder bidiagonal reconstruction and factorization checks. |
| `tests/test_svd.c` | No code movement; remains the owner of Golub-Kahan extraction, validation, and bidiagonal QR iteration checks. |
| `tests/test_svd_helpers.h` | No general SVD helper absorbs bidiagonal/GK behavior. |
| `tests/test_svd_partial_helpers.h` | No change; partial-SVD vector/top-k helpers remain separate. |
| `include/sparse_bidiag.h` / `include/sparse_svd.h` | No API change. |
| Makefile / CMake / CTest | No change. |
| Production source | No change. |
| Public docs / package / ABI | No change. |

## Non-Claim Register

Day 10 does not claim:

- broad SVD helper consolidation;
- external dense-library parity for Bidiagonal or Golub-Kahan internals;
- LAPACK, SciPy, NumPy, PETSc, Trilinos, Eigen, package, ABI, performance,
  scalability, or state-of-the-art parity;
- any change in public API support level;
- any new test membership or Windows CTest count behavior.

## Validation Notes

Day 10 changed documentation only. Required validation is `git diff --check`
and a focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_122`.
The branch passed `make format`, `make lint`, and `make test` after the Day 8
C/header changes; Day 10 adds no new code changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 5 helper-boundary work is complete. | Complete | Day 9 documented minimum-norm ownership; Day 10 documents Bidiagonal/Golub-Kahan ownership. |
| Specialized transpose and reconstruction semantics are preserved. | Complete | No code movement was performed; future extraction boundaries explicitly preserve transposed wide behavior. |
| No general SVD helper absorbs behavior without proof-owner clarity. | Complete | General SVD helper consolidation is rejected; only future bidiagonal/GK-specific measurement extraction is allowed. |
