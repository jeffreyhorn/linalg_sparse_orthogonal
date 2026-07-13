# Sprint 121 Day 6: SVD Helper Extraction

## Purpose

Extract the first bounded batch of SVD proof helpers from `tests/test_svd.c`
without changing reviewed test ownership, source-list membership, CTest
registration, tolerances, expected failures, or product-truth claims.

## Touched Surfaces

| Surface | Change |
|---|---|
| `tests/test_svd_helpers.h` | Added header-only SVD fixture, reconstruction, orthogonality, pseudoinverse, and low-rank comparison helpers. |
| `tests/test_svd.c` | Included the helper header and replaced local helper bodies with `tf_svd_*` or `tf_dense_*` calls. |
| `docs/planning/EPIC_11/SPRINT_121/WORKING_NOTES.md` | Recorded Day 6 implementation evidence and residual helper queue. |
| `docs/planning/EPIC_11/SPRINT_121/artifacts/day6-svd-helper-extraction.md` | Added this implementation artifact. |

No Makefile, CMake, workflow, package, benchmark, public API, or production
source files were changed.

## Extracted Helper Boundaries

| Helper | New owner | Preserved behavior boundary |
|---|---|---|
| `tf_svd_insert_or_free` | `tests/test_svd_helpers.h` | Keeps allocation cleanup and `ASSERT_ERR(..., SPARSE_OK)` behavior from the original local helper. |
| `tf_svd_make_diag_matrix` | `tests/test_svd_helpers.h` | Keeps exact diagonal fixture construction for singular-value, rank, low-rank, and condition-number tests. |
| `tf_svd_make_rank1_row_progression` | `tests/test_svd_helpers.h` | Keeps deterministic rank-1 fixture construction for singular-value and UV reconstruction tests. |
| `tf_svd_make_rank_deficient_colpair_5x4` | `tests/test_svd_helpers.h` | Keeps duplicate-column rank-deficient fixture construction for SVD-vs-QR rank and rank tests. |
| `tf_svd_make_full_uv_fixture_16x8` | `tests/test_svd_helpers.h` | Keeps deterministic full-UV fixture construction for full/economy orthogonality and reconstruction tests. |
| `tf_dense_column_orthogonality_error` | `tests/test_svd_helpers.h` | Keeps column-major `Q^T Q` max-error measurement for GK and SVD U/V checks. |
| `tf_svd_vt_row_orthogonality_error` | `tests/test_svd_helpers.h` | Keeps full-mode/economy Vt row-orthogonality Frobenius measurement. |
| `tf_svd_reconstruction_max_error` | `tests/test_svd_helpers.h` | Keeps max-entry `U Sigma Vt` reconstruction measurement. |
| `tf_svd_reconstruction_rel_frobenius` | `tests/test_svd_helpers.h` | Keeps relative Frobenius reconstruction measurement and zero-norm fallback. |
| `tf_svd_pinv_first_moore_penrose_error` | `tests/test_svd_helpers.h` | Keeps first Moore-Penrose identity dimensions and max-error measurement. |
| `tf_svd_dense_lowrank_frobenius_error` | `tests/test_svd_helpers.h` | Keeps dense low-rank Frobenius residual measurement. |
| `tf_svd_sparse_dense_frobenius_diff` | `tests/test_svd_helpers.h` | Keeps sparse-vs-dense low-rank Frobenius comparison. |
| `tf_svd_sparse_dense_max_abs_diff` | `tests/test_svd_helpers.h` | Keeps sparse-vs-dense low-rank max-entry comparison. |
| `tf_svd_sparse_sparse_rel_frobenius_diff` | `tests/test_svd_helpers.h` | Keeps sparse-vs-sparse relative Frobenius comparison and zero-norm fallback. |

## Preserved Local Owners

The extraction deliberately left these boundaries in `tests/test_svd.c`:

- `gk_reconstruction_error`, because it is specific to Golub-Kahan bidiagonal
  extraction rather than general SVD proof measurement.
- `validate_gk`, because it owns GK scenario labels, transposed-path handling,
  and local reconstruction/orthogonality thresholds.
- `assert_svd_cond_near` and `assert_svd_cond_inf`, because they encode
  condition-number assertion semantics and labels.
- Partial-SVD vector/residual helpers, because Day 10 should expand those
  internal-reference semantics before moving helper ownership.
- Scenario-level tolerances, expected failures, skips, and non-claim wording,
  because helpers should not hide proof interpretation.

## Source-List And CTest Impact

- Added a header-only test helper included by `tests/test_svd.c`.
- No new test executable was registered.
- No CTest count change is expected.
- No Makefile or CMake source membership change is required.

## Focused Behavior Evidence

The focused SVD executable validates the extracted fixture and measurement
helpers through the existing reviewed SVD test owner:

```text
make build/test_svd && ./build/test_svd
Tests run:    98
Tests failed: 0
Tests skipped: 0
Assertions:   1580
ALL TESTS PASSED
```

This evidence covers the extracted reconstruction, orthogonality, rank,
pseudoinverse, low-rank, sparse-vs-dense, and sparse-vs-sparse measurement
paths through their existing SVD scenarios.

## Required Quality Evidence

Because Day 6 changed `.c` and `.h` files, the required quality chain is:

```text
make format && make lint && make test
All tests passed.
```

Additional cleanliness checks passed:

```text
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_11/SPRINT_121 tests/test_svd.c tests/test_svd_helpers.h || true
```

## Residual Helper Queue

| Candidate | Owner | Reason deferred |
|---|---|---|
| Partial-SVD vector residual helpers | Day 10 | Existing proofs use internal full-SVD references and looser vector residual semantics that need expansion before extraction. |
| Bidiagonal/Golub-Kahan helpers | Future maintainability work | Current helpers encode GK-specific transpose and reconstruction semantics, not general SVD proof helpers. |
| QR reconstruction/residual helpers | Day 7 | Planned as the next bounded extraction under QR-specific tolerance ownership. |
| Minimum-norm helpers in `tests/test_colamd.c` | Day 9 or explicit deferral | Ownership remains historically tied to COLAMD/reordering tests until minimum-norm proof boundaries are revisited. |

## Non-Claims Preserved

Day 6 does not claim LAPACK, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, or
state-of-the-art parity. It only preserves existing in-repository SVD behavior
while moving reusable test fixtures and measurement code behind named helper
boundaries.

## Completion Criteria Status

| Criterion | Status |
|---|---|
| Item 3 SVD helper extraction has an implemented first batch. | Complete. |
| Focused SVD tests pass. | Complete. |
| Behavior-preservation evidence is recorded before broader refactor. | Complete. |
