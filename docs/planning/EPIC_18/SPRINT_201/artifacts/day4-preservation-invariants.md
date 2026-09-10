# Sprint 201 Day 4: Behavior-Preservation Invariants

## Summary

Day 4 converts the selected Sprint 201 cluster boundary into pre-edit
invariants. These invariants define what must stay unchanged when the selected
`tests/test_svd.c` rank, pseudoinverse, and dense low-rank helper cluster is
reduced.

Selected cluster:

- `tests/test_svd.c` rank, pseudoinverse, and low-rank helper cluster.

Preferred extraction surface:

- `tests/test_svd_selected_helpers.h`, reusing shared fixtures from
  `tests/test_svd_helpers.h`

Proof owner:

- `test_svd` remains the only executable proof owner for the selected tests.

## Pre-Edit Invariant Matrix

| Invariant | Required Preservation | Existing Coverage | Later Work |
| --- | --- | --- | --- |
| Function signatures | Selected test functions remain callable through the same `RUN_TEST(...)` names or equivalent static inline helper-owned definitions included by `tests/test_svd.c`. | Current `test_svd` registrations compile and run the selected tests. | Day 5 design must name the exact movement pattern before code edits. |
| Proof ownership | `tests/test_svd.c` keeps the selected `RUN_TEST(...)` registrations and their current order. | Current registration block keeps rank, pseudoinverse, and low-rank tests together. | Day 8 guard should check registration ownership and order. |
| Public API and ABI | No production sources or public headers change. | Day 3 boundary excludes `src/sparse_svd.c`, `src/sparse_svd_partial.c`, and `include/sparse_svd.h`. | Day 10+ validation should confirm no public headers were touched unless explicitly approved. |
| Fixture dimensions | Matrix dimensions remain unchanged for every selected test. | Existing selected tests create fixed 5x5, 5x4, 3x3, 4x4, 4x3, 3x2, 2x4, and 6x6 fixtures. | Day 5 implementation map must preserve fixture constructors and dimensions exactly. |
| Fixture values | Diagonal values, duplicate-column values, dependent-row values, tridiagonal values, and underdetermined-system values remain unchanged. | Existing assertions and printed evidence depend on exact fixture values. | Any helper extraction must copy values verbatim. |
| Sparse insertion behavior | Insertions still use the same success/failure semantics and cleanup ownership. | Existing helper `tf_svd_insert_or_free` frees and nulls the matrix on failed insert. | New helper movement must preserve early-return cleanup and must not hide failed insertions. |
| Error propagation | `ASSERT_ERR(...)` expectations and early-return behavior remain unchanged. | Selected tests assert `SPARSE_OK`, `SPARSE_ERR_NULL`, and `SPARSE_ERR_BADARG` on current paths. | Day 9 focused regression should keep null and bad-argument paths visible. |
| Numerical tolerances | All selected tolerances stay byte-for-byte equivalent. | Current selected tests use `1e-14`, `1e-12`, `1e-10`, `1e-8`, `1e-6`, and default `0.0` tolerance paths. | Extraction must not normalize, rename, or recalculate tolerances unless a later review explicitly approves it. |
| Dense layout | Pseudoinverse and low-rank dense buffers keep current column-major indexing. | Existing selected tests document and assert column-major indexing. | Helper movement must preserve indexing expressions and leading dimensions exactly. |
| Numerical output | Rank values, Moore-Penrose max error thresholds, low-rank diagonal entries, Frobenius errors, and min-norm solution values remain unchanged. | Current selected assertions prove each output contract. | Day 9 focused regression should run `test_svd` after extraction. |
| Deterministic ordering | Matrix construction, loops, assertions, and output prints keep current order. | Current `test_svd` output and assertion sequence are deterministic. | Moving code must not reorder tests, fixtures, or assertions. |
| Cleanup behavior | `free`, `sparse_free`, and `sparse_svd_free` calls remain on the same success and failure paths. | Current selected tests manually release `SparseMatrix`, SVD, pseudoinverse, and low-rank buffers. | Day 6 extraction review should inspect every early return and cleanup branch. |
| Helper scope | New or moved helpers remain SVD-test-local and do not create production or cross-suite dependencies. | Existing `tests/test_svd_helpers.h` is a family-local static inline helper header. | Day 8 guard should prevent selected helpers from becoming external build artifacts. |

## Selected Test Expectations

| Test | Expected Behavior To Preserve |
| --- | --- |
| `test_svd_rank_full` | `sparse_svd_rank(A, 0.0, &rank)` returns `SPARSE_OK` and rank `5` for the 5x5 diagonal fixture. |
| `test_svd_rank_deficient` | duplicate-column 5x4 fixture returns rank `2`. |
| `test_svd_rank_nearly_singular` | default tolerance returns rank `3`; explicit `1e-12` tolerance returns rank `2`. |
| `test_svd_rank_diagonal_threshold_fixture` | diagonal threshold fixture returns ranks `3`, `2`, and `1` at `1e-14`, `1e-10`, and `1e-6`. |
| `test_svd_qr_rank_dependent_row_fixture` | SVD rank and QR rank both return `2` for the dependent-row fixture. |
| `test_svd_rank_null` | null matrix input returns `SPARSE_ERR_NULL`. |
| `test_pinv_diagonal` | diagonal pseudoinverse entries remain `0.5`, `0.25`, and `0.2`, with off-diagonal entries near zero. |
| `test_pinv_moore_penrose` | first Moore-Penrose max error remains below `1e-10`. |
| `test_pinv_null` | null input paths return `SPARSE_ERR_NULL`. |
| `test_pinv_rectangular` | rectangular first Moore-Penrose max error remains below `1e-10`. |
| `test_pinv_underdetermined_minnorm_solution` | solution entries remain near `0.5`, reconstructed right-hand side remains near `[1, 1]`, and solution norm remains near `1.0`. |
| `test_lowrank_diagonal` | rank-2 low-rank diagonal approximation keeps entries `10`, `5`, `0`, `0` and Frobenius error `sqrt(5)`. |
| `test_lowrank_error_bound` | tridiagonal low-rank error remains equal to the full-SVD singular-value tail within `1e-8`. |
| `test_lowrank_errors` | null and bad-rank calls return current error codes. |

## Registration Surfaces

No registration changes are planned for Sprint 201 unless Day 5 or Day 6 proves
that a helper-only extraction cannot meet the reviewability goal.

| Surface | Expected Day 4 Disposition |
| --- | --- |
| `tests/test_svd.c` | Keep selected `RUN_TEST(...)` registrations and order. |
| `tests/test_svd_selected_helpers.h` | Preferred static inline selected-helper destination; `tests/test_svd_helpers.h` remains shared fixture support. |
| `Makefile` | No planned change. |
| `CMakeLists.txt` | No planned change. |
| source-list checks | Run later only if registration or source-list files change; otherwise keep as drift safety. |
| CI workflow files | No planned change. |

## Reviewability Goals

Sprint 201 should leave the selected SVD cluster easier to review by:

- reducing the line count and conceptual load inside `tests/test_svd.c`;
- making the rank, pseudoinverse, and dense low-rank helper ownership explicit;
- keeping selected helpers named with the existing `tf_svd_` family prefix;
- avoiding broad helper headers that mix full SVD, partial SVD, condition-number,
  and external-reference concerns;
- keeping all solver behavior, public surface, and generated evidence unchanged.

## Pre-Edit Checklist

Before Day 5+ code edits:

- confirm the exact helper/test movement pattern;
- confirm selected `RUN_TEST(...)` entries remain in `tests/test_svd.c`;
- confirm no production source or public header edits are needed;
- copy fixture values and tolerances verbatim if helpers move;
- preserve column-major indexing and leading-dimension assumptions;
- preserve cleanup on every allocation and SVD/pseudoinverse/low-rank failure
  path;
- plan focused validation with `test_svd`;
- plan the full C quality gate once `.h` or `.c` files are modified.

## Day 4 Completion Evidence

Item 201.2 remains complete and Day 4 prepares item 201.3. The selected cluster
now has pre-edit invariants, existing coverage mapping, registration-surface
disposition, reviewability goals, and a checklist for implementation.
