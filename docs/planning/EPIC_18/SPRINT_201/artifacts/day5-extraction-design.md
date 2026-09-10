# Sprint 201 Day 5: Extraction Design

## Summary

Day 5 designs the smallest extraction that improves reviewability for the
selected Sprint 201 SVD cluster without changing behavior.

Selected cluster:

- `tests/test_svd.c` rank, pseudoinverse, and dense low-rank helper cluster.

Extraction decision:

- use a header-only test-helper split into `tests/test_svd_selected_helpers.h`;
- keep `tests/test_svd.c` as the proof-owner binary;
- keep the existing `RUN_TEST(...)` names and registration order in
  `tests/test_svd.c`;
- avoid Makefile, CMake, source-list, and CI registration changes.

## Minimal Extraction Shape

| Decision | Selected Shape |
| --- | --- |
| Extraction type | Test-helper split. |
| Destination | Private selected test helper header: `tests/test_svd_selected_helpers.h`; reuse existing shared fixtures from `tests/test_svd_helpers.h`. |
| Source owner | `tests/test_svd.c` remains the registered proof owner. |
| Visibility | `static inline` helper-owned test implementations with `tf_svd_test_*` names. |
| Wrapper strategy | Keep one thin `static void test_*` wrapper per existing `RUN_TEST(...)` name in `tests/test_svd.c`. |
| Build impact | No new compiled source and no build registration updates. |
| Public impact | No public headers, exported symbols, ABI, or production behavior changes. |

The wrapper strategy intentionally preserves the test framework surface while
moving the implementation detail out of the large proof-owner file. This gives
reviewers a smaller `tests/test_svd.c` surface without changing the executable,
test names, or registration order.

## Candidate Helper Groups

| Group | Current Ownership | Dependencies | Call Direction | Day 5 Decision |
| --- | --- | --- | --- | --- |
| Rank fixture helpers | Already partly in `tests/test_svd_helpers.h` through `tf_svd_make_diag_matrix`, `tf_svd_make_rank_deficient_colpair_5x4`, and `tf_svd_make_dependent_row_4x3`. | `SparseMatrix`, `sparse_insert`, `sparse_free`, `ASSERT_ERR`. | Helper header called by selected tests. | Keep and reuse in helper-owned test implementations. |
| Rank test implementations | Currently in `tests/test_svd.c`. | `sparse_svd_rank`, QR rank cross-check for dependent-row fixture, rank assertions. | `tests/test_svd.c` wrappers call helper-owned implementations. | Move selected implementations behind `tf_svd_test_rank_*` helpers. |
| Pseudoinverse test implementations | Currently in `tests/test_svd.c`, with `tf_svd_pinv_first_moore_penrose_error` already in helper header. | `sparse_pinv`, `sparse_matvec`, `vec_norm2`, allocation/free behavior. | `tests/test_svd.c` wrappers call helper-owned implementations. | Move selected implementations behind `tf_svd_test_pinv_*` helpers. |
| Dense low-rank test implementations | Currently in `tests/test_svd.c`, with `tf_svd_dense_lowrank_frobenius_error` already in helper header. | `sparse_svd_compute`, `sparse_svd_lowrank`, `sparse_svd_free`, `sqrt`, allocation/free behavior. | `tests/test_svd.c` wrappers call helper-owned implementations. | Move selected dense low-rank implementations behind `tf_svd_test_lowrank_*` helpers. |
| Sparse low-rank extension tests | Currently adjacent later in `tests/test_svd.c`. | Sparse low-rank output, sparse/dense comparison helpers. | Separate proof block. | Out of scope for Days 6-8 unless implementation proves dense low-rank tests cannot be isolated. |
| Partial SVD helpers | In `tests/test_svd_partial_helpers.h` and shared helper header. | Partial SVD vectors, residuals, corpus behavior. | Separate included helper block. | Out of scope. |

## Proposed Wrapper Map

| Existing `tests/test_svd.c` Test | Proposed Helper Implementation |
| --- | --- |
| `test_svd_rank_full` | `tf_svd_test_rank_full` |
| `test_svd_rank_deficient` | `tf_svd_test_rank_deficient` |
| `test_svd_rank_nearly_singular` | `tf_svd_test_rank_nearly_singular` |
| `test_svd_rank_diagonal_threshold_fixture` | `tf_svd_test_rank_diagonal_threshold_fixture` |
| `test_svd_qr_rank_dependent_row_fixture` | `tf_svd_test_qr_rank_dependent_row_fixture` |
| `test_svd_rank_null` | `tf_svd_test_rank_null` |
| `test_pinv_diagonal` | `tf_svd_test_pinv_diagonal` |
| `test_pinv_moore_penrose` | `tf_svd_test_pinv_moore_penrose` |
| `test_pinv_null` | `tf_svd_test_pinv_null` |
| `test_pinv_rectangular` | `tf_svd_test_pinv_rectangular` |
| `test_pinv_underdetermined_minnorm_solution` | `tf_svd_test_pinv_underdetermined_minnorm_solution` |
| `test_lowrank_diagonal` | `tf_svd_test_lowrank_diagonal` |
| `test_lowrank_error_bound` | `tf_svd_test_lowrank_error_bound` |
| `test_lowrank_errors` | `tf_svd_test_lowrank_errors` |

## Include And Dependency Plan

`tests/test_svd_helpers.h` already includes:

- `sparse_matrix.h`
- `sparse_svd.h`
- `test_framework.h`
- `<math.h>`
- `<stdlib.h>`

Day 6+ should verify whether helper-owned implementations also require:

- `sparse_qr.h` for the dependent-row QR rank cross-check;
- existing declarations visible from `tests/test_svd.c`, such as `vec_norm2`,
  if the underdetermined pseudoinverse test is moved directly.

Preferred dependency resolution:

1. Move only implementations whose dependencies are already valid in the helper
   header.
2. If a selected implementation needs `sparse_qr.h`, add that include to
   `tests/test_svd_helpers.h` only if it is required by the moved helper.
3. If `vec_norm2` is not header-visible, either keep the underdetermined wrapper
   body in `tests/test_svd.c` or move a minimal test-local helper with unchanged
   behavior and naming.
4. Do not introduce production dependencies or cross-suite helper dependencies.

## Formatting, Naming, And Diagnostics Plan

Implementation must preserve:

- existing assertion macros and error checks;
- existing `printf(...)` diagnostic text;
- existing comments where they explain matrix shape, column-major layout, or
  numerical expectations;
- existing cleanup order;
- current tolerances and threshold literals;
- current matrix construction loop ordering.

New helper names should use:

- `tf_svd_test_rank_*` for rank helpers;
- `tf_svd_test_pinv_*` for pseudoinverse helpers;
- `tf_svd_test_lowrank_*` for dense low-rank helpers.

The existing public-facing test names remain unchanged in `RUN_TEST(...)`.

## Build And Registration Plan

| Surface | Day 5 Plan |
| --- | --- |
| `tests/test_svd.c` | Keep include, proof-owner wrappers, and `RUN_TEST(...)` registrations. |
| `tests/test_svd_selected_helpers.h` | Add selected `static inline` helper-owned implementations. |
| `Makefile` | No change. |
| `CMakeLists.txt` | No change. |
| Source-list manifests | No change expected. |
| CI workflow files | No change. |

Because the extraction is header-only, there should be no new object file,
library source, executable target, or install artifact.

## Implementation Checklist For Days 6 Through 8

1. Move the rank helper implementations first and keep thin wrappers in
   `tests/test_svd.c`.
2. Build and run `test_svd` after the rank move before moving the next group.
3. Move the pseudoinverse helper implementations only after confirming helper
   header dependencies.
4. Keep `test_pinv_underdetermined_minnorm_solution` local if moving it would
   require broader helper leakage for `vec_norm2`.
5. Move dense low-rank helper implementations after pseudoinverse checks pass.
6. Preserve `RUN_TEST(...)` registrations and order exactly.
7. Add or update a narrowly scoped ownership guard on Day 8 to protect wrapper
   ownership and registration order.
8. Run the mandatory full C quality gate once `.h` or `.c` files are modified.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Circular helper dependencies | Use only `tests/test_svd_helpers.h` and avoid including partial SVD helper headers from it. |
| Public leakage | Keep all moved code in private test helper headers with `static inline` visibility. |
| Registration drift | Keep all `RUN_TEST(...)` entries in `tests/test_svd.c`; add Day 8 guard coverage. |
| Hidden tolerance drift | Copy tolerance literals verbatim and do not introduce shared tolerance constants. |
| Cleanup-path drift | Review every early return during Day 6 and Day 7 implementation. |
| Header dependency expansion | Add only the includes required by moved helper implementations. |
| Review churn | Move one helper group at a time and keep wrappers mechanical. |

## Day 5 Completion Evidence

Item 201.3 now has a local extraction design. The design avoids public API,
production source, build registration, and CI workflow changes while giving Days
6 through 8 a concrete implementation path.
