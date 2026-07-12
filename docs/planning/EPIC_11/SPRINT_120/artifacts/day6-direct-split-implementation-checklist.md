# Sprint 120 Day 6 Direct Split Implementation Checklist

## Purpose

Day 6 converts the Day 5 direct split selection into an exact implementation
plan. The selected batch is the QR solve scenario owner split from
`tests/test_qr.c`. The plan defines the proposed new owner, exact test blocks,
helper ownership, Makefile/CMake/CTest impact, focused validation, expected
failure behavior, and rollback path required before Day 7 edits begin.

This artifact is design-only. It does not move C code.

## Selected Direct Batch

| Field | Decision |
|---|---|
| Batch | QR solve scenario owner split |
| Current owner | `tests/test_qr.c` |
| Proposed new owner | `tests/test_qr_solve.c` |
| Current CTest executable | `test_qr` |
| Proposed new CTest executable | `test_qr_solve` |
| Primary goal | Move QR least-squares solve, SuiteSparse solve, and QR-vs-LU solve proof blocks into a focused owner while preserving QR-specific residual and tolerance contracts. |
| Out of scope | QR factorization basics, reconstruction-only tests, rank/null-space tests, reordering tests, economy QR tests, sparse-mode tests, refinement tests, package/platform/API claims, and broad direct/iterative parity. |

## Exact Direct Test Blocks

| Order | Function | Current role | Day 7 action |
|---:|---|---|---|
| 1 | `test_qr_solve_square` | Square full-rank QR solve compared with LU and true residual. | Move to `tests/test_qr_solve.c`. |
| 2 | `test_qr_solve_overdetermined` | Least-squares solve with positive residual and reported-vs-true residual agreement. | Move to `tests/test_qr_solve.c`. |
| 3 | `test_qr_solve_analytical` | Analytical 2x1 least-squares solution and absolute residual. | Move to `tests/test_qr_solve.c`. |
| 4 | `test_qr_solve_rank_deficient` | Rank-deficient solve with reasonable residual and QR rank assertion. | Move to `tests/test_qr_solve.c`. |
| 5 | `test_qr_solve_nos4` | SuiteSparse `nos4` generated-RHS QR solve with true residual threshold. | Move to `tests/test_qr_solve.c`. |
| 6 | `test_qr_solve_null_residual` | QR solve accepts `NULL` residual pointer and still solves accurately. | Move to `tests/test_qr_solve.c`. |
| 7 | `test_qr_bcsstk04` | SuiteSparse `bcsstk04` QR rank, reconstruction, generated-RHS solve, and loose solve tolerance. | Move to `tests/test_qr_solve.c`; keep reconstruction assertion because the current test couples rank/reconstruction with solve proof. |
| 8 | `test_qr_west0067` | SuiteSparse `west0067` generated-RHS QR solve. | Move to `tests/test_qr_solve.c`. |
| 9 | `test_qr_vs_lu` | SuiteSparse `nos4` QR-vs-LU comparison with true residuals and max solution difference. | Move to `tests/test_qr_solve.c`; keep wording as a bounded comparison, not broad parity. |
| 10 | `test_qr_tall_synthetic` | Tall 50x20 QR reconstruction plus generated-RHS solve. | Move only if Day 7 can keep reconstruction semantics visible in the focused owner; otherwise leave in `test_qr.c` and record the residual. |

Day 7 should remove the corresponding `RUN_TEST(...)` calls from
`tests/test_qr.c` when a function moves, and register them in the new
`tests/test_qr_solve.c` `main` in the same order.

## Helper Ownership Contract

| Helper / fixture | Current owner | Day 7 design |
|---|---|---|
| `compute_rel_residual` | Static helper in `tests/test_qr.c` | Duplicate or move into `tests/test_qr_solve.c` only for QR solve ownership. Do not replace with a generic helper during Day 7. |
| `assert_qr_true_residual_below` | Static helper in `tests/test_qr.c` | Move with QR solve tests because it owns reported residual versus true residual print/assert semantics. |
| `make_qr_exact_rhs` | Static helper near file top in `tests/test_qr.c` | Keep in both owners if still needed by remaining QR tests, or extract into a narrow QR-only test helper only if Day 7 proves duplication is worse than a helper. The generated vector remains `A * [1, 2, ...]`. |
| `qr_idx_count_bytes` | Static allocation guard for QR exact RHS | Keep with any owner that uses `make_qr_exact_rhs`; do not generalize. |
| `qr_reconstruction_error` / `assert_qr_reconstruction_below` | Static helpers in `tests/test_qr.c` | If `test_qr_bcsstk04` or `test_qr_tall_synthetic` moves with reconstruction assertions, either keep a QR-solve-local copy or introduce a narrow QR test helper. Do not change reconstruction tolerance semantics. |
| `make_qr_duplicate_column_4x3` | Static QR fixture helper used by rank-deficient solve and other rank tests | If moving `test_qr_solve_rank_deficient`, Day 7 may duplicate this small fixture locally or extract a narrow QR fixture helper if remaining `test_qr.c` tests still need it. |

Design preference for Day 7: use a new focused `tests/test_qr_solve.c` with
local static helpers copied from `tests/test_qr.c` where needed. Avoid a broad
shared helper header in the first split unless duplication causes a concrete
compile or maintenance problem.

## Proposed New File Shape

`tests/test_qr_solve.c` should contain:

1. The same includes required by the moved tests:
   - `sparse_lu.h`;
   - `sparse_matrix.h`;
   - `sparse_qr.h`;
   - `sparse_types.h`;
   - `test_framework.h`;
   - standard headers needed by moved helpers.
2. `DATA_DIR` and `SS_DIR` definitions matching `tests/test_qr.c`.
3. QR-solve-local helper prototypes and static helper definitions.
4. Moved QR solve test functions in the order listed above.
5. A `main` that runs only the moved QR solve scenario tests.

`tests/test_qr.c` should remain the owner for the rest of QR coverage and
should retain any helper still needed by unmoved tests.

## Build and CTest Impact

| Surface | Required Day 7 update |
|---|---|
| Makefile `TEST_SRCS` | Add `$(TESTDIR)/test_qr_solve.c` near `$(TESTDIR)/test_qr.c`. |
| Makefile focused build target | `make build/test_qr build/test_qr_solve` should compile both owners through existing pattern rules. |
| CMake | Add `add_sparse_test(test_qr_solve)` immediately after `add_sparse_test(test_qr)`. |
| CTest | Expected local CTest count increases by one where `test_qr_solve` is registered. Existing `test_qr` remains registered. |
| Source-list check | `make source-list-check` must pass after the new test file is added to Makefile/CMake inventories. |
| Full quality | Because Day 7 will modify `.c` and likely build metadata, run `make format && make lint && make test`. |

## Focused Direct Validation Checklist

Run these in order during Day 7 after implementation:

1. `make build/test_qr build/test_qr_solve`
2. `./build/test_qr`
3. `./build/test_qr_solve`
4. `make source-list-check`
5. CMake configure/build and `ctest -N` count proof if CMake membership is
   changed:
   - `cmake -S . -B build/quality-review-cmake`
   - `cmake --build build/quality-review-cmake --parallel 1 --clean-first`
   - `ctest -N --test-dir build/quality-review-cmake`
6. `make format && make lint && make test`

If any focused command fails, stop implementation work, inspect whether the
failure is a movement error or pre-existing unrelated failure, and do not
proceed to broader quality until the focused failure is resolved or explicitly
deferred.

## Expected Failure Behavior to Preserve

| Behavior | Required preservation |
|---|---|
| Square QR solve | QR and LU solutions agree within `1e-8`; true residual and reported residual remain below `1e-10`. |
| Overdetermined solve | Residual is positive; reported residual divided by `||b||` matches true relative residual within `1e-8`; tolerance remains intentionally loose at `1.0`. |
| Analytical solve | Solution remains `2.0` within `1e-10`; residual remains `sqrt(2.0)` within `1e-10`. |
| Rank-deficient solve | QR rank assertion remains `2`; residual remains reasonable under the current threshold. |
| SuiteSparse `nos4` solve | Generated RHS remains `A * [1, 2, ...]`; true residual threshold remains `1e-8`. |
| `NULL` residual solve | `sparse_qr_solve(..., NULL)` remains accepted and true residual remains below `1e-10`. |
| SuiteSparse `bcsstk04` | Rank and reconstruction remain checked if the test moves; solve threshold remains `1e-4`. |
| SuiteSparse `west0067` | Solve threshold remains `1e-8`. |
| QR-vs-LU `nos4` | QR and LU residuals remain below `1e-8`; max solution difference remains below `1e-4`; wording remains a bounded comparison only. |
| Tall synthetic solve | Reconstruction threshold and solve threshold remain unchanged if moved. |

## Rollback Checklist

If Day 7 movement fails:

1. Move any selected QR solve functions back into `tests/test_qr.c`.
2. Restore the original `RUN_TEST(...)` calls in `tests/test_qr.c`.
3. Remove `tests/test_qr_solve.c`.
4. Remove `$(TESTDIR)/test_qr_solve.c` from Makefile `TEST_SRCS`.
5. Remove `add_sparse_test(test_qr_solve)` from `CMakeLists.txt`.
6. Re-run `make build/test_qr && ./build/test_qr`.
7. Re-run `make source-list-check` if build metadata was touched.
8. Re-run `make format && make lint && make test` if `.c` or `.h` files were
   modified before rollback.
9. Record the residual direct split owner and failure reason in the Day 7
   artifact.

Rollback is required if the split hides QR-specific residual semantics,
changes tolerances, changes SuiteSparse skip/load behavior, changes CTest
membership unexpectedly, or broadens direct/iterative parity claims.

## Non-Claims

This design does not claim new QR functionality, broad direct solver parity,
external-oracle completeness, platform/package support, public API expansion,
performance improvement, or state-of-the-art validation. It only prepares a
bounded QR solve test-owner split.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 3 implementation can proceed from exact file boundaries | Complete: exact QR solve functions, helper ownership, and new file shape are identified. |
| Any build-system impact is explicit | Complete: Makefile, CMake, CTest, source-list, focused build, and full-quality expectations are documented. |
| Direct-solver behavior preservation is measurable before and after the split | Complete: focused commands and behavior-specific residual/tolerance expectations are recorded. |
