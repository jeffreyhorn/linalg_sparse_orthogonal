# Sprint 120 Day 9 Iterative Split Implementation Checklist

## Purpose

Day 9 converts the selected iterative split from the Day 5 ranking into an
implementation-ready checklist. The selected batch is the block BiCGSTAB
scenario owner split from `tests/test_bicgstab.c` into a focused test owner.

This artifact is a design artifact only. It does not move code. Day 10 owns
the C source, Makefile, CMake, CTest, and full-quality implementation proof.

## Selected Iterative Split

| Field | Decision |
|---|---|
| Split name | Block BiCGSTAB scenario owner split |
| Current owner | `tests/test_bicgstab.c` |
| New owner | `tests/test_bicgstab_block.c` |
| New executable | `build/test_bicgstab_block` |
| Existing owner after split | `tests/test_bicgstab.c` remains owner for scalar BiCGSTAB, SuiteSparse, numerical hardening, matrix-free, callback, and adjacent comparison proofs. |
| Implementation day | Day 10 |
| Primary value | Isolate multi-RHS, aggregation, preconditioned block, single-RHS equivalence, argument validation, and block error propagation from the main BiCGSTAB hotspot. |

## Exact Block Boundary

Day 10 should move the contiguous block currently beginning at
`test_block_bicgstab_null_inputs` and ending at
`test_block_bicgstab_error_propagation`.

| Current symbol | Day 10 action | Behavior owner after split |
|---|---|---|
| `test_block_bicgstab_null_inputs` | Move to `tests/test_bicgstab_block.c`. | Block argument null handling. |
| `test_block_bicgstab_nrhs_zero` | Move to `tests/test_bicgstab_block.c`. | `nrhs == 0` no-op success and converged result semantics. |
| `test_block_bicgstab_nrhs_negative` | Move to `tests/test_bicgstab_block.c`. | Negative `nrhs` bad-argument handling. |
| `test_block_bicgstab_nonsquare` | Move to `tests/test_bicgstab_block.c`. | Shape rejection for block solves. |
| `test_block_bicgstab_2rhs` | Move to `tests/test_bicgstab_block.c`. | Two-column solve residual proof. |
| `test_block_bicgstab_4rhs` | Move to `tests/test_bicgstab_block.c`. | Four-column solve residual proof. |
| `test_block_bicgstab_matches_single_rhs` | Move to `tests/test_bicgstab_block.c`. | Block solution equivalence to repeated scalar BiCGSTAB solves. |
| `test_block_bicgstab_mixed_convergence` | Move to `tests/test_bicgstab_block.c`. | Mixed RHS convergence and aggregate iteration proof. |
| `test_block_bicgstab_nrhs_1` | Move to `tests/test_bicgstab_block.c`. | Single-column block parity with scalar BiCGSTAB result fields. |
| `test_block_bicgstab_preconditioned` | Move to `tests/test_bicgstab_block.c`. | ILU-preconditioned block solve residual proof. |
| `test_block_bicgstab_result_aggregation` | Move to `tests/test_bicgstab_block.c`. | Aggregate iteration and residual fields equal the max per-column scalar result. |
| `failing_precond` | Move to `tests/test_bicgstab_block.c`. | Local callback-like preconditioner failure fixture. |
| `test_block_bicgstab_error_propagation` | Move to `tests/test_bicgstab_block.c`. | Block preconditioner failure propagates `SPARSE_ERR_SINGULAR`. |

After moving the block, remove the corresponding `RUN_TEST(...)` calls from
`tests/test_bicgstab.c` and add a new `main` in `tests/test_bicgstab_block.c`
that runs only the block BiCGSTAB scenarios above.

## Helper Ownership Contract

The split should preserve solver-local meaning instead of creating broad
shared helper APIs.

| Helper or include | Day 10 ownership |
|---|---|
| `build_identity` | Copy a local static version into `tests/test_bicgstab_block.c` for null and `nrhs` validation cases. Do not add a shared helper header. |
| `build_unsym_tridiag` | Copy a local static version into `tests/test_bicgstab_block.c` for all block convergence, equivalence, aggregation, and preconditioned cases. |
| `tf_relative_residual_l2` | Reuse from `tests/test_solver_helpers.h`; this is already a shared residual helper with explicit caller tolerances. |
| `sparse_ilu_factor`, `sparse_ilu_precond`, `sparse_ilu_free` | Keep in the new block owner for the preconditioned block scenario only. |
| `sparse_solve_bicgstab` | Keep in the new block owner only where block behavior is compared with repeated scalar solves. |
| `sparse_bicgstab_solve_block` | Primary API under test in the new owner. |
| `failing_precond` | Keep static and scenario-local in `tests/test_bicgstab_block.c`. |
| Block RHS construction loops | Keep inline in each test. Do not introduce a generic block RHS builder in this sprint, because column layout and expected behavior differ by scenario. |

The new file should include only the headers it needs:

- `sparse_ilu.h`;
- `sparse_iterative.h`;
- `sparse_matrix.h`;
- `sparse_types.h`;
- `test_framework.h`;
- `test_solver_helpers.h`;
- standard headers required by the moved code (`math.h`, `stdlib.h`,
  `string.h`).

`tests/test_bicgstab.c` should retain scalar-only helpers that remain used by
its non-block tests. Day 10 should not move matrix-free callbacks, SuiteSparse
fixtures, scalar numerical-hardening helpers, or public handle logic.

## Build and CTest Impact

| Surface | Day 10 change |
|---|---|
| Makefile | Add `$(TESTDIR)/test_bicgstab_block.c` to `TEST_SRCS` immediately after `$(TESTDIR)/test_bicgstab.c`. |
| CMake | Add `add_sparse_test(test_bicgstab_block)` immediately after `add_sparse_test(test_bicgstab)`. |
| Source list check | `make source-list-check` must pass after Makefile/CMake membership changes. |
| CTest count | Current Sprint 120 CMake count is 55 after the Day 7 QR split. Adding `test_bicgstab_block` should make the reviewed CMake registration count 56. |
| Public headers | No public header change expected. |
| Product docs | No public claim or support wording change expected. |

## Focused Iterative Validation Checklist

Day 10 should run the focused and full-quality checks below because it will
modify `.c` and build metadata:

1. `make format`
2. `make build/test_bicgstab build/test_bicgstab_block`
3. `./build/test_bicgstab`
4. `./build/test_bicgstab_block`
5. `make source-list-check`
6. `cmake -S . -B build/quality-review-cmake`
7. `cmake --build build/quality-review-cmake --parallel 1 --clean-first`
8. `ctest -N --test-dir build/quality-review-cmake`
9. `make lint`
10. `make test`
11. `git diff --check`
12. Focused trailing-whitespace scan over
    `docs/planning/EPIC_11/SPRINT_120`, `tests/test_bicgstab.c`, and
    `tests/test_bicgstab_block.c`.

## Behavior Expectations

| Behavior | Required outcome after split |
|---|---|
| Null inputs | Block solve still returns `SPARSE_ERR_NULL` for null matrix, RHS, and solution pointers. |
| `nrhs == 0` | Block solve still returns `SPARSE_OK` and reports convergence. |
| Negative `nrhs` | Block solve still returns `SPARSE_ERR_BADARG`. |
| Nonsquare matrix | Block solve still returns `SPARSE_ERR_SHAPE`. |
| Two-RHS and four-RHS solves | Each RHS column keeps relative residual below the existing `1e-8` threshold. |
| Scalar equivalence | Block output still matches repeated scalar BiCGSTAB to the existing exactness thresholds. |
| Mixed convergence | Both columns converge and aggregate iterations remain positive. |
| `nrhs == 1` parity | Block and scalar result fields and solution values continue to match current thresholds. |
| ILU-preconditioned block solve | ILU factorization, solve, residual checks, and cleanup remain in the block owner. |
| Result aggregation | Block result iterations and residual norm continue to match the max per-column scalar results. |
| Error propagation | `failing_precond` still causes `SPARSE_ERR_SINGULAR`. |
| Progress/callback semantics | Progress callbacks are not part of the selected block split. Callback-like failure propagation remains local through `failing_precond`; matrix-free callback tests stay in `tests/test_bicgstab.c`. |

## Rollback Checklist

If Day 10 movement fails focused or full validation:

1. Move the block BiCGSTAB tests and `failing_precond` back into
   `tests/test_bicgstab.c`.
2. Restore the block `RUN_TEST(...)` registrations in the original
   `tests/test_bicgstab.c` `main`.
3. Remove `tests/test_bicgstab_block.c`.
4. Remove `$(TESTDIR)/test_bicgstab_block.c` from `TEST_SRCS`.
5. Remove `add_sparse_test(test_bicgstab_block)` from `CMakeLists.txt`.
6. Re-run `make format`, focused `test_bicgstab`, `make source-list-check`,
   CMake/CTest membership inspection if CMake was touched, `make lint`, and
   `make test`.
7. Record the failed split reason as residual debt in the Sprint 120
   closeout artifacts.

## Non-Claims Preserved

This split is maintainability and proof-owner cleanup only. It does not claim
new BiCGSTAB capability, broader iterative solver parity, external oracle
completeness, package support, platform support, ABI stability, public API
expansion, or performance improvement.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 4 implementation can proceed from exact file boundaries | Complete: exact symbols, helper ownership, target file, and registrations are named. |
| Convergence and progress-callback behavior remains solver-local and testable | Complete: block convergence and aggregation remain in `test_bicgstab_block`; matrix-free progress/callback tests stay in `test_bicgstab.c`. |
| Any build-system impact is explicit | Complete: Makefile, CMake, CTest count, source-list, focused proof, and full-quality commands are recorded. |
