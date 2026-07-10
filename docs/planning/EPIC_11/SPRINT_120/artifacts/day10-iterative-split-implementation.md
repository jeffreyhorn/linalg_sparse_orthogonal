# Sprint 120 Day 10 Iterative Split Implementation

## Purpose

Day 10 implements the Day 9 block BiCGSTAB split. The goal is to move the
block BiCGSTAB proof owner out of the main scalar BiCGSTAB test file while
preserving block-specific convergence, aggregation, preconditioner, and error
propagation behavior.

## Implementation Summary

| Surface | Change |
|---|---|
| `tests/test_bicgstab_block.c` | Added focused block BiCGSTAB owner with local static matrix builders, block solve tests, ILU-preconditioned block proof, aggregation proof, and local failing preconditioner proof. |
| `tests/test_bicgstab.c` | Removed block BiCGSTAB tests and their `RUN_TEST(...)` registrations; retained scalar BiCGSTAB, SuiteSparse, numerical hardening, matrix-free, callback, and adjacent comparison proofs. |
| `Makefile` | Added `$(TESTDIR)/test_bicgstab_block.c` after `$(TESTDIR)/test_bicgstab.c` in `TEST_SRCS`. |
| `CMakeLists.txt` | Added `add_sparse_test(test_bicgstab_block)` after `add_sparse_test(test_bicgstab)`. |

## Moved Test Owners

| Test | New owner |
|---|---|
| `test_block_bicgstab_null_inputs` | `tests/test_bicgstab_block.c` |
| `test_block_bicgstab_nrhs_zero` | `tests/test_bicgstab_block.c` |
| `test_block_bicgstab_nrhs_negative` | `tests/test_bicgstab_block.c` |
| `test_block_bicgstab_nonsquare` | `tests/test_bicgstab_block.c` |
| `test_block_bicgstab_2rhs` | `tests/test_bicgstab_block.c` |
| `test_block_bicgstab_4rhs` | `tests/test_bicgstab_block.c` |
| `test_block_bicgstab_matches_single_rhs` | `tests/test_bicgstab_block.c` |
| `test_block_bicgstab_mixed_convergence` | `tests/test_bicgstab_block.c` |
| `test_block_bicgstab_nrhs_1` | `tests/test_bicgstab_block.c` |
| `test_block_bicgstab_preconditioned` | `tests/test_bicgstab_block.c` |
| `test_block_bicgstab_result_aggregation` | `tests/test_bicgstab_block.c` |
| `test_block_bicgstab_error_propagation` | `tests/test_bicgstab_block.c` |

The local `failing_precond` helper moved with the error-propagation scenario.

## Helper Ownership

| Helper | Day 10 decision |
|---|---|
| `build_identity` | Copied as a local static helper in `tests/test_bicgstab_block.c`. |
| `build_unsym_tridiag` | Copied as a local static helper in `tests/test_bicgstab_block.c`. |
| RHS construction loops | Kept inline per scenario so column layout and expected behavior stay visible. |
| `tf_relative_residual_l2` | Reused from `tests/test_solver_helpers.h` with block-owner tolerances. |
| ILU preconditioner calls | Kept in the block owner for the preconditioned block scenario. |
| Matrix-free callbacks | Left in `tests/test_bicgstab.c`; they were not part of this split. |

No broad iterative helper header was introduced.

## Behavior Preserved

| Behavior | Proof |
|---|---|
| Null argument rejection | `test_block_bicgstab_null_inputs` passed. |
| `nrhs == 0` success semantics | `test_block_bicgstab_nrhs_zero` passed. |
| Negative `nrhs` rejection | `test_block_bicgstab_nrhs_negative` passed. |
| Nonsquare shape rejection | `test_block_bicgstab_nonsquare` passed. |
| Two-RHS and four-RHS residual thresholds | `test_block_bicgstab_2rhs` and `test_block_bicgstab_4rhs` passed. |
| Scalar/block equivalence | `test_block_bicgstab_matches_single_rhs` and `test_block_bicgstab_nrhs_1` passed. |
| Mixed convergence and aggregate iteration visibility | `test_block_bicgstab_mixed_convergence` passed. |
| ILU-preconditioned block solve | `test_block_bicgstab_preconditioned` passed. |
| Result aggregation | `test_block_bicgstab_result_aggregation` passed. |
| Preconditioner error propagation | `test_block_bicgstab_error_propagation` passed with `SPARSE_ERR_SINGULAR`. |

## Validation Results

| Command | Result |
|---|---|
| `make format` | Passed. |
| `make build/test_bicgstab build/test_bicgstab_block && ./build/test_bicgstab && ./build/test_bicgstab_block` | Passed: `test_bicgstab` ran 49 tests with 0 failures; `test_bicgstab_block` ran 12 tests with 0 failures. |
| `make source-list-check` | Passed: 49 library sources. |
| `cmake -S . -B build/quality-review-cmake` | Passed. |
| `cmake --build build/quality-review-cmake --parallel 1 --clean-first` | Passed. |
| `ctest -N --test-dir build/quality-review-cmake` | Passed: `test_bicgstab_block` registered as test #40; total tests increased to 56. |
| `make lint` | Passed. |
| `make test` | Passed: all tests passed, including `test_bicgstab` and `test_bicgstab_block`. |

Final whitespace validation is recorded in the Day 10 working notes.

## CTest Membership

The reviewed CMake CTest surface now includes:

- `test_bicgstab` as test #39;
- `test_bicgstab_block` as test #40;
- `Total Tests: 56`.

This matches the Day 9 expected count increase from 55 to 56.

## Non-Claims Preserved

This split is proof-owner cleanup only. It does not claim new BiCGSTAB
behavior, new block solver capability, broader iterative parity,
external-oracle completeness, package support, platform support, ABI stability,
public API expansion, or performance improvement.

## Residual Queue

The following iterative cleanup candidates remain outside Day 10:

- block MINRES split;
- GMRES SuiteSparse/restart/right-preconditioner split;
- matrix-free BiCGSTAB callback split;
- public iterative handle helper movement;
- broad CG/GMRES shared fixture extraction.

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 4 implementation proceeds from exact file boundaries | Complete: the Day 9 block boundary was implemented in `tests/test_bicgstab_block.c`. |
| Convergence and progress-callback behavior remains solver-local and testable | Complete: block convergence and aggregation are in the block owner; matrix-free callback tests remain in `tests/test_bicgstab.c`. |
| Build-system impact is explicit and validated | Complete: Makefile, CMake, source-list, CTest count, focused tests, lint, and full tests passed. |
