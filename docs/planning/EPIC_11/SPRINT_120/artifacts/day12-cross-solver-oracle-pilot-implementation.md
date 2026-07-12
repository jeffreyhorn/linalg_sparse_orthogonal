# Sprint 120 Day 12 Cross-Solver Oracle Pilot Implementation

## Purpose

Day 12 implements the bounded cross-solver oracle pilot designed on Day 11.
The pilot adds one focused test owner that compares LU, Cholesky, QR, and CG on
one small generated-RHS SPD fixture without expanding public claims, benchmark
scope, package scope, or platform guarantees.

## Implementation Summary

| Surface | Change |
|---|---|
| `tests/test_cross_solver_oracle.c` | Added a focused cross-solver oracle pilot owner. |
| Makefile | Added `$(TESTDIR)/test_cross_solver_oracle.c` to `TEST_SRCS` after `test_bicgstab_block.c`. |
| `CMakeLists.txt` | Added `add_sparse_test(test_cross_solver_oracle)` after `test_bicgstab_block`. |
| Public docs | No change. |
| Headers/API | No change. |

## Pilot Fixture

| Field | Value |
|---|---|
| Matrix | Local SPD tridiagonal fixture. |
| Dimension | `8 x 8`. |
| Diagonal | `4.0`. |
| Off-diagonal | `-1.0`. |
| Exact solution | `x_exact[i] = 1.0 + 0.25 * i`. |
| RHS | Computed locally as `b = A * x_exact`. |
| Oracle model | Generated exact solution plus relative residual checks. |

The fixture builder, exact-solution builder, RHS generation, and solution
max-difference helper are local to the new test file. Day 12 does not promote
the pilot helpers into shared test infrastructure.

## Solver Coverage

| Solver | API path | Acceptance |
|---|---|---|
| LU | `sparse_lu_factor(..., SPARSE_PIVOT_PARTIAL, 1e-12)` plus `sparse_lu_solve`. | Relative residual below `1e-10`; max solution difference below `1e-8`. |
| Cholesky | `sparse_cholesky_factor` plus `sparse_cholesky_solve`. | Relative residual below `1e-10`; max solution difference below `1e-8`. |
| QR | `sparse_qr_factor` plus `sparse_qr_solve`. | Relative residual below `1e-10`; reported QR residual below `1e-10`; max solution difference below `1e-8`. |
| CG | `sparse_solve_cg` with `max_iter = 100` and `tol = 1e-12`. | Converges; relative residual below `1e-10`; max solution difference below `1e-8`. |

## Test Shape

| Test | Purpose |
|---|---|
| `test_spd_generated_rhs_lu_chol_qr_cg_agree` | Solves the generated-RHS SPD fixture with LU, Cholesky, QR, and CG, then checks residuals and generated-solution recovery for each solver. |

The test uses `tf_relative_residual_l2` as a measurement helper and a local
`max_abs_diff` helper for generated-solution agreement.

## Focused Validation Evidence

| Command | Result |
|---|---|
| `make format` | Passed. |
| `make build/test_cross_solver_oracle && ./build/test_cross_solver_oracle` | Passed. |
| `make source-list-check` | Passed with 49 library sources. |
| `cmake -S . -B build/quality-review-cmake` | Passed. |
| `cmake --build build/quality-review-cmake --parallel 1 --clean-first` | Passed. |
| `ctest -N --test-dir build/quality-review-cmake` | Passed; `test_cross_solver_oracle` registered as test #41; total registered tests increased to 57. |
| `make lint` | Passed. |
| `make test` | Passed. |

Focused pilot output:

```text
=== Cross-solver oracle pilot ===
    LU: rel_res=2.052e-16, max|x-x_exact|=4.441e-16
    Cholesky: rel_res=2.163e-16, max|x-x_exact|=4.441e-16
    QR: rel_res=3.299e-16, max|x-x_exact|=4.441e-16
    CG: rel_res=2.490e-16, max|x-x_exact|=4.441e-16
  [PASS] test_spd_generated_rhs_lu_chol_qr_cg_agree
Tests run: 1, failed 0, assertions 20.
```

## CTest Registration

| Registration item | Result |
|---|---|
| New CMake test name | `test_cross_solver_oracle`. |
| CTest position | Test #41 in the reviewed CMake surface. |
| Total reviewed CTest count | 57. |
| Source-list status | Passed. |
| Full Makefile test status | Passed. |

## Non-Claims Preserved

- The pilot does not claim broad direct/iterative solver parity.
- The pilot does not claim external-oracle completeness.
- The pilot does not claim SuiteSparse matrix coverage for these solvers.
- The pilot does not claim performance, scalability, package, platform, ABI, or
  install behavior.
- The pilot does not add or change public API.
- The pilot does not promote one local generated-RHS fixture into a general
  product guarantee.

## Residual Limitations

| Limitation | Deferred owner |
|---|---|
| GMRES SuiteSparse, restart, and right-preconditioner cross-solver split | Future iterative owner cleanup. |
| MINRES LDLT and GMRES comparison split | Future iterative owner cleanup. |
| LDLT cross-backend and cross-solver split | Future direct owner cleanup. |
| External dense-reference oracle expansion | Future external oracle sprint. |
| Package/install/platform validation for oracle tests | Future packaging or CI sprint. |

## Completion Check

| Criterion | Status |
|---|---|
| New focused cross-solver oracle owner exists. | Complete. |
| Makefile registration exists. | Complete. |
| CMake registration exists. | Complete. |
| Generated-RHS SPD fixture is local and bounded. | Complete. |
| LU, Cholesky, QR, and CG all solve the fixture within local tolerances. | Complete. |
| CTest registration count increased to 57. | Complete. |
| Public claims remain unchanged. | Complete. |
