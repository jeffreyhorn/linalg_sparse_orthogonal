# Sprint 120 Day 13 Validation Package

## Purpose

Day 13 packages validation evidence for the Sprint 120 direct/iterative oracle
architecture work completed so far. It revalidates the touched direct split,
iterative split, and cross-solver pilot owners; confirms source-list and CMake
membership; and records full quality-gate results for the branch's `.c` and
build metadata changes.

## Touched Validation Surfaces

| Surface | Why it is in scope |
|---|---|
| `tests/test_qr.c` | Direct QR proof owner after Day 7 split. |
| `tests/test_qr_solve.c` | Focused QR solve scenario owner added by the direct split. |
| `tests/test_bicgstab.c` | Scalar BiCGSTAB owner after Day 10 split. |
| `tests/test_bicgstab_block.c` | Focused block BiCGSTAB owner added by the iterative split. |
| `tests/test_cross_solver_oracle.c` | Cross-solver oracle pilot owner added on Day 12. |
| Makefile | Test source registration changed for new focused owners. |
| `CMakeLists.txt` | CTest registration changed for new focused owners. |
| `docs/planning/EPIC_11/SPRINT_120` | Sprint planning, evidence, and validation package artifacts. |

## Validation Commands

| Command | Result |
|---|---|
| `make format` | Passed. |
| `make build/test_qr build/test_qr_solve build/test_bicgstab build/test_bicgstab_block build/test_cross_solver_oracle` | Passed. |
| `./build/test_qr` | Passed. |
| `./build/test_qr_solve` | Passed. |
| `./build/test_bicgstab` | Passed. |
| `./build/test_bicgstab_block` | Passed. |
| `./build/test_cross_solver_oracle` | Passed. |
| `make source-list-check` | Passed. |
| `cmake -S . -B build/quality-review-cmake` | Passed. |
| `cmake --build build/quality-review-cmake --parallel 1 --clean-first` | Passed. |
| `ctest -N --test-dir build/quality-review-cmake` | Passed. |
| `make lint` | Passed. |
| `make test` | Passed. |

## Focused Test Evidence

| Test executable | Tests run | Failed | Skipped | Notes |
|---|---:|---:|---:|---|
| `test_qr` | 63 | 0 | 0 | Direct QR factorization owner remained stable after QR solve scenarios were split out. |
| `test_qr_solve` | 10 | 0 | 0 | Focused QR solve scenario owner passed square, overdetermined, analytical, rank-deficient, SuiteSparse, LU comparison, and synthetic tall cases. |
| `test_bicgstab` | 49 | 0 | 0 | Scalar BiCGSTAB owner remained stable after block cases were split out. |
| `test_bicgstab_block` | 12 | 0 | 0 | Focused block BiCGSTAB owner passed argument validation, multi-RHS, scalar equivalence, convergence aggregation, preconditioner, and error-propagation cases. |
| `test_cross_solver_oracle` | 1 | 0 | 0 | LU, Cholesky, QR, and CG agreed on the bounded generated-RHS SPD fixture. |

## Cross-Solver Pilot Evidence

| Solver | Relative residual | Max solution difference |
|---|---:|---:|
| LU | `2.052e-16` | `4.441e-16` |
| Cholesky | `2.163e-16` | `4.441e-16` |
| QR | `3.299e-16` | `4.441e-16` |
| CG | `2.490e-16` | `4.441e-16` |

The pilot remains bounded to one local SPD generated-RHS fixture and does not
create a broad direct/iterative parity claim.

## Source-List And Build Membership

| Check | Result |
|---|---|
| `make source-list-check` | Passed with 49 library sources. |
| CMake clean build | Passed. |
| CTest total | 57 registered tests. |
| `test_qr_solve` registration | Test #21. |
| `test_bicgstab_block` registration | Test #40. |
| `test_cross_solver_oracle` registration | Test #41. |

The CTest registration surface includes both split owners and the new pilot in
the expected adjacent positions.

## Full Quality Evidence

| Gate | Result |
|---|---|
| Formatting | `make format` passed. |
| Static analysis | `make lint` passed, including strict warnings, clang-tidy, and cppcheck. |
| Full Makefile test suite | `make test` passed all tests. |

## Skipped Supplemental Lanes

| Lane | Rationale |
|---|---|
| Windows CI/manual MSVC run | Not available locally; Day 13 validated local Make and CMake membership only. |
| Package/install validation | Sprint 120 did not change install, package, ABI, or public header surfaces. |
| Benchmark execution | Sprint 120 validation concerns test ownership and oracle behavior, not performance claims. |
| External Python or SuiteSparse oracle expansion beyond existing focused tests | Day 12 pilot intentionally used a local generated-RHS fixture and did not expand external-oracle scope. |
| Runtime sanitizer lanes | Not part of the existing required local Sprint 120 quality gate. |

## Residual Validation Risk

| Risk | Disposition |
|---|---|
| Platform-specific CTest count drift can still occur in CI if staged exclusions differ. | Document in closeout if CI reports a platform count mismatch. Local reviewed CMake count is 57. |
| The cross-solver pilot covers one compatible SPD fixture only. | Preserved as an explicit non-claim; broader external-oracle expansion remains future work. |
| Giant-test reduction is partial. | Remaining direct/iterative split candidates stay in the Sprint 120 residual queue. |
| Benchmark and package behavior are untested by this validation package. | Out of scope because Sprint 120 did not alter those surfaces. |

## Completion Check

| Criterion | Status |
|---|---|
| Source-list evidence captured. | Complete. |
| Focused direct evidence captured. | Complete. |
| Focused iterative evidence captured. | Complete. |
| Focused pilot evidence captured. | Complete. |
| CMake and CTest count evidence captured. | Complete. |
| Required full quality evidence captured. | Complete. |
| Skipped supplemental lanes and residual validation risks documented. | Complete. |
