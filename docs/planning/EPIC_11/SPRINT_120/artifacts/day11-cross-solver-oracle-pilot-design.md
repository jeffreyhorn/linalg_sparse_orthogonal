# Sprint 120 Day 11 Cross-Solver Oracle Pilot Design

## Purpose

Day 11 designs one bounded cross-solver oracle pilot for Day 12
implementation. The pilot proves that a single compatible generated-RHS
fixture can compare a small direct/iterative solver set without implying broad
direct/iterative parity, external-oracle completeness, platform support, or
performance claims.

This artifact is design-only. No C source, header, Makefile, CMake, or CTest
metadata changes are made by Day 11.

## Scope

| Field | Value |
|---|---|
| Sprint/day | Sprint 120 Day 11 |
| Artifact owner | Sprint 120 cross-solver oracle pilot |
| Solver or behavior family | Bounded direct/iterative generated-RHS comparison |
| Touched surfaces on Day 11 | Planning artifact and working notes only |
| Planned Day 12 test owner | `tests/test_cross_solver_oracle.c` |
| Explicitly out of scope | SuiteSparse fixtures, external Python references, package/install lanes, benchmark timing, public API expansion, broad direct/iterative parity, matrix-free callbacks, block solvers, preconditioner comparisons, LDLT CSC lifecycle, and platform-specific claims. |

## Selected Pilot

| Field | Decision |
|---|---|
| Pilot name | Small SPD generated-RHS cross-solver oracle |
| Fixture | Dense-ish SPD tridiagonal matrix with `n = 8`, diagonal `4.0`, off-diagonal `-1.0`. |
| RHS model | Generate `x_exact`, then compute `b = A * x_exact`. |
| Solver set | LU, Cholesky, QR, and CG. |
| Primary oracle | Known generated solution plus solver residuals. |
| Secondary comparison | Pairwise solution agreement against `x_exact`, not pairwise solver superiority. |
| Failure model | All selected solvers are expected to return success on the selected SPD square fixture. |
| Implementation strategy | Add a new focused C test executable so the pilot does not expand an existing giant test owner. |

## Baseline

| Baseline item | Current value |
|---|---|
| Existing direct proof owners | QR solve owner in `tests/test_qr_solve.c`; LU owner in `tests/test_sparse_lu.c`; Cholesky owner in `tests/test_cholesky.c`. |
| Existing iterative proof owners | CG and GMRES owner in `tests/test_iterative.c`; BiCGSTAB and block BiCGSTAB are now split across `tests/test_bicgstab.c` and `tests/test_bicgstab_block.c`. |
| Existing oracle/reference style | Generated RHS, direct cross-solver comparisons, SuiteSparse comparisons, and external dense references exist in separate local owners. |
| Current product truth references | Sprint 118 product-truth map and Sprint 120 Day 4 helper architecture prohibit broad parity claims from narrow proof lanes. |
| Current non-claims | No broad direct/iterative parity, no external-oracle completeness, no package/platform claim, no performance claim, and no public API expansion. |

## Solver And Fixture Scope

| Solver | API plan | Why included | Explicit exclusions |
|---|---|---|---|
| LU | Copy matrix, call `sparse_lu_factor(..., SPARSE_PIVOT_PARTIAL, 1e-12)`, then `sparse_lu_solve`. | Mature direct baseline for square nonsingular systems. | Complete pivoting, transpose solve, condition estimate, iterative refinement. |
| Cholesky | Copy matrix, call `sparse_cholesky_factor`, then `sparse_cholesky_solve`. | SPD-specific direct baseline on the selected fixture. | CSC/supernodal paths, reorder options, non-SPD rejection. |
| QR | Factor original-compatible copy with `sparse_qr_factor`, then `sparse_qr_solve`. | Existing Day 7 QR solve owner already supports generated-RHS true-residual checks. | Least-squares, rank-deficient, min-norm, SuiteSparse solve claims. |
| CG | Call `sparse_solve_cg` with local options. | Iterative SPD-compatible solver for generated-RHS comparison. | GMRES, BiCGSTAB, MINRES, block solvers, preconditioners, stagnation/progress history. |

## Fixture Taxonomy

| Fixture | Symmetry | Definiteness | Rank | Conditioning/scaling | Sparsity pattern | Expected behavior |
|---|---|---|---|---|---|---|
| `spd_tridiag_n8_diag4_offneg1` | Symmetric | SPD | Full rank | Well conditioned, unscaled | Tridiagonal | LU, Cholesky, QR, and CG all solve successfully and recover generated `x_exact` within local tolerances. |

## Matrix And RHS Construction

- Matrix source: local static helper in `tests/test_cross_solver_oracle.c`.
- Dimensions: `8 x 8`.
- Nonzeros or sparsity: tridiagonal with diagonal `4.0` and symmetric
  off-diagonal `-1.0`.
- Exact solution: deterministic vector such as
  `x_exact[i] = 1.0 + 0.25 * i`.
- RHS construction: local `compute_rhs(A, x_exact, b)` helper.
- Ordering/reorder/backend/runtime settings: none beyond solver defaults and
  explicit LU partial pivoting.

The matrix builder and RHS helper should stay local to the pilot on Day 12.
They should not be promoted to a shared helper from a single pilot.

## Oracle Or Reference Source

| Oracle/reference | Invocation | Trust boundary | Skip/error handling |
|---|---|---|---|
| Generated exact solution | Build `x_exact`, compute `b = A * x_exact`, solve with each selected solver. | Trusts local matrix construction and direct `A*x` RHS generation only. | No skip expected. Allocation or solver failure is a test failure. |
| Residual measurement | Use `tf_relative_residual_l2(A, b, x, n, HUGE_VAL)`. | Measurement helper only; caller owns tolerances. | Non-finite or tolerance violation is a test failure. |
| Solution max-difference | Local helper comparing `x_solver` to `x_exact`. | Bounded same-fixture comparison only. | Tolerance violation is a test failure. |

## Tolerance And Acceptance Model

| Metric | Tolerance | Rationale |
|---|---:|---|
| LU relative residual | `1e-10` | Small well-conditioned SPD fixture; direct solve should be near machine precision. |
| Cholesky relative residual | `1e-10` | SPD direct solve should recover the generated RHS tightly. |
| QR relative residual | `1e-10` | Square full-rank QR solve should match generated RHS tightly. |
| CG relative residual | `1e-10` | CG is run on a small SPD fixture with a tight tolerance and sufficient iteration budget. |
| Max `|x_solver - x_exact|` | `1e-8` | Allows minor algorithmic differences while keeping the generated solution contract tight. |
| CG convergence flag | `true` | Selected fixture is SPD and well conditioned. |
| CG iteration budget | `max_iter = 100`, `tol = 1e-12` | Budget exceeds `n` and avoids turning iteration count into a performance claim. |

The pilot must not assert that one solver is faster, more stable, or broadly
equivalent to another. It only asserts that the selected solvers agree on one
compatible generated-RHS fixture.

## Planned File And Build Impact

| Surface | Day 12 planned change |
|---|---|
| `tests/test_cross_solver_oracle.c` | Add new focused pilot test owner. |
| Makefile | Add `$(TESTDIR)/test_cross_solver_oracle.c` to `TEST_SRCS`, preferably near adjacent solver comparison tests after `test_bicgstab_block.c` or before `test_stagnation.c`. |
| CMake | Add `add_sparse_test(test_cross_solver_oracle)`, preferably after `add_sparse_test(test_bicgstab_block)` or before `add_sparse_test(test_stagnation)`. |
| Source-list check | `make source-list-check` must continue to pass. |
| CTest count | Current Sprint 120 count after Day 10 is 56. Adding the pilot executable should increase reviewed CTest registration to 57. |
| Public docs | No public claim wording expected. |

## Planned Test Shape

Day 12 should add one or two focused tests:

| Test | Purpose |
|---|---|
| `test_spd_generated_rhs_lu_chol_qr_cg_agree` | Main pilot: solve one SPD generated-RHS fixture with LU, Cholesky, QR, and CG; check residuals, generated solution recovery, and CG convergence. |
| `test_cross_solver_oracle_rejects_null_fixture_helpers` | Optional only if local helper shape needs argument validation; otherwise skip to keep the pilot small. |

The preferred Day 12 implementation is a single main pilot test plus local
helper functions. Avoid adding a general cross-solver helper header.

## Focused Validation Checklist

Day 12 will modify `.c` and build metadata, so it must run:

1. `make format`
2. `make build/test_cross_solver_oracle && ./build/test_cross_solver_oracle`
3. `make source-list-check`
4. `cmake -S . -B build/quality-review-cmake`
5. `cmake --build build/quality-review-cmake --parallel 1 --clean-first`
6. `ctest -N --test-dir build/quality-review-cmake`
7. `make lint`
8. `make test`
9. `git diff --check`
10. Focused trailing-whitespace scan over
    `docs/planning/EPIC_11/SPRINT_120`,
    `tests/test_cross_solver_oracle.c`, `Makefile`, and `CMakeLists.txt`.

## Unsupported Or Expected-Failure Cases

| Case | Disposition | Reason |
|---|---|---|
| Nonsymmetric fixtures | Out of scope | Cholesky and CG require SPD-compatible input for this pilot. |
| Indefinite/KKT fixtures | Out of scope | Would pull in LDLT/MINRES interpretation and broaden scope. |
| SuiteSparse matrices | Out of scope | Matrix-specific tolerances and runtime/platform behavior are not needed for the pilot. |
| External dense references | Out of scope | Day 11 selects local generated RHS only. |
| Preconditioned iterative solvers | Out of scope | Would add callback/preconditioner ownership beyond the pilot. |
| Block solvers | Out of scope | Already handled separately for block BiCGSTAB; block MINRES is deferred. |
| Performance or iteration comparison | Out of scope | Iteration counts and wall time are not product claims in this pilot. |

## Rollback Checklist

If Day 12 validation fails:

1. Remove `tests/test_cross_solver_oracle.c`.
2. Remove `$(TESTDIR)/test_cross_solver_oracle.c` from Makefile `TEST_SRCS`.
3. Remove `add_sparse_test(test_cross_solver_oracle)` from
   `CMakeLists.txt`.
4. Re-run `make source-list-check`.
5. Re-run CMake configure/build and `ctest -N` if CMake was touched.
6. Re-run `make format && make lint && make test` because C/build surfaces
   were modified.
7. Record the failed pilot and reason in Sprint 120 residual closeout.

## Drift Check

| Public/support surface | Impact | Action |
|---|---|---|
| README | None | Do not update. |
| Solver-selection docs | None | Do not update. |
| Examples/tutorial | None | Do not update. |
| Benchmark/performance wording | None | Do not update. |
| Package/platform docs | None | Do not update. |

## Non-Claims Preserved

- The pilot does not prove broad direct/iterative parity.
- The pilot does not prove external-oracle completeness.
- The pilot does not prove SuiteSparse coverage for all selected solvers.
- The pilot does not prove package, platform, ABI, or install support.
- The pilot does not prove performance, scalability, or state-of-the-art
  behavior.
- The pilot does not add or change public API.

## Residual Handoff

| Residual | Next owner | Evidence link |
|---|---|---|
| Add the bounded pilot implementation | Sprint 120 Day 12 | This artifact. |
| GMRES SuiteSparse/restart/right-preconditioner cross-solver split | Future iterative owner cleanup | Day 5 deferred queue. |
| MINRES LDLT/GMRES comparison split | Future iterative owner cleanup | Day 5 deferred queue. |
| LDLT cross-backend/cross-solver split | Future direct owner cleanup | Day 5 deferred queue. |
| External dense-reference oracle expansion | Future external oracle sprint | Day 2 and Day 4 external-reference notes. |

## Completion Check

| Criterion | Status |
|---|---|
| Fixture taxonomy is recorded. | Complete. |
| Oracle or reference trust boundary is recorded. | Complete. |
| Tolerances are explicit. | Complete. |
| Unsupported cases are explicit. | Complete. |
| Validation commands are recorded. | Complete. |
| Drift and non-claims are recorded. | Complete. |
| Residual handoff is recorded. | Complete. |
