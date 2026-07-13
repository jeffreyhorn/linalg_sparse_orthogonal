# Sprint 121 Day 5: Helper Extraction Plan

## Purpose

Define exact helper extraction boundaries for Sprint 121 SVD and QR proof work
before any `.c` or `.h` edits. This plan selects the helpers that Days 6-7 may
extract, names the behavior owners that must keep assertion semantics visible,
records build/CTest impact, and defines validation and rollback steps.

This is a design artifact only. No C source, header, build, CMake, workflow, or
test membership changes are made by Day 5.

## Extraction Strategy

- Add test-only helper headers rather than new test executables.
- Keep CTest membership unchanged: `test_svd`, `test_qr`, `test_qr_solve`, and
  `test_colamd` remain the existing proof executables.
- Keep Makefile and CMake source lists unchanged unless a later day chooses a
  new `.c` helper file, which this plan does not recommend.
- Move only measurement, deterministic fixture construction, and mechanical
  allocation cleanup into helpers.
- Leave tolerances, expected rank, expected residual, expected reconstruction,
  expected orthogonality, expected error code, skip policy, and non-claim
  wording in named scenario tests.

## File Boundary Plan

| Planned file | Change type | Consumers | Responsibility | Build membership impact |
|---|---|---|---|---|
| `tests/test_svd_helpers.h` | New test-only header on Day 6 | `tests/test_svd.c` | Full-SVD reconstruction measurement, SVD orthogonality measurement, pseudoinverse identity measurement, dense/sparse low-rank residual measurement, selected SVD fixture builders | Header include only; no Makefile/CMake/CTest count change. |
| `tests/test_qr_helpers.h` | New test-only header on Day 7 | `tests/test_qr.c`, `tests/test_qr_solve.c`; optionally `tests/test_colamd.c` only for minimum-norm helpers | QR reconstruction measurement, QR residual measurement, exact RHS builder, duplicate-column/near-dependent builders, minimum-norm norm/residual measurement | Header include only; no Makefile/CMake/CTest count change. |
| `tests/test_svd_partial_helpers.h` | Existing header; avoid broad rewrite | `tests/test_svd.c` | Keep current partial-SVD test bodies unless Day 10 selects a partial-vector helper split | No membership change. |
| `tests/test_solver_helpers.h` | Existing broad solver helper; no Sprint 121 edits planned | Existing solver tests | Keep generic vector norm and broad residual helpers as-is | No membership change. |
| `tests/test_svd.c` | Day 6 include and call-site edits | `test_svd` | Scenario assertions, tolerances, skip behavior, SVD/partial-SVD/pinv/low-rank claim boundaries | Existing test executable. |
| `tests/test_qr.c` | Day 7 include and call-site edits | `test_qr` | QR factorization, reconstruction, rank, nullspace, economy/sparse-mode scenario assertions | Existing test executable. |
| `tests/test_qr_solve.c` | Day 7 include and call-site edits | `test_qr_solve` | QR solve, least-squares, residual, QR-vs-LU scenario assertions | Existing test executable. |
| `tests/test_colamd.c` | Optional Day 9 include for minimum-norm helper reuse only | `test_colamd` | Minimum-norm scenario assertions and COLAMD/reorder proof ownership | Existing test executable. |

## SVD Helper Extraction Checklist

| Helper candidate | Select? | Planned helper name | Source owner today | Behavior owner after extraction | Tolerance owner |
|---|---|---|---|---|---|
| Full-SVD max reconstruction error | Yes | `tf_svd_reconstruction_max_error` | `tests/test_svd.c::svd_reconstruction_max_error` | Named full-SVD tests such as `test_svd_with_uv`, `test_svd_rank1_uv`, `test_svd_rank2_dense`, `test_svd_wide_5x10_uv` | Calling test. |
| Full-SVD relative Frobenius reconstruction | Yes | `tf_svd_reconstruction_rel_frobenius` | `tests/test_svd.c::svd_reconstruction_rel_frobenius` | Full/economy reconstruction tests, including full-mode leading-dimension owners | Calling test. |
| Dense matrix orthogonality | Yes | `tf_dense_column_orthogonality_error` | `tests/test_svd.c::orthogonality_error` | SVD U checks and potential QR economy checks only if later adopted explicitly | Calling test. |
| Vt row orthogonality | Yes | `tf_svd_vt_row_orthogonality_error` | `tests/test_svd.c::svd_vt_row_orthogonality_error` | SVD Vt/full-mode tests | Calling test; helper requires rows/cols/leading dimension. |
| First Moore-Penrose identity | Yes | `tf_svd_pinv_first_moore_penrose_error` | `tests/test_svd.c::svd_pinv_first_moore_penrose_error` | Pseudoinverse identity tests | Calling test. |
| Dense low-rank Frobenius error | Yes | `tf_svd_dense_lowrank_frobenius_error` | `tests/test_svd.c::svd_dense_lowrank_frobenius_error` | Dense low-rank tests | Calling test. |
| Sparse-vs-dense low-rank difference | Yes | `tf_svd_sparse_dense_frobenius_diff` | `tests/test_svd.c::svd_sparse_dense_frobenius_diff` | Sparse low-rank tests | Calling test. |
| Sparse-vs-sparse relative Frobenius | Yes | `tf_svd_sparse_sparse_rel_frobenius_diff` | `tests/test_svd.c::svd_sparse_sparse_rel_frobenius_diff` | Low-rank env-on/off mode-equivalence tests | Calling test. |
| SVD rank-1 row-progression builder | Yes | `tf_svd_make_rank1_row_progression` | `tests/test_svd.c::make_svd_rank1_row_progression` | Rank-1 SVD/low-rank tests | Calling test owns expected rank and residual. |
| Duplicate-column rank-deficient builder | Yes | `tf_svd_make_rank_deficient_colpair_5x4` | `tests/test_svd.c::make_svd_rank_deficient_colpair_5x4` | SVD rank and SVD-vs-QR rank tests | Calling test owns rank threshold. |
| Partial-SVD vector residual helpers | Defer to Day 10 | Not selected on Day 6 | `tests/test_svd_partial_helpers.h` | Partial-SVD vector tests | Partial-SVD tests; looser tolerances must remain local. |
| Bidiagonal iteration helpers | Defer | Not selected | low-level SVD tests | Bidiagonal tests | Bidiagonal tests; low-level owner stays local. |

## QR Helper Extraction Checklist

| Helper candidate | Select? | Planned helper name | Source owner today | Behavior owner after extraction | Tolerance owner |
|---|---|---|---|---|---|
| QR reconstruction error for `A*P = Q*R` | Yes | `tf_qr_reconstruction_max_error` | `tests/test_qr.c::qr_reconstruction_error`, `tests/test_qr_solve.c::qr_solve_reconstruction_error` | Reconstruction tests in `test_qr` and SuiteSparse reconstruction tests in `test_qr_solve` | Calling test. |
| QR reconstruction assertion wrapper | No | Do not extract assertion | `assert_qr_reconstruction_below`, `assert_qr_solve_reconstruction_below` | Scenario tests keep assertion messages and tolerances | Calling test. |
| QR relative residual measurement | Yes | `tf_qr_relative_residual_l2` | `tests/test_qr.c::compute_rel_residual`, `tests/test_qr_solve.c::qr_solve_rel_residual` | QR solve and least-squares tests | Calling test chooses absolute/relative target. |
| QR true-residual assertion wrapper | No | Do not extract assertion | `assert_qr_solve_true_residual_below` | `test_qr_solve.c` scenario tests | Calling test. |
| Exact RHS builder | Yes | `tf_qr_make_exact_rhs` | `make_qr_exact_rhs`, `make_qr_solve_exact_rhs` | Generated-RHS QR solve, refinement, and SuiteSparse tests | Calling test chooses expected residual. |
| Duplicate-column builder | Yes | `tf_qr_make_duplicate_column_4x3` | duplicate builders in `test_qr.c` and `test_qr_solve.c` | QR rank, nullspace, and rank-deficient solve tests | Calling test owns rank and residual. |
| Near-duplicate builder | Yes | `tf_qr_make_near_duplicate_4x3` | `tests/test_qr.c::make_qr_near_duplicate_4x3` | Near-rank-deficient QR tests | Calling test owns perturbation and rank relation. |
| Tall diagonal-dominant builder | Yes | `tf_qr_make_tall_diagonal_dominant` | `tests/test_qr.c::make_qr_tall_diagonal_dominant` | Economy, refinement, and tall solve tests | Calling test owns residual and rank. |
| Insert-or-free utility | Yes, internal to QR helper header | `tf_qr_insert_or_free` | local insert helpers in QR files | Fixture builders only | Builder returns allocation status; tests assert. |
| Minimum-norm norm/residual helpers | Defer to Day 9 | Not selected on Day 7 unless Day 9 needs it | `tests/test_colamd.c` minnorm helpers | Minimum-norm tests | `test_colamd.c` or future QR owner keeps assertions. |
| Economy/sparse-mode comparison assertion | No | Do not extract assertion | mode tests in `test_qr.c` | Mode-equivalence tests | Calling test keeps backend labels and slack. |

## Naming Rules

- Prefix test helper functions with `tf_`.
- Include solver family in helper names when semantics are solver-specific:
  `tf_svd_*` and `tf_qr_*`.
- Include metric names in measurement helpers: `reconstruction_max_error`,
  `relative_residual_l2`, `orthogonality_error`, `frobenius_diff`.
- Include fixture semantics in builders: `rank1_row_progression`,
  `rank_deficient_colpair`, `duplicate_column`, `near_duplicate`,
  `tall_diagonal_dominant`.
- Do not name helpers `oracle`, `parity`, `validated`, or `state_of_art`.

## Build And CTest Impact

| Planned implementation | Makefile impact | CMake impact | CTest count impact | Quality checks |
|---|---|---|---|---|
| Add `tests/test_svd_helpers.h` and include from `tests/test_svd.c` | None if header-only | None if header-only | None; `test_svd` remains one executable | Because `.h`/`.c` changes: `make format && make lint && make test`. |
| Add `tests/test_qr_helpers.h` and include from `tests/test_qr.c` / `tests/test_qr_solve.c` | None if header-only | None if header-only | None; `test_qr` and `test_qr_solve` remain existing executables | Because `.h`/`.c` changes: `make format && make lint && make test`. |
| Optional include from `tests/test_colamd.c` | None if header-only | None if header-only | None; `test_colamd` remains existing executable | Because `.c` changes: `make format && make lint && make test`. |
| Any new `.c` helper file | Not recommended; would require `TEST_SRCS` or source-list review | Would require explicit CMake/source membership if executable changes | Possible count or link impact if new test executable is added | Stop and document before implementation. |

## Focused Validation Commands

Before the required full quality chain, run focused proof commands for touched
areas:

| Day | If touched | Focused command | Expected result |
|---|---|---|---|
| Day 6 | `tests/test_svd.c` / `tests/test_svd_helpers.h` | `make build/test_svd && ./build/test_svd` | Existing SVD suite passes with no CTest count change. |
| Day 7 | `tests/test_qr.c` / `tests/test_qr_helpers.h` | `make build/test_qr && ./build/test_qr` | Existing QR suite passes with no CTest count change. |
| Day 7 | `tests/test_qr_solve.c` / `tests/test_qr_helpers.h` | `make build/test_qr_solve && ./build/test_qr_solve` | Existing QR solve suite passes with no CTest count change. |
| Day 9 optional | `tests/test_colamd.c` minimum-norm helper use | `make build/test_colamd && ./build/test_colamd` | Existing COLAMD/minnorm suite passes with no CTest count change. |
| Any Makefile/CMake source-list edit | Build membership touched | `make source-list-check`; `cmake -S . -B build/quality-review-cmake`; `ctest --test-dir build/quality-review-cmake -N` | Source-list and CTest registration remain intentional. |
| Any `.c`/`.h` edit | Required final chain | `make format && make lint && make test` | All pass before proceeding. |

## Rollback Instructions

| Change | Rollback path |
|---|---|
| New `tests/test_svd_helpers.h` causes focused SVD failure | Remove the include from `tests/test_svd.c`, inline or restore the original static helpers in `tests/test_svd.c`, delete `tests/test_svd_helpers.h`, rerun `make build/test_svd && ./build/test_svd`. |
| New `tests/test_qr_helpers.h` causes focused QR failure | Remove the include from affected QR test file, restore the original static helper(s), delete unused helper functions, rerun the affected focused QR command. |
| Helper changes hide tolerance or assertion meaning | Roll back the helper extraction for that behavior and keep only measurement helpers; leave assertions and tolerances in scenario tests. |
| Build membership changes unexpectedly | Revert Makefile/CMake membership edits and keep helper header-only; rerun `make source-list-check` and CMake `ctest -N` proof if membership was touched. |
| Minimum-norm ownership becomes unclear | Leave minimum-norm helpers in `tests/test_colamd.c` for Sprint 121 and document residual handoff to Day 9 or closeout. |

## Deferred Candidates

| Deferred item | Reason | Proposed owner |
|---|---|---|
| Partial-SVD vector helper split | Uses looser residual/orthogonality semantics and internal full-SVD references; should be expanded with Day 10 partial-SVD evidence. | Day 10. |
| Bidiagonal SVD helper split | Low-level algorithm owner is already localized and not a high-risk duplication target. | Defer unless future SVD source split needs it. |
| Minimum-norm helper movement out of `test_colamd.c` | Ownership is historically tied to COLAMD/reordering tests; moving it now could obscure reviewed membership. | Day 9 optional. |
| SuiteSparse loader wrapper | Skip/error policy must remain scenario-local for bounded smoke tests. | Scenario tests. |
| Assertion wrappers | Assertions encode tolerance and non-claim semantics; extracting them would hide behavior ownership. | Do not extract in Sprint 121. |

## Completion Criteria Status

| Criterion | Status |
|---|---|
| Item 3 extraction can proceed from exact boundaries | Complete: planned files, consumers, and header-only membership impact are recorded. |
| Every selected helper has a named behavior owner | Complete: SVD and QR checklists identify the scenario owner and tolerance owner for each selected helper. |
| Validation and rollback commands are recorded before implementation | Complete: focused commands, full quality chain, CMake/source-list trigger, and rollback paths are documented. |

## Non-Claims

This plan does not claim broad LAPACK, SciPy, SuiteSparse, PETSc, Trilinos,
Eigen, platform, packaging, benchmark, or state-of-the-art parity. It only
defines bounded test-helper extraction steps for existing SVD and QR proof
owners.
