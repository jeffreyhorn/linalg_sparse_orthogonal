# Sprint 124 Day 4 QR Minimum-Norm Behavior Contract

## Purpose

Day 4 defines the QR minimum-norm behavior contract required before Sprint 124
can accept or defer external minimum-norm oracle evidence. The contract keeps
QR solve, COLAMD/reordering, SVD-pseudoinverse, fallback, refinement,
rank-deficient, zero-row, and optional SuiteSparse behavior visible instead of
hiding those semantics behind a generic `minnorm` helper or external fixture.

This is a contract artifact only. No C source, header, Python helper, build,
CMake, CTest, workflow, public API, or public wording changes are made by Day
4.

## Inputs Reviewed

| Input | Contract Use |
| --- | --- |
| Sprint 124 Plan Day 4 | Requires minimum-norm coverage inventory, expected outputs, residual/norm policy, SVD-pseudoinverse boundary, helper ownership, optional SuiteSparse skip policy, and behavior contract. |
| Sprint 123 Day 7 QR minimum-norm/Q-economy decision | Defers minimum-norm external evidence until QR, COLAMD, SVD-pseudoinverse, fallback, refinement, and SuiteSparse ownership remain visible. |
| Sprint 123 Day 11 minimum-norm helper migration decision | Defers helper migration and defines behavior-specific helper naming and promotion gates. |
| `tests/test_qr_solve.c` | Owns focused QR solve minimum-norm scenario and bounded external QR solve fixtures. |
| `tests/test_colamd.c` | Owns broad minimum-norm scenarios: null args, known values, minimality, COLAMD, fallback, rank-deficient, square, 1xn, refinement, zero-row, QR-vs-pinv, and SuiteSparse submatrix. |
| `tests/test_svd.c` | Owns SVD pseudoinverse and Moore-Penrose behavior, including underdetermined pseudoinverse minimum-norm evidence. |
| `tests/test_qr_helpers.h` and `tests/test_solver_helpers.h` | Potential future helper locations, but not current owners of minimum-norm assertions. |

## Current Minimum-Norm Behavior Matrix

| Behavior Class | Current Owner | Evidence Summary | Sprint 124 Contract |
| --- | --- | --- | --- |
| Focused 2x4 QR solve | `tests/test_qr_solve.c` | `test_qr_solve_minnorm_underdetermined_known_solution` checks exact `[0.5, 0.5, 0.5, 0.5]`, residual, and norm. | Primary visible QR solve minimum-norm smoke owner. |
| Known 2x4 minimum-norm | `tests/test_colamd.c` | `test_minnorm_2x4_known` repeats exact known solution with tighter owner-local tolerance. | Broad minimum-norm owner; do not collapse into QR solve fixture. |
| Minimality comparison | `tests/test_colamd.c` | `test_minnorm_is_minimal` compares against a valid non-minimum-norm solution. | Norm-optimality evidence remains scenario-local. |
| Larger underdetermined systems | `tests/test_colamd.c` | `test_minnorm_3x6`, `test_minnorm_5x10`, and `test_minnorm_1xn` check residuals and norms across shapes. | External evidence must name which shape behavior it asserts. |
| COLAMD/reorder behavior | `tests/test_colamd.c` | `test_minnorm_with_colamd` checks minimum-norm solve with COLAMD options. | Reorder ownership must remain explicit; not covered by a generic minnorm fixture. |
| Overdetermined and square fallback | `tests/test_colamd.c` | `test_minnorm_fallback_overdetermined` and `test_minnorm_square` check fallback to ordinary QR solve behavior. | Fallback behavior is separate from underdetermined minimum-norm. |
| Rank-deficient minimum-norm | `tests/test_colamd.c` | `test_minnorm_rank_deficient` checks consistent rank-deficient minimum-norm behavior. | Depends on rank policy and norm policy; not owned by Day 3 rank-only evidence. |
| Refinement | `tests/test_colamd.c` | `test_refine_minnorm` and `test_refine_minnorm_null` check refinement improvement and argument validation. | Refinement is a separate behavior owner with residual-improvement semantics. |
| Zero-row behavior | `tests/test_colamd.c` | `test_minnorm_zero_row` checks consistent zero-row constraints. | Zero-row evidence must remain distinct from generic underdetermined evidence. |
| QR-vs-SVD pseudoinverse | `tests/test_colamd.c` | `test_minnorm_vs_pinv` compares QR minimum-norm against SVD pseudoinverse for a bounded case. | This is a cross-check, not an external oracle claim. |
| SVD pseudoinverse minimum-norm | `tests/test_svd.c` | Pseudoinverse and Moore-Penrose tests own SVD-side minimum-norm behavior. | SVD is not automatically the oracle for every QR minimum-norm claim. |
| SuiteSparse submatrix | `tests/test_colamd.c` | `test_minnorm_ss_submatrix` smokes west0067 submatrix minimum-norm behavior when data/support is available. | Optional corpus smoke; not broad SuiteSparse support or platform parity. |

## Behavior-Specific Acceptance Criteria

| Candidate Behavior | Required Expected Outputs | Required Comparisons | Explicit Non-Claims |
| --- | --- | --- | --- |
| Exact small underdetermined fixture | Matrix, RHS, expected solution, expected residual, expected solution norm | `||A*x-b||`, max solution difference, `||x||_2`, optional comparison to named alternate solution | no broad global optimality beyond fixture |
| COLAMD/reordered minimum-norm | Matrix, RHS, ordering options, expected residual/norm behavior | residual, norm, and option path evidence | no COLAMD superiority or broad reorder parity |
| Overdetermined/square fallback | Matrix, RHS, fallback path, expected ordinary QR solve result | solution/residual against QR solve behavior | no underdetermined minimum-norm claim |
| Rank-deficient minimum-norm | Rank model, RHS consistency, expected norm behavior, rank threshold if asserted | residual, norm, optional expected values | no nullspace basis or global rank policy claim |
| Refinement | Initial solution source, iteration budget, before/after residual | residual non-increase or bounded improvement | no convergence-rate or superiority claim |
| SVD-pseudoinverse cross-check | Named QR fixture, SVD tolerance, expected comparison scope | QR solution vs SVD-pinv solution for the same fixture | no SVD-as-global-oracle claim |
| SuiteSparse submatrix | Corpus matrix, submatrix extraction, skip policy, expected smoke behavior | residual and bounded norm check under availability gates | no SuiteSparse-wide or platform parity claim |

## Residual and Norm Comparison Policy

| Quantity | Policy |
| --- | --- |
| Residual | Report `||A*x-b||` or fixture-specific residual. Tolerance must remain at the scenario call site or artifact. |
| Solution values | Compare exact expected values only for tiny fixtures with mathematically derived solutions. |
| Solution norm | Required for any claim that the solution is minimum-norm; compare against expected norm or a named alternate valid solution. |
| Relative vs absolute tolerance | Use absolute tolerances for tiny exact fixtures; use residual/norm ratios only when the fixture defines scale-sensitive behavior. |
| Rank | If rank is part of the claim, use the Day 2/Day 3 rank policy and pin the threshold. |
| Failure diagnostics | Identify fixture key, behavior class, residual, norm, expected values if any, optional path, and whether the failure is solve, norm, residual, rank, reorder, refinement, SVD-pinv, or corpus availability. |

## Oracle, Fallback, and Non-Claim Boundary

| Surface | May Act As | Boundary |
| --- | --- | --- |
| Python standard-library external helper | Bounded external reference for a tiny accepted minimum-norm fixture if Day 5 designs one. | Must not depend on NumPy, SciPy, LAPACK, BLAS, SuiteSparse, or external packages. |
| SVD pseudoinverse | Bounded cross-check when explicitly named. | Not a global QR oracle and not a broad dense-library parity claim. |
| Ordinary QR solve | Fallback behavior for square/overdetermined paths. | Not evidence for underdetermined minimum-norm optimality. |
| COLAMD/reorder options | Scenario behavior under `test_colamd.c`. | Not a generic minimum-norm property and not reorder superiority. |
| SuiteSparse submatrix | Optional corpus smoke. | Not SuiteSparse-wide minimum-norm support or equal platform support. |
| Deterministic internal fixtures | Baseline local behavior evidence. | Not external oracle parity unless a separate external reference is accepted. |

## Helper Ownership Notes

| Helper Surface | Day 4 Rule |
| --- | --- |
| `tests/test_qr_solve.c` | May own focused QR solve external minimum-norm evidence if Day 5 accepts a fixture. |
| `tests/test_colamd.c` | Continues owning broad minimum-norm, COLAMD, fallback, refinement, rank-deficient, zero-row, QR-vs-pinv, and SuiteSparse scenarios. |
| `tests/test_svd.c` | Continues owning SVD pseudoinverse and Moore-Penrose semantics. |
| `tests/test_qr_helpers.h` | May host future fixture builders or measurements only with behavior-specific names and caller-visible tolerances. |
| `tests/test_solver_helpers.h` | May host external-process plumbing, not minimum-norm behavior semantics. |
| Generic `assert_minnorm` helpers | Not allowed for Sprint 124 because they hide behavior-specific ownership. |

## Optional Backend and SuiteSparse Skip Policy

| Scenario | Policy |
| --- | --- |
| Missing Python external helper | Preserve existing external-reference skip behavior where helper availability is optional. |
| Windows external helper lanes | Preserve explicit Windows skip unless a platform sprint promotes the reviewed surface. |
| Optional SuiteSparse data | Skip only through the existing corpus availability conventions; record skip as corpus availability, not behavior success. |
| SuiteSparse minimum-norm failure | Treat expected unsupported/failure paths as scenario-specific outcomes, not broad package/platform claims. |
| Helper `ERROR` output | Treat as test failure for accepted external helper lanes, not skip. |

## Day 5 Decision Criteria

Day 5 should evaluate QR minimum-norm external evidence in this order:

1. Treat existing deterministic QR, COLAMD, and SVD minimum-norm evidence as
   owned and fenced.
2. Prefer a tiny exact underdetermined fixture only if the external helper can
   produce expected solution, residual, and norm without dense-library
   dependencies.
3. Reject or defer any fixture that hides COLAMD, fallback, refinement,
   rank-deficient, SVD-pseudoinverse, or SuiteSparse behavior under a generic
   minimum-norm label.
4. If accepted, define fixture key, matrix, RHS, expected solution, expected
   residual, expected norm, output protocol, affected owners, diagnostics, and
   validation commands.
5. If deferred, name future owner, dependency, and promotion gate.

## Non-Claim Register

Day 4 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  broad external dense-library parity;
- broad QR minimum-norm external oracle parity;
- global minimum-norm optimality beyond named fixtures;
- SVD-pseudoinverse as a global QR oracle;
- COLAMD, reorder, fallback, refinement, or SuiteSparse superiority;
- rank-deficient solve, nullspace, Q-basis, economy-mode, sparse-mode, or
  backend parity;
- package, ABI, platform, public API, CMake, Makefile, CI, or CTest expansion;
- performance, scalability, memory behavior, or state-of-the-art behavior.

## Validation Notes

Day 4 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_124`

No `.c` or `.h` files changed for Day 4, so the full `make format && make
lint && make test` gate is not required for this day. The branch already passed
the full gate after Day 3's code changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 2 has behavior-specific acceptance criteria. | Complete | See behavior-specific acceptance criteria and Day 5 decision criteria. |
| Minimum-norm semantics are not hidden behind generic helper names. | Complete | See helper ownership notes. |
| Optional backend behavior is explicitly fenced. | Complete | See optional backend and SuiteSparse skip policy. |
