# Sprint 125 Day 10 Minimum-Norm Behavior Owner Map

## Purpose

Split Sprint 125 minimum-norm work into behavior-specific owners before Days
11-12 decide whether to add or explicitly defer additional evidence.

This artifact is policy-only. It does not add tests, move helpers, change
`sparse_qr_solve_minnorm`, change SVD pseudoinverse behavior, or update public
solver claims.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 124 Day 4 QR minimum-norm behavior contract | Defines owner boundaries across QR solve, COLAMD, fallback, rank-deficient, refinement, SVD-pseudoinverse, and SuiteSparse paths. |
| Sprint 124 Day 5 QR minimum-norm decision | Accepted `qr_underdetermined_minnorm_2x4` and deferred broader minimum-norm evidence. |
| Sprint 124 Day 12 helper ownership follow-through | Rejects generic minimum-norm helpers and keeps scenario assertions visible. |
| `tests/test_qr_solve.c` | Owns focused QR solve and external-reference exact 2x4 minimum-norm evidence. |
| `tests/test_colamd.c` | Owns broad QR minimum-norm behavior, COLAMD, fallback, rank-deficient, refinement, zero-row, QR-vs-pinv, and SuiteSparse submatrix scenarios. |
| `tests/test_svd.c` | Owns SVD pseudoinverse and Moore-Penrose behavior. |
| `tests/qr_external_dense_reference.py` | Owns bounded standard-library QR fixture output protocols. |
| Sprint 125 Days 8-9 SuiteSparse artifacts | Define optional-corpus support tiers and deferral behavior. |

## Scenario Inventory

| Scenario | Current owner | Existing evidence | Day 10 disposition |
| --- | --- | --- | --- |
| Exact focused underdetermined 2x4 | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_underdetermined_minnorm_2x4` compares solution, residual, and norm against standard-library reference. | Complete; do not duplicate. |
| Internal 2x4 known minimum-norm | `tests/test_colamd.c` | `test_minnorm_2x4_known` checks exact values and norm. | Control only; not a new external lane. |
| Minimality comparison | `tests/test_colamd.c` | `test_minnorm_is_minimal` compares against a named non-minimum valid solution. | Candidate support evidence for Day 11 only if behavior remains local. |
| Larger underdetermined shapes | `tests/test_colamd.c` | `test_minnorm_3x6`, `test_minnorm_5x10`, and `test_minnorm_1xn`. | Candidate for Day 11 only with shape-specific expected residual/norm rules. |
| COLAMD/reordered minimum-norm | `tests/test_colamd.c` | `test_minnorm_with_colamd` exercises `SPARSE_REORDER_COLAMD`. | Day 11 core owner. |
| Overdetermined and square fallback | `tests/test_colamd.c` | `test_minnorm_fallback_overdetermined` and `test_minnorm_square`. | Day 11 core owner. |
| Rank-deficient minimum-norm | `tests/test_colamd.c` | `test_minnorm_rank_deficient`. | Day 11 core owner, dependent on rank/nullspace boundaries from Days 2-7. |
| Minimum-norm refinement | `tests/test_colamd.c` | `test_refine_minnorm` and `test_refine_minnorm_null`. | Day 11 core owner. |
| Zero-row minimum-norm | `tests/test_colamd.c` | `test_minnorm_zero_row`. | Day 11 supporting owner; do not hide under generic rank-deficient label. |
| QR-vs-SVD pseudoinverse | `tests/test_colamd.c`, `tests/test_svd.c` | `test_minnorm_vs_pinv` compares QR minimum-norm with `sparse_pinv`; SVD owns Moore-Penrose tests. | Day 12 oracle/cross-check owner. |
| SuiteSparse submatrix minimum-norm | `tests/test_colamd.c` | `test_minnorm_ss_submatrix` uses a `west0067` 30x67 submatrix and skips on unavailable data or solve failure. | Day 12 corpus owner under Days 8-9 support-tier rules. |

## Behavior Owners For Days 11-12

| Owner key | Primary file | Behavior | Required proof if accepted |
| --- | --- | --- | --- |
| `qr_minnorm_colamd_behavior` | `tests/test_colamd.c` | Minimum-norm solve with `SPARSE_REORDER_COLAMD`. | Residual and norm diagnostics with explicit ordering options; no COLAMD superiority claim. |
| `qr_minnorm_fallback_behavior` | `tests/test_colamd.c` | `sparse_qr_solve_minnorm` fallback for square or overdetermined systems. | Compare against ordinary QR solve semantics or exact residual/solution expectations; no underdetermined optimality claim. |
| `qr_minnorm_rankdef_behavior` | `tests/test_colamd.c` | Rank-deficient minimum-norm solve. | Expected rank model, residual, solution norm, and nullspace non-claim; fixture-local rank threshold if rank is asserted. |
| `qr_minnorm_refinement_behavior` | `tests/test_colamd.c` | `sparse_qr_refine_minnorm` residual behavior. | Initial solution source, iteration budget, before/after residual, and non-increase or bounded-improvement rule. |
| `qr_minnorm_zero_row_behavior` | `tests/test_colamd.c` | Zero-row or structurally deficient constraints. | Consistency condition, residual, expected norm or named alternate, and no broad inconsistent-system claim. |
| `qr_minnorm_vs_svd_pinv_crosscheck` | `tests/test_colamd.c`, `tests/test_svd.c` | Bounded QR-vs-SVD pseudoinverse comparison. | Same fixture, same RHS, QR solution, SVD-pinv solution, tolerance, and explicit cross-check wording. |
| `qr_minnorm_suitesparse_submatrix` | `tests/test_colamd.c` | Optional corpus underdetermined submatrix smoke. | Matrix path, extraction rule, support tier, skip behavior, residual, norm bound, and no broad SuiteSparse claim. |

## Comparison Policy

| Quantity | Policy |
| --- | --- |
| Residual | Required for every accepted minimum-norm evidence lane. Report absolute residual or fixture-local relative residual with the scale rule in the artifact. |
| Solution norm | Required for every lane that claims minimum-norm behavior. Compare against an expected norm, SVD-pinv solution norm, or a named alternate feasible solution. |
| Solution values | Use exact solution-value comparisons only for tiny fixtures with derived values. Prefer norm and residual comparisons for larger or reordered paths. |
| Rank | If rank is part of the claim, pin fixture-local threshold semantics and expected rank before asserting minimum-norm behavior. |
| Pseudoinverse | May act only as a bounded cross-check when named. It is not a global oracle for QR minimum-norm behavior. |
| Fallback | Must be labeled as ordinary QR solve fallback, not underdetermined minimum-norm optimality. |
| SuiteSparse | Must follow Day 8-9 support-tier rules. Missing optional data may skip; accepted numerical disagreement must fail. |

## Helper Boundary

Future helper names must encode behavior ownership. Acceptable future names
include:

- `tf_qr_minnorm_make_colamd_2x5`
- `tf_qr_minnorm_measure_residual_norm2`
- `tf_qr_minnorm_measure_solution_norm2`
- `tf_qr_minnorm_make_rankdef_2x4`
- `tf_qr_minnorm_make_zero_row_2x4`
- `tf_qr_minnorm_make_west0067_submatrix`

Rejected names and patterns:

- `assert_minnorm`
- `check_minnorm`
- `minnorm_oracle`
- helpers that compare residual, norm, rank, and pseudoinverse behavior without
  naming the behavior owner
- helpers that hide optional-corpus skip behavior or SVD-pseudoinverse
  cross-check semantics

Measurement-only helpers may be acceptable in a future sprint, but assertion
tolerances and behavior claims must stay at the call site or in a
behavior-specific owner.

## Day 11 Implementation Gate

Day 11 may accept a core evidence batch only if each accepted lane names:

- behavior owner key
- matrix/RHS fixture
- expected residual and norm rule
- expected rank/threshold when rank-deficient behavior is asserted
- fallback, COLAMD, refinement, zero-row, or rank-deficient interpretation
- diagnostics printed on failure
- focused validation executable

Likely Day 11 focused validation, if `.c` or `.h` files change:

```text
make build/test_colamd && ./build/test_colamd
make build/test_qr_solve && ./build/test_qr_solve
make format && make lint && make test
```

## Day 12 Implementation Gate

Day 12 owns two separate decisions:

| Lane | Gate |
| --- | --- |
| QR-vs-SVD-pseudoinverse | Define whether the comparison is an oracle, bounded cross-check, or deferral; run affected QR/COLAMD and SVD focused tests if code changes. |
| SuiteSparse minimum-norm | Apply Day 8-9 corpus support tiers, extraction rule, skip behavior, residual/norm diagnostics, and no-broad-corpus wording. |

Likely Day 12 focused validation, if `.c` or `.h` files change:

```text
make build/test_colamd && ./build/test_colamd
make build/test_svd && ./build/test_svd
make format && make lint && make test
```

## Non-Claims Preserved

- No broad QR minimum-norm parity.
- No global minimum-norm optimality beyond named fixtures.
- No SVD pseudoinverse as a global QR oracle.
- No COLAMD, reorder, fallback, refinement, rank-deficient, zero-row, or
  SuiteSparse superiority claim.
- No broad SuiteSparse corpus support or platform parity.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK, dense-library,
  backend, package, ABI, public API, CMake, CTest, performance, scalability,
  memory, or state-of-the-art claim.

## Validation

Day 10 changed documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_125 docs/maintainer_guide.md tests/qr_external_dense_reference.py tests/test_qr.c tests/test_qr_solve.c tests/test_colamd.c tests/test_svd.c
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 6 is decomposed into behavior-specific evidence lanes. | Complete | See behavior owners for Days 11-12. |
| Helper names do not hide QR, COLAMD, SVD, fallback, or SuiteSparse semantics. | Complete | See helper boundary. |
| Validation expectations are known before implementation decisions. | Complete | See Day 11 and Day 12 implementation gates. |
