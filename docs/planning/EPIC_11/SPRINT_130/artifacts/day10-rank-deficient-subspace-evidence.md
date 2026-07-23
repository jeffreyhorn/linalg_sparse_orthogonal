# Sprint 130 Day 10 - Rank-Deficient Subspace Evidence

## Purpose

Day 10 applies the Day 9 rank-deficient subspace gate. It implements one
bounded rank-deficient range-projector lane with `k == rank`, then keeps
zero-crossing, null-space, duplicate-column, near-zero tail,
minimum-norm/pseudoinverse, and solver-selection claims deferred.

The accepted lane is `partial_svd_rankdef_diag6x4_k2_range_projector`.

## Accepted Evidence Lane

| Field | Value |
| --- | --- |
| Fixture key | `partial_svd_rankdef_diag6x4_k2_range_projector` |
| Matrix | 6x4 diagonal with nonzero entries `9` and `6`, remaining diagonal entries zero, and two extra zero rows. |
| Expected rank | `2`, checked with `sparse_svd_rank(A, 1e-8)`. |
| Requested `k` | `2`, equal to rank so the lane does not cross into zero singular slots. |
| Options | `compute_uv = 1`, `economy = 1`, default iteration and tolerance settings. |
| Oracle | Analytic positive singular values `[9, 6]` and analytic left/right coordinate range projectors onto coordinates `0..1`. |
| Primary metric | Left and right range-projector Frobenius errors. |
| Secondary metrics | Top-2 singular values, both singular-triplet residual equations, U/V orthogonality, shape and vector-availability checks. |
| Tolerance | `1e-8` for this exact diagonal rank-deficient range fixture only. |

## Why Range Only

The fixture has right nullity `2` and left nullity `4`, but Day 10 does not
assert either null space. It requests only `k=2`, matching the positive rank,
so the evidence checks the positive left and right range subspaces without
depending on zero singular-vector publication, null-space basis orientation,
or any minimum-norm/pseudoinverse behavior.

## Implementation Summary

| File | Change |
| --- | --- |
| `tests/test_svd_partial_helpers.h` | Added range-projector error helpers for the current partial-SVD U/Vt layouts and `test_partial_svd_rankdef_diag6x4_k2_range_projector`. |
| `tests/test_svd.c` | Registered the new rank-deficient range-projector test next to the bounded partial-SVD evidence lanes. |
| `docs/maintainer_guide.md` | Added the bounded rank-deficient range-projector fixture while preserving null-space, minimum-norm, and broad rank-deficient non-claims. |

## Evidence Diagnostics

The focused SVD test reports:

1. expected numerical rank;
2. max top-2 singular-value difference;
3. left range-projector Frobenius error;
4. right range-projector Frobenius error;
5. max `A v_i - sigma_i u_i` residual;
6. max `A^T u_i - sigma_i v_i` residual;
7. U and V orthogonality errors.

Focused validation produced:

| Metric | Observed |
| --- | --- |
| Reported rank | `2` |
| Max top-2 singular-value difference | `1.776e-15` |
| Left range-projector Frobenius error | `0.000e+00` |
| Right range-projector Frobenius error | `2.371e-16` |
| Max `A v - sigma u` residual | `1.776e-15` |
| Max `A^T u - sigma v` residual | `2.480e-15` |
| U orthogonality error | `0.000e+00` |
| V orthogonality error | `1.570e-16` |

## Deferrals

| Deferred lane | Reason | Future owner and promotion gate |
| --- | --- | --- |
| `k > rank` zero-crossing evidence | Needs zero singular-value tolerance, zero-vector/basis semantics, and null-space projector policy. | Future rank/null-space owner must define zero-space publication and failure classes. |
| Null-space projector evidence | Day 10 checks only positive range projectors. | Future null-space owner must define left/right nullity, analytic/projector oracle, and basis ambiguity policy. |
| Duplicate-column 5x4 range projector | Requires a clear left projector oracle or external projector protocol; product full-SVD alone would be internal consistency. | Future external/projector owner may add helper projector output or analytic derivation. |
| Existing `test_partial_svd_rank_deficient` upgrade | Current test requests `k=4` and crosses into zero slots; upgrading it would mix value, rank, range, and null-space evidence. | Future owner should split range-only and zero-crossing tests before changing it. |
| Day 6 near-zero nonsymmetric tail | Near-zero values are clustered and require rank threshold plus convergence interpretation. | Future rank/convergence owner. |
| Minimum-norm and pseudoinverse behavior | Separate solver behavior, not partial-SVD range subspace evidence. | Minimum-norm/pseudoinverse owner. |
| Public solver-selection wording | One exact range-projector fixture does not justify broader public guidance. | Day 14 claim gate. |

## Non-Claim Register

Day 10 does not claim:

- rank-deficient null-space projector correctness;
- zero singular-vector or null-space basis stability;
- partial-SVD behavior when `k > rank`;
- near-zero clustered-tail behavior;
- duplicate-column partial-SVD projector behavior;
- minimum-norm or pseudoinverse correctness;
- broad rank-deficient solver robustness;
- public solver-selection wording readiness;
- LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity.

## Validation Plan

Because Day 10 touches C/header tests and maintainer evidence, run:

1. `make format && make build/test_svd && ./build/test_svd`
2. `make format && make lint && make test`
3. `git diff --check`
4. focused Sprint 130 markdown trailing-whitespace scan

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Accepted evidence validates bounded rank-deficient subspace behavior. | Complete | Focused SVD and full quality validation passed with projector, residual, value, rank, and orthogonality diagnostics below `1e-8`. |
| No raw basis uniqueness or broad optimality claim is introduced. | Complete | The lane uses range projectors and explicitly excludes null-space, minimum-norm, pseudoinverse, and broad rank-deficient claims. |
| Every deferral has blocker, dependency, and future owner. | Complete | Deferral table records zero-crossing, null-space, duplicate-column, existing-test upgrade, near-zero tail, minimum-norm/pseudoinverse, and solver-selection gates. |
