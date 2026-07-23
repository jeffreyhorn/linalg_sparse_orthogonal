# Sprint 130 Day 4 - Rectangular Residual Evidence

## Decision

Implement one bounded tall rectangular partial-SVD vector-residual lane:
`partial_svd_vector_residual_tall8x5_k3`.

This lane extends Sprint 124's square
`partial_svd_vector_residual_diag6_k2` evidence across a tall exact diagonal
shape. It keeps the external helper singular-value only and keeps vector
quality product-owned through triplet residual and orthogonality checks.

## Accepted Evidence

| Field | Decision |
| --- | --- |
| Fixture key | `partial_svd_vector_residual_tall8x5_k3` |
| Dense-reference key | `partial_svd_tall_diag_8x5_k3` |
| Matrix | 8x5 exact diagonal with values `8.0`, `5.0`, `3.0`, `1.0`, `0.25` and three structural zero rows |
| `k` | `3` |
| Options | `compute_uv = 1`, `economy = 1`, default `max_iter` and `tol` |
| Product owner | `tests/test_svd_partial_helpers.h` |
| Test registration owner | `tests/test_svd.c` |
| External helper owner | `tests/svd_external_dense_reference.py`, singular values only |
| Metrics | external singular-value agreement, `A v_i - sigma_i u_i`, `A^T u_i - sigma_i v_i`, `U^T U - I`, `V^T V - I`, and `m/n/k` shape checks |
| Tolerances | `1e-8` for singular values, residuals, and orthogonality on this exact diagonal fixture |
| Failure interpretation | helper skip, helper protocol error, shape/API regression, singular-value regression, vector residual regression, or orthogonality regression |

## Implementation Summary

| File | Change |
| --- | --- |
| `tests/test_svd_partial_helpers.h` | Added `test_partial_svd_external_dense_reference_vector_residual_tall8x5_k3`, reusing the existing external singular-value helper and `partial_svd_max_triplet_residuals`. |
| `tests/test_svd.c` | Registered the new test after the square vector-residual fixture. |
| `docs/maintainer_guide.md` | Added the bounded tall rectangular vector-residual fixture to the SVD evidence table and preserved rectangular/nonsymmetric non-claims. |

No Python helper protocol changed. The existing
`partial_svd_tall_diag_8x5_k3` singular-value fixture remains the only
external oracle for this lane.

## Diagnostics

The test prints:

- maximum singular-value difference against the external helper;
- maximum `A v_i - sigma_i u_i` residual;
- maximum `A^T u_i - sigma_i v_i` residual;
- `U` orthogonality error;
- `V` orthogonality error.

These diagnostics classify failures without comparing raw singular-vector
components.

## Deferrals

| Deferred lane | Reason | Future owner and promotion gate |
| --- | --- | --- |
| Wide rectangular vector residual | Tall shape evidence does not prove wide right-space behavior. | Future rectangular owner must add a wide fixture key, independent or analytic value oracle, both triplet residuals, orthogonality, shape checks, and bounded wording. |
| Existing wide vector smoke upgrade | Current smoke remains product-owned and checks only `A v`; converting it now would blur smoke coverage with external evidence. | Future owner must decide whether to replace or supplement it before adding `A^T u` and external values. |
| Rectangular low-rank reconstruction | Reconstruction and optimality are separate from triplet residual evidence. | Day 12 low-rank owner must define Frobenius/spectral/reconstruction metrics and sparse-output policy. |
| Nonsymmetric rectangular residual | Exact diagonal tall evidence intentionally avoids non-diagonal conditioning and nonsymmetric claims. | Days 5-6 owner must define non-diagonal fixture, value oracle, residual tolerance, and no-subspace boundary. |

## Maintainer Evidence Boundary

The maintainer guide now names
`partial_svd_vector_residual_tall8x5_k3` as a bounded SVD evidence fixture.
The wording remains limited to the named fixture and preserves no-claims for:

- broad SVD or partial-SVD external parity;
- broad vector/subspace parity;
- broad rectangular or nonsymmetric behavior;
- low-rank optimality;
- convergence-budget, performance, or platform parity.

No public solver-selection wording changes are made on Day 4.

## Validation Plan

Because Day 4 changes `.c` and `.h` test files, validation requires:

1. `python3 tests/svd_external_dense_reference.py partial_svd_tall_diag_8x5_k3`
2. `make build/test_svd && ./build/test_svd`
3. `make format && make lint && make test`
4. `git diff --check`

## Non-Claim Register

Day 4 does not claim:

- broad rectangular partial-SVD parity;
- wide rectangular vector-residual behavior;
- nonsymmetric rectangular residual behavior;
- repeated-spectrum, clustered-spectrum, rank-deficient subspace, or
  null-space behavior;
- SuiteSparse corpus residual parity;
- low-rank global optimality;
- convergence-budget guarantees;
- raw vector equality, sign, orientation, ordering, or unique-basis stability;
- public solver-selection wording readiness;
- LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Rectangular evidence is bounded, validated, and non-duplicative, or explicitly deferred. | Complete | New tall rectangular lane is bounded, non-duplicative, and passed focused plus full validation. |
| Touched files have focused validation. | Complete | External helper, focused SVD, and full quality validation passed. |
| No broad rectangular or nonsymmetric partial-SVD claim is introduced. | Complete | Maintainer wording is fixture-bounded and nonsymmetric lanes remain deferred. |

## Day 5 Handoff

Day 5 should address nonsymmetric rectangular evidence separately. It should
not treat the new tall exact diagonal lane as nonsymmetric coverage. A
nonsymmetric lane needs a non-diagonal fixture, value oracle, residual
tolerance, conditioning note, left/right vector semantics, and explicit
non-subspace wording before implementation.
