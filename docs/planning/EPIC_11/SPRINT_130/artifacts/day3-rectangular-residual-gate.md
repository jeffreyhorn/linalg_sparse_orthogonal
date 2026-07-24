# Sprint 130 Day 3 - Rectangular Residual Gate

## Purpose

Day 3 decides how rectangular partial-SVD residual evidence should be measured
and bounded before any Day 4 implementation.

The gate applies the Day 2 metric map to existing tall and wide rectangular
partial-SVD surfaces. It accepts one bounded Day 4 candidate and defers the
rest with explicit duplicate and claim boundaries.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 130 Day 1 baseline | Completed evidence fence and deferred-lane owner map. |
| Sprint 130 Day 2 metric map | Metric, tolerance, oracle, diagnostics, and failure policy for rectangular vector residuals. |
| Sprint 124 partial-SVD residual artifacts | Source of the original rectangular residual deferral and promotion gate. |
| `tests/svd_external_dense_reference.py` | Already owns `partial_svd_tall_diag_8x5_k3` singular-value references. |
| `tests/test_svd_partial_helpers.h` | Owns existing value-only tall external fixture, internal tall/wide value tests, square triplet residual helper, and wide `A v` smoke. |
| `docs/maintainer_guide.md` | Current maintainer evidence wording; Day 3 does not change it. |

## Current Rectangular Evidence Inventory

| Evidence | Shape | Current metric | Boundary |
| --- | --- | --- | --- |
| `partial_svd_tall_diag_8x5_k3` | Tall 8x5 exact diagonal | External singular-value agreement only | No vector residual, `A^T u`, subspace, or solver-selection claim. |
| `test_partial_svd_tall` | Tall 10x5 exact diagonal | Product partial values compared with product full-SVD values | Internal consistency only; no external residual claim. |
| `test_partial_svd_wide` | Wide 5x10 exact diagonal | Product partial values compared with product full-SVD values, loose wide tolerance | Internal consistency only; no vector residual or external claim. |
| `test_partial_svd_vectors_rectangular_lowrank_recon` | Rectangular low-rank fixture | Reconstruction and `A v` residual smoke | Low-rank/reconstruction owner; not rectangular triplet residual evidence. |
| `test_partial_svd_vectors_wide` | Wide 4x8 exact diagonal | Top values and `A v` residual smoke | Checks only one triplet side; no `A^T u` residual and no external value oracle. |
| `partial_svd_vector_residual_diag6_k2` | Square 6x6 exact diagonal | External values plus `A v`, `A^T u`, U/V orthogonality | Square-only baseline; not rectangular shape evidence. |

## Rectangular Candidate Table

| Candidate | Shape | Distinct trust value | Required metrics | Oracle | Day 3 decision |
| --- | --- | --- | --- | --- | --- |
| `partial_svd_vector_residual_tall8x5_k3` | Tall exact diagonal, `k=3` | Extends the accepted square triplet-residual protocol across thin-left/tall output shape using an existing external value fixture. | `sigma` agreement, `A v_i - sigma_i u_i`, `A^T u_i - sigma_i v_i`, U/V orthogonality, `m/n/k` shape checks. | Existing external singular-value helper plus product-owned residuals. | Accept for Day 4 implementation. |
| `partial_svd_vector_residual_wide5x8_k3` or similar | Wide exact diagonal | Would cover thin-right/wide output shape, but overlaps with existing wide vector smoke and needs separate shape policy. | Same triplet residuals plus shape checks. | Would need new analytic or external value fixture. | Defer until the tall lane lands and wide-specific output semantics are pinned. |
| Upgrade `test_partial_svd_vectors_wide` | Wide exact diagonal, existing smoke | Improves existing smoke, but it is not external and currently only checks `A v`. | Add `A^T u`, external values, and stricter diagnostics. | New external fixture or analytic values. | Defer to avoid silently converting an internal smoke into external evidence. |
| Rectangular low-rank residual lane | Rectangular low-rank fixture | Could cover reconstruction and residual behavior. | Reconstruction norm, `A v`, optional `A^T u`, retained values. | Analytic or full dense reference. | Defer to Day 12 low-rank owner. |
| Rectangular nonsymmetric residual lane | Non-diagonal rectangular | Covers shape plus nonsymmetric behavior. | Value and triplet residual metrics with conditioning note. | New dense-reference fixture. | Defer to Days 5-6 nonsymmetric owner. |

## Accepted Day 4 Candidate

Day 4 may implement `partial_svd_vector_residual_tall8x5_k3` if it preserves
the following contract:

| Field | Required value |
| --- | --- |
| Matrix | 8x5 exact diagonal with values `8.0`, `5.0`, `3.0`, `1.0`, `0.25` and three structural zero rows. |
| `k` | `3` |
| Options | `compute_uv = 1`, `economy = 1`, default `max_iter` and `tol` unless Day 4 records a reason to pin them. |
| Value oracle | Existing `partial_svd_tall_diag_8x5_k3` external singular-value helper. |
| Product-owned metrics | `||A v_i - sigma_i u_i||_2`, `||A^T u_i - sigma_i v_i||_2`, `U^T U - I`, `V^T V - I`. |
| Shape checks | `partial.k == 3`, `partial.m == 8`, `partial.n == 5`, `U` is `k x m` under current storage convention, and `Vt` stores `k` rows over `n` columns. |
| Tolerance | `1e-8` for singular values, triplet residuals, and orthogonality because this is an exact diagonal, well-separated fixture. |
| Failure classes | Helper skip, helper protocol error, shape/API regression, singular-value regression, vector residual regression, or orthogonality regression. |
| Public wording | No public solver-selection update on Day 4; maintainer evidence may update only as a bounded tall rectangular vector-residual fixture. |

## Shape-Specific Metric Policy

- Tall and wide rectangular evidence are separate. Passing a tall lane does not
  prove wide behavior, and passing a wide lane does not prove tall behavior.
- The first accepted rectangular lane should be tall because the existing
  external helper already provides `partial_svd_tall_diag_8x5_k3` values.
- A wide lane needs its own fixture key, external or analytic singular-value
  oracle, `A^T u` residual coverage, and shape diagnostics before promotion.
- Rectangular vector residual evidence should not imply nonsymmetric behavior.
  Exact diagonal rectangular fixtures preserve shape risk while avoiding
  non-diagonal conditioning and vector orientation ambiguity.
- Rectangular reconstruction and low-rank optimality are separate evidence
  classes and must stay with the Day 12 low-rank owner unless Day 4 records a
  narrow residual-only reason to touch them.

## Day 4 Implementation Checklist

Before editing code on Day 4, confirm:

1. The fixture key is `partial_svd_vector_residual_tall8x5_k3`.
2. The existing external helper returns three singular values for
   `partial_svd_tall_diag_8x5_k3`.
3. The test requests vectors with `compute_uv = 1` and `economy = 1`.
4. Both triplet residual sides and both orthogonality metrics are checked.
5. The implementation reuses existing residual helpers where possible instead
   of adding a generic vector equality helper.
6. The assertion messages or diagnostics identify value, residual,
   orthogonality, and shape failure classes.
7. The maintainer guide is updated only if the new evidence passes and the
   wording remains bounded to this fixture.
8. Focused and full validation are planned before closeout.

## Validation Plan

If Day 4 changes only `tests/test_svd_partial_helpers.h` and optionally
`docs/maintainer_guide.md`, run:

1. `python3 tests/svd_external_dense_reference.py partial_svd_tall_diag_8x5_k3`
2. `make build/test_svd && ./build/test_svd`
3. `make format && make lint && make test`
4. `git diff --check`

If Day 4 changes the Python helper protocol, also run
`python3 -m py_compile tests/svd_external_dense_reference.py` and a focused
parser/protocol check.

## Deferrals

| Deferred lane | Reason | Future owner and promotion gate |
| --- | --- | --- |
| Wide rectangular vector residual | Wide output shape and right-space semantics should not be conflated with the accepted tall candidate. | Future rectangular owner must add a wide fixture key, value oracle, both triplet residuals, orthogonality, shape checks, and bounded wording. |
| Existing wide vector smoke upgrade | Current smoke checks only `A v` and product-owned values; converting it to external evidence would blur evidence class. | Future owner must decide whether to replace, supplement, or leave the smoke before adding external values and `A^T u`. |
| Rectangular low-rank reconstruction | Belongs to low-rank optimality, not rectangular triplet residual. | Day 12 owner must define Frobenius/spectral/reconstruction metric and sparse-output policy. |
| Nonsymmetric rectangular residual | Needs non-diagonal fixture, conditioning notes, and dense-reference value boundary. | Days 5-6 owner must define fixture, oracle, residual tolerance, and non-subspace claim boundary. |

## Non-Claim Register

Day 3 preserves the following non-claims:

- no broad rectangular partial-SVD parity;
- no wide rectangular vector-residual claim;
- no nonsymmetric rectangular residual claim;
- no repeated-spectrum, clustered-spectrum, rank-deficient subspace, or
  null-space claim;
- no SuiteSparse corpus residual, low-rank optimality, or convergence-budget
  claim;
- no raw vector equality, sign, orientation, ordering, or unique-basis claim;
- no public solver-selection wording readiness;
- no LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Accepted rectangular evidence has distinct proof value beyond Sprint 124. | Complete | `partial_svd_vector_residual_tall8x5_k3` adds tall shape triplet-residual evidence beyond the square Day 9 Sprint 124 baseline. |
| Tall and wide interpretations are not conflated. | Complete | Wide lanes are explicitly deferred with separate promotion gates. |
| No rectangular result implies broad nonsymmetric or solver-selection parity. | Complete | Nonsymmetric lanes stay with Days 5-6 and solver-selection wording stays deferred to Day 14. |

## Day 4 Handoff

Day 4 should implement or explicitly defer
`partial_svd_vector_residual_tall8x5_k3`. If implementation proceeds, keep the
helper protocol singular-value only, reuse product-owned residual helpers, add
bounded maintainer wording only after validation, and avoid adding a wide,
nonsymmetric, subspace, corpus, low-rank, convergence, or solver-selection
claim.
