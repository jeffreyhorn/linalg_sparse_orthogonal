# Sprint 130 Day 5 - Nonsymmetric Rectangular Gate

## Purpose

Day 5 decides whether nonsymmetric rectangular partial-SVD evidence can add
trust beyond the Day 4 tall exact diagonal residual lane, and defines the
policy required before Day 6 implementation.

The key distinction is deliberate: Day 4 covered a tall rectangular shape, but
the matrix was diagonal. Day 5 covers non-diagonal, nonsymmetric rectangular
behavior only if the evidence has its own fixture, oracle, residual metrics,
orientation rules, tolerance, and failure interpretation.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 130 Day 2 metric map | Defines nonsymmetric rectangular residual metric and oracle policy. |
| Sprint 130 Day 3 rectangular gate | Keeps tall and wide shape coverage separate from nonsymmetric behavior. |
| Sprint 130 Day 4 rectangular evidence | Adds `partial_svd_vector_residual_tall8x5_k3` and explicitly defers nonsymmetric rectangular evidence. |
| Sprint 124 residual scenario matrix | Names nonsymmetric rectangular value residual as deferred external-value work. |
| `tests/test_svd_partial_helpers.h` | Owns current `test_partial_svd_nonsymmetric` internal full-SVD comparison and reusable triplet residual helpers. |
| `tests/svd_external_dense_reference.py` | Current singular-value helper; it already supports full-SVD nonsymmetric rectangular fixture `svd_rect_fullrank_6x4`, but not a partial-SVD nonsymmetric fixture. |
| `docs/maintainer_guide.md` | Evidence wording owner; Day 5 does not change public or maintainer wording. |

## Current Nonsymmetric Rectangular Coverage

| Evidence | Shape | Current metric | Boundary |
| --- | --- | --- | --- |
| `test_partial_svd_nonsymmetric` | 10x8 deterministic non-diagonal matrix, `k=4` | Product partial-SVD singular values compared to product full-SVD singular values with `0.05 * sigma + 1e-10` tolerance | Internal consistency only; no external oracle, vector residual, or subspace claim. |
| `svd_rect_fullrank_6x4` | 6x4 deterministic mixed-sign non-diagonal matrix | External full-SVD singular-value fixture | Full-SVD value evidence only; not partial-SVD residual evidence. |
| `partial_svd_vector_residual_tall8x5_k3` | 8x5 exact diagonal, `k=3` | External values plus product-owned triplet residuals and orthogonality | Tall rectangular exact diagonal only; no nonsymmetric or non-diagonal claim. |
| `test_partial_svd_vectors_wide` | 4x8 exact diagonal, `k=2` | Product-owned values plus `A v` residual smoke | Wide vector smoke only; no `A^T u`, external value, or nonsymmetric claim. |

## Candidate Table

| Candidate | Fixture | Distinct trust value | Required metrics | Oracle | Day 5 decision |
| --- | --- | --- | --- | --- | --- |
| `partial_svd_nonsym_rect10x8_k4_values` | Existing 10x8 deterministic non-diagonal matrix from `test_partial_svd_nonsymmetric` | Adds independent singular-value reference for current nonsymmetric partial-SVD value behavior. | Ordered top-4 singular values, dimensions, conditioning note. | New external helper fixture that emits singular values only. | Accept as the minimum Day 6 lane if vector residuals are not stable under `compute_uv`. |
| `partial_svd_nonsym_rect10x8_k4_residual` | Same 10x8 deterministic non-diagonal matrix with vectors requested | Adds nonsymmetric left/right triplet residual coverage beyond value comparison. | External top-4 singular values, `A v_i - sigma_i u_i`, `A^T u_i - sigma_i v_i`, U/V orthogonality, dimensions. | New external singular-value fixture plus product-owned residuals. | Preferred Day 6 lane if a focused preflight proves residual tolerances are stable. |
| Reuse `svd_rect_fullrank_6x4` for partial SVD | Existing 6x4 full-SVD external fixture | Avoids adding a new matrix to the helper. | Top-k singular values and optional triplet residuals. | Existing external helper key, but full-SVD fixture semantics are already owned. | Defer; would blur full-SVD fixture ownership unless a new partial-SVD key and boundary are added. |
| Nonsymmetric wide residual | New wide non-diagonal fixture | Would cover right-space/wide behavior. | Same as residual lane plus wide shape policy. | New external helper fixture. | Defer until nonsymmetric tall/rectangular value-residual policy lands. |
| Nonsymmetric subspace fixture | Non-diagonal with repeated or clustered values | Would cover subspace behavior. | Projector/principal-angle metrics. | Future helper protocol expansion. | Defer to Days 7-8 subspace owner. |

## Accepted Day 6 Path

Day 6 may implement a bounded nonsymmetric rectangular lane using the existing
10x8 matrix from `test_partial_svd_nonsymmetric`, but only under this staged
contract:

| Field | Required value |
| --- | --- |
| Fixture key | `partial_svd_nonsym_rect10x8_k4` or a more explicit suffix such as `_values` or `_residual`. |
| Matrix | 10x8 deterministic non-diagonal matrix with entries `(i + 1) / (j + 1)` when `(i + j) % 3 != 0`. |
| `k` | `4` |
| Spectrum | Must be treated as ordered only after the external helper confirms a useful gap between the fourth and fifth singular values. |
| Value oracle | New singular-value output from `tests/svd_external_dense_reference.py`; no vector or projector oracle. |
| Preferred residual metrics | `||A v_i - sigma_i u_i||_2`, `||A^T u_i - sigma_i v_i||_2`, U/V orthogonality, and shape checks. |
| Fallback value-only metrics | External top-4 singular-value agreement plus dimensions and diagnostics if vector residuals are not stable under current partial-SVD vector recovery. |
| Tolerance | Fixture-specific. Day 6 must measure preflight diagnostics before choosing a bound; copying exact-diagonal `1e-8` is not allowed. |
| Public wording | No public solver-selection update. Maintainer evidence may update only as bounded nonsymmetric rectangular partial-SVD evidence after validation. |

## Left/Right Vector And Residual Policy

- Nonsymmetric rectangular residual evidence must check both singular-triplet
  equations when vectors are requested.
- `A v_i - sigma_i u_i` alone is insufficient because it misses right-to-left
  consistency through `A^T`.
- U and V orthogonality diagnostics are required because the matrix is
  non-diagonal and the basis is not visually inspectable.
- Left and right singular vectors must not be compared componentwise with any
  dense helper output unless Day 6 explicitly adds sign/orientation handling.
- If values are close or repeated, the lane must fall back to subspace policy
  and defer to Days 7-8 rather than using vector equality.

## Orientation, Sign, And Multiplicity Rules

- Raw vector equality is forbidden for Day 6.
- Sign flips are not failures.
- Singular values may be compared in descending order only if the helper
  diagnostics show the top-4 values are separated enough for deterministic
  ordering.
- If the fourth and fifth singular values are clustered under the selected
  tolerance, Day 6 must not publish vector-residual evidence for individual
  triplets; it should either narrow to fewer values or defer to the clustered
  subspace owner.
- If `sparse_svd_partial` returns a valid rotated basis inside an ambiguous
  subspace, Day 6 must treat that as subspace work, not vector failure.

## Skip And Failure Interpretation

| Failure class | Meaning |
| --- | --- |
| Helper skip | Missing `python3` or existing platform skip under the external-helper policy. |
| Helper protocol error | New helper fixture failed to generate or parse singular values; infrastructure failure. |
| Shape/API regression | Returned `m`, `n`, `k`, `U`, `Vt`, or vector availability does not match the fixture contract. |
| Singular-value mismatch | Bounded nonsymmetric rectangular value regression for the named fixture only. |
| Vector residual mismatch | Bounded nonsymmetric triplet-quality regression; not a sign or orientation failure. |
| Orthogonality mismatch | Vector publication quality regression for the named fixture. |
| Clustered/tie ambiguity | Deferral trigger unless Day 6 has a subspace metric. |
| Non-convergence | Day 6 must classify whether this is fixture failure, unsupported budget behavior, or a deferral to Day 13. |

## Day 6 Acceptance Checklist

Before Day 6 implementation proceeds, confirm:

1. The fixture key and matrix builder are named and non-duplicative.
2. The external helper emits top-4 singular values for the same matrix.
3. The top-4 ordering and fourth/fifth gap are diagnostically acceptable.
4. The chosen tolerance is fixture-specific and justified by preflight output.
5. If vectors are requested, both triplet residual equations and U/V
   orthogonality are checked.
6. If vector residuals are unstable, the fallback value-only lane is explicitly
   documented and does not claim vector or subspace behavior.
7. Maintainer wording is updated only after validation and remains bounded.
8. Focused helper, focused SVD, full quality, and diff hygiene validation are
   run for any `.c`, `.h`, or Python helper changes.

## Validation Plan For Day 6

If Day 6 changes the Python helper and SVD tests, run:

1. `python3 -m py_compile tests/svd_external_dense_reference.py`
2. `python3 tests/svd_external_dense_reference.py partial_svd_nonsym_rect10x8_k4`
3. `make build/test_svd && ./build/test_svd`
4. `make format && make lint && make test`
5. `git diff --check`

If Day 6 changes documentation only because the lane defers, run
`git diff --check` and the focused Sprint 130 markdown whitespace scan.

## Deferrals

| Deferred lane | Reason | Future owner and promotion gate |
| --- | --- | --- |
| Wide nonsymmetric rectangular residual | Wide right-space behavior should not be folded into the first nonsymmetric rectangular lane. | Future rectangular owner must define a wide non-diagonal fixture, value oracle, triplet residuals, shape policy, and bounded wording. |
| Nonsymmetric subspace evidence | Needs projector or principal-angle metrics and cannot rely on raw vector equality. | Days 7-8 subspace owner. |
| Nonsymmetric low-rank optimality | Reconstruction/optimality is a different evidence class. | Day 12 low-rank owner. |
| Nonsymmetric convergence-budget behavior | Needs options, iteration cap, tolerance, status, and partial-result policy. | Day 13 convergence owner. |
| Public solver-selection wording | One nonsymmetric rectangular fixture is insufficient for public solver-selection guidance. | Day 14 claim gate after all Sprint 130 evidence is reconciled. |

## Non-Claim Register

Day 5 does not claim:

- nonsymmetric rectangular partial-SVD behavior is already covered by Day 4;
- wide nonsymmetric vector-residual behavior;
- raw vector equality, sign, orientation, ordering, or unique-basis stability;
- repeated-spectrum, clustered-spectrum, rank-deficient subspace, or
  null-space behavior;
- SuiteSparse corpus residual parity;
- low-rank global optimality;
- convergence-budget guarantees;
- public solver-selection wording readiness;
- LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Nonsymmetric behavior is not treated as already covered by rectangular shape evidence. | Complete | Day 4 exact diagonal tall evidence is fenced from Day 5 non-diagonal nonsymmetric evidence. |
| Vector comparisons are allowed only under stable orientation rules. | Complete | Raw vector equality is forbidden; vector evidence requires residuals and orthogonality. |
| Unsupported candidates have explicit deferral blockers. | Complete | Wide, subspace, low-rank, convergence, and solver-selection lanes are deferred with owner gates. |

## Day 6 Handoff

Day 6 should either implement the `partial_svd_nonsym_rect10x8_k4` lane or
explicitly defer it after preflight diagnostics. The preferred implementation
adds an external singular-value fixture for the existing 10x8 deterministic
matrix and checks product-owned triplet residuals. If vector residuals are not
stable under a defensible fixture-specific tolerance, Day 6 may narrow to a
value-only external partial-SVD lane or defer with blocker and future owner.
