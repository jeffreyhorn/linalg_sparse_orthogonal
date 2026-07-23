# Sprint 130 Day 7 - Repeated And Clustered Spectrum Policy

## Purpose

Day 7 defines the policy for repeated and clustered partial-SVD spectra before
any Day 8 implementation. The key rule is that repeated or clustered singular
values make individual singular vectors non-unique or numerically unstable, so
raw vector equality and per-index vector orientation are not valid evidence.

Day 7 accepts a narrow Day 8 repeated-spectrum projector lane if the
implementation adds explicit subspace metrics. Clustered-spectrum evidence
remains deferred unless Day 8 can also define spectral gap, ordering, tolerance,
and convergence-budget semantics without blurring it into the repeated lane.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 130 Day 2 metric map | Requires projector, principal-angle, or two-way projection metrics for repeated and clustered lanes. |
| Sprint 130 Day 6 nonsymmetric evidence | Shows why near-zero clustered tails must not be promoted into individual-vector evidence. |
| Sprint 124 Day 8 partial-SVD semantics | Establishes that basis-ambiguous fixtures compare subspaces, not raw vectors. |
| Sprint 124 Day 10 residual scenario matrix | Names repeated and clustered partial-SVD subspace lanes as deferred. |
| Sprint 124 Day 11 deferral package | Carries forward projector/principal-angle protocol and failure-policy requirements. |
| `tests/test_svd.c` | Contains full-SVD repeated singular-value coverage, but not partial-SVD subspace evidence. |
| `tests/test_svd_partial_helpers.h` | Owns partial-SVD residual helpers and is the likely Day 8 owner for local projector checks. |
| `tests/svd_external_dense_reference.py` | Emits singular values only today; projector output would require a helper protocol expansion. |
| `src/sparse_svd_partial.c` | Notes that clustered spectra need larger subspaces, but does not expose a budgeted convergence contract. |

## Current Coverage Inventory

| Evidence | Spectrum | Current metric | Boundary |
| --- | --- | --- | --- |
| `test_svd_repeated` | Full-SVD 3x3 diagonal with all singular values equal to `5` | Full-SVD singular values only | Not partial-SVD evidence and no vector/subspace metric. |
| `partial_svd_vector_residual_diag6_k2` | Square diagonal with separated leading values | External top-2 values plus triplet residuals and orthogonality | Vector-residual evidence only because the top values are separated. |
| `partial_svd_vector_residual_tall8x5_k3` | Tall diagonal with separated leading values | External top-3 values plus triplet residuals and orthogonality | Rectangular shape evidence only; no repeated or clustered claim. |
| `partial_svd_vector_residual_nonsym_rect10x8_k3` | Non-diagonal 10x8 with stable top-3 values | External top-3 values plus triplet residuals and orthogonality | Explicitly avoids the near-zero clustered tail starting at the fourth value. |
| `test_partial_svd_vectors_vs_full` | Square diagonal with separated values | Product partial vectors compared with product full-SVD vectors by dot product | Internal and only valid because the selected values are separated. |
| `test_partial_svd_nos4` and corpus vector checks | Matrix-specific corpus spectra | Product full-SVD or product residual diagnostics | Internal smoke; not independent subspace or clustered-spectrum evidence. |

## Candidate Table

| Candidate | Fixture | Distinct trust value | Required metrics | Oracle | Day 7 decision |
| --- | --- | --- | --- | --- | --- |
| `partial_svd_repeated_diag6_k3_projector` | 6x6 diagonal `diag(7, 7, 7, 3, 2, 1)` with `k=3` | Proves the product returns the correct leading invariant subspace when individual leading vectors may rotate or permute. | External or analytic top-3 value multiset, left projector distance, right projector distance, U/V orthogonality, triplet residuals, shape checks. | Analytic projector onto coordinates `0..2`; no raw vector oracle required. | Accept as the lowest-risk Day 8 implementation path. |
| `partial_svd_repeated_diag6_k2_partial_cluster` | 6x6 diagonal `diag(7, 7, 7, 3, 2, 1)` with `k=2` | Would test a partial slice through a repeated multiplicity-3 leading subspace. | Needs policy for under-selecting inside a repeated block and comparing any two-dimensional subspace of a three-dimensional invariant space. | Analytic containing-subspace projection, not exact projector equality. | Defer; this is a containment metric, not the first projector lane. |
| `partial_svd_clustered_diag6_k3_projector` | 6x6 diagonal with values such as `7, 7 - delta, 7 - 2 delta, 3, 2, 1` | Would test near-tie stability while preserving a unique ordered top-3 span. | Declared gap, projector or principal-angle metric, value set policy, residuals, orthogonality, iteration diagnostics. | Analytic values/projector for diagonal fixture, but tolerance must account for gap and budget. | Defer by default to avoid conflating spectral-gap and convergence-budget policy. |
| Clustered tail from `partial_svd_nonsym_rect10x8_k4` | Existing Day 6 10x8 non-diagonal matrix, `k=4` | Would revisit the near-zero cluster discovered on Day 6. | Rank threshold, near-zero tolerance, subspace/rank split, and convergence semantics. | Dense-reference values only today; no projector protocol. | Defer to rank-deficient and clustered owners. |
| Corpus clustered spectrum | `nos4`, `west0067`, or bounded SuiteSparse subset | Would exercise realistic clustered spectra. | Optional-data support tier, conditioning, residual windows, projector tolerance, runtime, skip behavior. | Product-owned unless external metadata is added. | Defer to corpus and convergence owners. |

## Subspace Metric Policy

Repeated and clustered spectrum evidence must compare subspaces rather than
individual basis vectors.

Accepted Day 8 projector metrics:

- Left projector: `P_U = U_k U_k^T`
- Right projector: `P_V = V_k V_k^T`
- Projector error: `||P_product - P_expected||_F`
- Optional principal-angle signal: `max sin(theta_i)`, derived from the
  singular values of `U_expected^T U_product` or `V_expected^T V_product`

For a diagonal repeated-leading fixture, the expected left and right projectors
are both the diagonal matrix with ones in the repeated leading coordinate block
and zeros elsewhere. This analytic oracle is sufficient for the first repeated
projector lane and avoids expanding the external helper protocol.

Day 8 must check dimensions before projector construction. It must also keep
left and right subspace failures distinct because rectangular and
nonsymmetric fixtures can fail on only one side.

## Value, Gap, And Ordering Policy

| Spectrum class | Value policy | Gap policy | Ordering policy |
| --- | --- | --- | --- |
| Exact repeated leading block selected as a whole | Compare values as a multiset or identical repeated values. | Gap after the selected repeated block must be declared and well separated from the next value. | Per-vector order inside the repeated block is irrelevant. |
| Exact repeated leading block selected partially | Requires containment or principal-angle-to-containing-subspace policy. | The selected `k` cuts through a repeated multiplicity; exact projector equality to one canonical basis is invalid. | Defer until containment semantics are owned. |
| Clustered but separated leading block selected as a whole | Compare ordered values only if gap diagnostics justify it; otherwise use set-based values. | Declare within-cluster gap, post-cluster gap, and tolerance relation. | Raw vector order is unstable; projector metric is primary. |
| Near-zero clustered tail | Requires rank threshold and zero singular-value tolerance. | Tail gaps near numerical precision are not enough for individual triplet claims. | Defer to rank/subspace owner unless a rank policy is present. |

For the accepted Day 8 repeated fixture, the selected block is the whole
three-dimensional leading repeated subspace and the post-block gap is `4`
between singular values `7` and `3`. That keeps the subspace itself stable
while still making basis columns non-unique.

## Diagnostics Required For Day 8

Any Day 8 implementation must print or record:

1. fixture key, dimensions, and `k`;
2. selected singular values and the post-selected-block gap;
3. max singular-value difference or repeated-value multiset difference;
4. left and right projector Frobenius errors;
5. max `A v_i - sigma_i u_i` and `A^T u_i - sigma_i v_i` residuals;
6. U and V orthogonality errors;
7. explicit statement that raw vector equality is not tested.

Recommended first-lane tolerance for the exact repeated diagonal fixture is
`1e-8` for singular values, projector errors, residuals, and orthogonality.
That tolerance is not transferable to clustered, non-diagonal, corpus, or
rank-deficient fixtures.

## Failure Interpretation

| Failure class | Meaning |
| --- | --- |
| Shape/API regression | Returned `m`, `n`, `k`, `sigma`, `U`, or `Vt` violates the fixture contract. |
| Singular-value mismatch | Bounded repeated-spectrum value regression for the named fixture only. |
| Projector mismatch | Product vectors span the wrong left or right leading subspace. |
| Residual mismatch | Published vectors do not satisfy the singular-triplet equations. |
| Orthogonality mismatch | Published basis is not orthonormal enough for projector evidence. |
| Raw vector difference | Not a failure for repeated or clustered spectra. |
| Cluster gap ambiguity | Deferral trigger unless a fixture-specific clustered tolerance and budget policy exists. |
| Near-zero tail ambiguity | Deferral trigger unless rank threshold and null/range policy exist. |

## Day 8 Accepted Path

Day 8 may implement `partial_svd_repeated_diag6_k3_projector` if it keeps this
contract:

| Field | Required value |
| --- | --- |
| Fixture | 6x6 diagonal `diag(7, 7, 7, 3, 2, 1)` |
| `k` | `3` |
| Options | `compute_uv = 1`, `economy = 1` |
| Oracle | Analytic top-3 value multiset and analytic coordinate projector. |
| Metrics | Singular values, left/right projector errors, both triplet residual equations, U/V orthogonality, shape checks. |
| Tolerance | `1e-8` for the exact diagonal repeated fixture only. |
| Maintainer wording | May add a bounded repeated-leading-subspace fixture name only after validation. |
| Public wording | No public solver-selection update. |

If any of these requirements are not met, Day 8 should explicitly defer
repeated-spectrum implementation and carry this policy forward.

## Deferrals

| Deferred lane | Reason | Future owner and promotion gate |
| --- | --- | --- |
| Partial selection inside a repeated block | Requires containment or principal-angle-to-containing-subspace semantics, not exact projector equality. | Future subspace owner must define containment metrics and failure classes. |
| Clustered-spectrum projector fixture | Requires spectral-gap and tolerance policy tied to the fixture and convergence behavior. | Day 8 may defer; Day 13 convergence owner may share budget semantics. |
| Near-zero clustered tail from Day 6 | Needs rank threshold and zero singular-value policy before it can become subspace evidence. | Days 9-10 rank-deficient subspace owner. |
| Corpus clustered spectra | Optional data, conditioning, support tier, residual windows, and runtime policy are not owned by Day 7. | Days 11-12 corpus owner. |
| Public solver-selection wording | Repeated projector policy or one fixture is insufficient for public solver-selection guidance. | Day 14 claim gate. |

## Non-Claim Register

Day 7 does not claim:

- raw singular-vector equality for repeated or clustered spectra;
- stable ordering inside a repeated or clustered singular-value block;
- partial selection through a repeated block;
- clustered-spectrum convergence or budget behavior;
- rank-deficient range/null-space behavior;
- corpus clustered-spectrum parity;
- low-rank optimality;
- public solver-selection wording readiness;
- LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Repeated and clustered candidates are reviewed. | Complete | Candidate table covers exact repeated, partial repeated, clustered diagonal, Day 6 near-zero tail, and corpus clustered lanes. |
| Vector equality is rejected where bases are non-unique or unstable. | Complete | Policy forbids raw vector equality and treats basis rotation/order as non-failure. |
| Projector, principal-angle, residual, gap, and tolerance metrics are defined. | Complete | Subspace, value/gap, diagnostics, tolerance, and failure sections define the required metrics. |
| Day 8 implementation or deferral paths are chosen. | Complete | `partial_svd_repeated_diag6_k3_projector` is accepted as the first path; clustered and ambiguous lanes are deferred unless their missing policies are added. |
| Validation is run for documentation-only changes. | Complete | `git diff --check` and the focused Sprint 130 markdown whitespace scan passed. |
