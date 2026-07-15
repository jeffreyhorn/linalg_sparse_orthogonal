# Day 8 Partial-SVD Vector and Subspace Semantics

## Purpose

Define the semantic gates required before Sprint 124 extends partial-SVD
evidence beyond top-k singular-value comparisons. This artifact keeps singular
value, vector, subspace, residual, convergence, rank-deficient, and low-rank
evidence separate so Day 9 can either accept one bounded lane or defer the lane
with a clear proof owner.

## Inputs Reviewed

| Input | Relevance |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_124/PLAN.md` Day 8-9 | Defines the vector/subspace semantics and decision scope. |
| `docs/planning/EPIC_11/SPRINT_123/artifacts/day9-partial-svd-external-semantics-design.md` | Prior semantic baseline for partial-SVD external lanes. |
| `tests/test_svd.c` | Current partial-SVD value, vector, residual, rectangular, rank-deficient, low-rank, and timing evidence owner. |
| `tests/test_svd_partial_helpers.h` | Current bounded external partial-SVD singular-value fixture owner. |
| `tests/svd_external_dense_reference.py` | Current dense-reference helper protocol for SVD and partial-SVD value fixtures. |
| `docs/maintainer_guide.md` | Public evidence and non-claim wording that must not overstate vector/subspace coverage. |

## Current Evidence Inventory

| Evidence family | Current evidence | Current proof shape | Boundary |
| --- | --- | --- | --- |
| External top-k values | `partial_svd_diag6_k2`, `partial_svd_tall_diag_8x5_k3` | Compares ordered leading singular values against pure-Python dense references | Value-only; no vector, subspace, convergence, or low-rank claim. |
| Internal top-k values | Partial-SVD diagonal, dense, tall, wide, SuiteSparse, and rank-deficient value tests | Compares to deterministic expectations or this library's full SVD | Regression evidence, not independent dense-library parity. |
| Vector availability and orthogonality | Partial-SVD vector availability, no-vector, and orthogonality tests | Checks output presence, dimensions, and local orthogonality | Internal only; does not establish external vector orientation. |
| Singular-triplet residuals | `A*v ~= sigma*u`, reconstruction, and SuiteSparse vector checks | Checks product residuals and reconstruction consistency | Good behavior evidence, but still local to the implementation. |
| Full-SVD vector comparison | Partial-SVD vectors compared against this library's full SVD | Cross-checks internal implementations | Sign and basis orientation remain internal. |
| Rectangular vector behavior | Wide and rectangular low-rank reconstruction tests | Confirms rectangular shape and reconstruction behavior | Does not prove external subspace or optimality parity. |
| Rank-deficient behavior | Rank-deficient partial-SVD value tests | Exercises zero or near-zero singular slots | Threshold and subspace semantics are not externalized. |
| Low-rank approximation | Dense and sparse low-rank tests | Checks fixture-specific low-rank reconstruction | Separate owner; not global Eckart-Young optimality evidence. |
| Timing and convergence smoke | Timing and corpus partial-SVD tests | Bounded smoke checks | Not a performance or convergence-budget guarantee. |

## Candidate Table

| Candidate | Evidence class | Metric owner | Day 9 posture |
| --- | --- | --- | --- |
| `partial_svd_vector_residual_diag6_k2` | Singular-triplet residual | Product residuals and orthogonality, not raw vector equality | Acceptable only if it stays residual-only and documents sign irrelevance. |
| `partial_svd_vector_residual_tall8x5_k3` | Rectangular singular-triplet residual | Product residuals, output dimensions, and orthogonality | Acceptable only after the square residual lane is stable. |
| `partial_svd_subspace_repeated_diag6_k3` | Repeated-spectrum subspace | Projector or principal-angle distance | Defer until subspace helper protocol exists. |
| `partial_svd_projector_clustered_diag6_k3` | Clustered-spectrum subspace | Projector distance with explicit gap and tolerance | Defer until convergence and near-tie semantics are owned. |
| `partial_svd_rankdef_subspace_6x4_k3` | Rank-deficient subspace | Rank threshold plus projector distance | Defer; threshold and zero-space semantics are not yet externalized. |
| `partial_svd_lowrank_projection_rect5x4_k2` | Low-rank approximation | Reconstruction error and projection residual | Defer to low-rank owner, not top-k singular-value owner. |
| SuiteSparse partial-SVD vector residual lane | Corpus residual smoke | Fixture-specific residual windows | Defer; optional corpus availability and failure meaning differ from dense fixtures. |

## Sign-Invariant Vector Policy

- Direct raw equality of singular vectors is not a valid external comparison.
- For well-separated singular values, vector evidence may align signs with a
  reference by the sign of the dot product before reporting a componentwise
  diagnostic, but the primary pass/fail metric should remain residual-based.
- Vector residual evidence must check both sides of each triplet when both
  vector families are returned:
  - `||A v_i - sigma_i u_i||`
  - `||A^T u_i - sigma_i v_i||`
  - `||U^T U - I||`
  - `||V^T V - I||`
- Fixture ordering must be explicit. If singular values are repeated or
  clustered, per-vector comparison is invalid even after sign alignment.
- A sign mismatch alone is never a failure. A residual, orthogonality, shape,
  or ordering mismatch is the meaningful failure.

## Projection and Subspace Metric Policy

- Repeated, clustered, rank-deficient, or otherwise basis-ambiguous fixtures
  must compare subspaces rather than individual singular vectors.
- Preferred metrics are projector distance or principal-angle distance:
  - left projector: `P_U = U_k U_k^T`
  - right projector: `P_V = V_k V_k^T`
  - projector error: `||P_product - P_reference||`
  - principal-angle signal: singular values of `U_reference^T U_product` or
    `V_reference^T V_product`
- Projection evidence must state whether it covers the left subspace, right
  subspace, or both.
- Output dimension checks are required before any projector or angle metric.
- Subspace agreement is separate from top-k singular-value agreement; passing
  singular values cannot stand in for vector or subspace evidence.

## Residual, Tolerance, Skip, and Failure Policy

| Policy area | Rule |
| --- | --- |
| Singular values | Existing bounded external value tolerance remains `1e-8` unless a fixture documents a narrower or looser reason. |
| Small exact vector residual fixtures | Target `1e-8` for product residual and orthogonality when the fixture is diagonal or otherwise well conditioned. |
| Corpus or ill-conditioned fixtures | Must state fixture-specific residual windows before implementation; do not inherit dense exact tolerances. |
| Rank-deficient fixtures | Must state rank threshold, zero singular-value tolerance, and whether `k` crosses numerical rank. |
| Repeated or clustered spectra | Must use projector/subspace metrics and explicit gap/tie interpretation. |
| Missing `python3` | External helper absence remains a skip through the existing helper harness. |
| Helper `ERROR` output | Protocol or reference-generation failure is a test failure, not a skip. |
| Windows behavior | Preserve the existing explicit skip unless the external helper policy changes with platform proof. |
| Timing or iteration budget | Timing smoke is not convergence proof; convergence lanes need iteration, tolerance, and failure-budget semantics. |

Failure interpretation must name the class of failure: helper protocol error,
shape/API regression, singular-value regression, vector residual regression,
orthogonality regression, subspace/projection regression, threshold-policy
failure, convergence-budget miss, or unsupported optional fixture.

## Top-k Value Evidence Versus Vector/Subspace Evidence

Existing bounded external partial-SVD lanes prove only that the requested
leading singular values match fixed dense-reference values for fixed fixtures.
They do not prove singular-vector orientation, sign behavior, repeated-spectrum
subspace behavior, low-rank optimality, convergence budgets, corpus parity, or
broad external-library parity.

Any Day 9 implementation must keep a separate assertion owner for vector or
subspace behavior. It must not add vector/subspace language to the maintainer
guide unless the implemented checks actually exercise that behavior.

## Day 9 Decision Gates

Day 9 may implement a bounded partial-SVD vector/subspace lane only if it can
answer all of the following before editing code:

1. Which evidence class is being tested: vector residual, subspace projection,
   rank-deficient threshold, repeated/clustered spectrum, convergence budget,
   or low-rank reconstruction?
2. What fixture key, matrix, `k`, output dimensions, and tolerance define the
   lane?
3. Are signs, ordering, repeated values, clustered values, and rank thresholds
   irrelevant or explicitly handled?
4. Does the lane duplicate `partial_svd_diag6_k2`,
   `partial_svd_tall_diag_8x5_k3`, or full-SVD external value evidence?
5. What exact failure class should a failed assertion report?
6. Which maintainer-guide wording, if any, is justified by the new evidence?
7. Which focused helper, test executable, and full quality gates are required?

The lowest-risk Day 9 implementation candidate is
`partial_svd_vector_residual_diag6_k2` if it remains a residual-only lane with
explicit sign-invariant semantics. Repeated-spectrum, clustered-spectrum,
rank-deficient subspace, convergence-budget, SuiteSparse corpus, and low-rank
optimality lanes should be deferred unless their metric and failure policies
are narrowed first.

## Non-Claim Register

Day 8 preserves the following non-claims:

- no LAPACK, SciPy, NumPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  vendor-backend parity claim;
- no broad partial-SVD external parity claim;
- no singular-vector external parity claim until a residual/vector lane is
  implemented and documented;
- no repeated-spectrum, clustered-spectrum, or rank-deficient subspace parity
  claim;
- no convergence-budget guarantee;
- no low-rank global optimality claim;
- no package, ABI, platform, performance, scalability, public API, or
  state-of-the-art claim.

## Validation

Day 8 is documentation-only. Validation is limited to `git diff --check` and a
focused trailing-whitespace scan of Sprint 124 documentation files.
