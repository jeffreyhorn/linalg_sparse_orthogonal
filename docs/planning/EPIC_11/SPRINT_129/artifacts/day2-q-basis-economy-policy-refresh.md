# Sprint 129 Day 2 Q-Basis And Economy Policy Refresh

## Purpose

Day 2 refreshes the Sprint 124 Q-basis/economy policy for Sprint 129 after the
Sprint 125-128 QR residual, nullspace/subspace, threshold, SuiteSparse,
optional-large, minimum-norm, QR-vs-SVD, and helper gates.

The policy keeps Sprint 129 focused on Q-basis, economy, sparse-mode, and
helper ownership. It does not reopen the Sprint 128 residual QR queue unless a
candidate directly supports a Sprint 129 behavior-specific claim and satisfies
the no-reopen promotion gate.

This is a documentation-only policy artifact. No C source, header, Python
helper, Matrix Market, build, maintainer guide, public API, or public wording
files are changed on Day 2.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 129 plan Day 2 | Requires raw Q-column, projection/subspace, economy/sparse-mode, SuiteSparse, and non-claim policy refresh. |
| Sprint 124 Day 6 Q-basis/economy semantics | Source policy for raw Q sign/orientation, projection metrics, economy shape expectations, and owner map. |
| Sprint 124 Day 7 Q-basis/economy decision | Accepted `qr_economy_projector_5x3` as the bounded economy projector baseline. |
| Sprint 125-128 nullspace/subspace artifacts | Provide projector/subspace metric rules for rank-deficient cases. |
| Sprint 128 retrospective residual queue | Keeps compatible/wide residual, near-threshold, SuiteSparse corpus, optional-large, extra exact, and extra QR-vs-SVD work out of Sprint 129 by default. |
| `tests/test_qr.c` | Current owner for Q formation, Q application, economy, sparse-mode, rank, nullspace, and reconstruction behavior. |
| `tests/qr_external_dense_reference.py` | Current bounded QR helper; may grow Q/economy protocol only after fixture and metric gates are explicit. |
| `docs/maintainer_guide.md` | Maintainer evidence table; update only after accepted bounded evidence. |

## Current Q/Economy Evidence Inventory

| Evidence surface | Current owner | Current proof | Sprint 129 interpretation |
| --- | --- | --- | --- |
| Full Q orthogonality and reconstruction | `tests/test_qr.c` | Forms Q, checks `Q^T Q`, and checks reconstruction through QR paths. | Internal invariant baseline; not raw Q parity. |
| Q apply and round trip | `tests/test_qr.c` | Applies `Q`/`Q^T` to vectors, including in-place use. | Product-behavior baseline; candidate only if a reference projection adds distinct trust. |
| Q applied to least-squares RHS | `tests/test_qr.c` | Checks `Q^T*b` and residual-tail interpretation. | Solve-adjacent; should not absorb basis/economy evidence unless explicitly solve-owned. |
| Economy projector `qr_economy_projector_5x3` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py` | Compares `Q Q^T` against `A(A^T A)^{-1}A^T`, shape, and orthogonality. | Completed bounded baseline; do not repeat as new Sprint 129 evidence. |
| Economy solve equivalence | `tests/test_qr.c` | Full and economy solve agree for a tall matrix. | Solve-equivalence baseline; not a basis proof. |
| Economy shape, square, wide, 1x1, and rank-deficient smoke | `tests/test_qr.c` | Shape/rank behavior under economy option. | Candidate source for Day 6-7 only if a distinct shape/projection claim is pinned. |
| Sparse-mode dense-mode agreement | `tests/test_qr.c` | Dense and sparse QR modes agree on selected product metrics. | Backend-mode baseline; no performance or broad sparse QR claim. |
| Rank-deficient nullspace projectors | `tests/test_qr.c`, `tests/qr_external_dense_reference.py` | Projector/subspace comparisons for named rank-deficient fixtures. | Metric policy input for Day 4-5; not raw Q-column evidence. |
| SuiteSparse QR controls | `tests/test_qr.c`, `tests/test_qr_solve.c` | Named corpus controls for QR rank/solve/sparse-mode behavior. | Candidate only after Day 8 support-tier, skip, runtime, and diagnostic gates. |

## Raw Q-Column Candidate Table

| Candidate | Value | Risk | Day 2 disposition |
| --- | --- | --- | --- |
| Full-rank tall raw Q-column fixture | Could test deterministic reflector orientation for one non-degenerate matrix. | High: sign, orientation, and permutation conventions are easy to overclaim. | Candidate for Day 3 only if sign normalization, column order, shape, tolerance, and diagnostic policy are fully pinned. |
| Economy raw Q-column fixture based on `qr_economy_projector_5x3` | Reuses an accepted matrix with known thin-Q shape. | Medium-high: duplicate of completed projector evidence unless raw basis adds distinct trust. | Deferred by default; Day 3 must prove distinct value beyond the completed projector lane. |
| Rank-deficient raw Q-column fixture | Might expose basis behavior in ambiguous rank-deficient cases. | Very high: valid bases can rotate inside deficient subspaces. | Rejected for raw equality; use projector or principal-angle metrics instead. |
| Wide raw Q-column fixture | Could test full `m x m` wide Q orientation. | High: wide output semantics and basis orientation risk. | Deferred to Day 6-7 wide/economy policy; raw equality is not accepted on Day 2. |
| SuiteSparse raw Q-column fixture | Corpus-backed basis check. | Very high: platform/runtime/support-tier and orientation risks. | Rejected unless future metadata and non-oracle basis rules exist. |

## Metric Policy

Raw Q equality is not the default metric. Sprint 129 evidence should use the
least basis-dependent metric that proves the intended behavior.

| Metric | Use when | Required preconditions | Non-claim protected |
| --- | --- | --- | --- |
| Shape | Verifying full/economy/wide/sparse-mode output dimensions. | Expected Q/R shape is pinned before implementation. | Does not prove numerical basis parity. |
| Orthogonality | Verifying formed full or thin Q columns. | Q shape and tolerance are explicit. | Does not prove external dense-library basis equality. |
| Reconstruction | Verifying `A*P ~= Q*R`. | Permutation interpretation and normalization are explicit. | Does not prove raw Q orientation. |
| Projection | Verifying span behavior for a vector or column space. | Fixed vector/reference projection and tolerance are pinned. | Avoids sign and basis-rotation claims. |
| Projector distance | Comparing `Q Q^T` or nullspace/economy projectors. | Subspace dimension, shape, and tolerance are pinned. | Avoids raw column equality. |
| Principal-angle bound | Comparing rank-deficient or repeated subspaces. | Rank/nullity and orthonormal basis construction are pinned. | Avoids unique-basis claims. |
| Raw Q column values | Verifying one deterministic non-degenerate orientation. | Sign normalization, column order, storage layout, permutation effect, fixture degeneracy, and tolerance are pinned. | Must remain fixture-local and non-general. |

Day 3 may accept raw Q equality only for a full-rank, non-degenerate,
fixture-local candidate with explicit sign/orientation rules. Otherwise raw Q
evidence should be explicitly deferred.

## Economy And Sparse-Mode Output Policy

| Surface | Expected interpretation | Evidence gate |
| --- | --- | --- |
| Tall full-rank economy | Thin Q has `m x n` shape and orthonormal columns; R remains the factor-storage shape expected by the API. | Shape plus orthogonality or projector metrics. |
| Tall rank-deficient economy | Shape may remain tied to column count while rank is smaller; basis uniqueness is not meaningful. | Projection/projector metrics only; no raw Q comparison. |
| Square economy | Economy flag should not change the formed Q shape. | Shape/reconstruction checks; raw comparison only if sign rules are pinned. |
| Wide economy | Q is already full row-space size; economy flag should not imply thin-Q semantics. | Day 6-7 must pin `m x m` Q, R shape, projection metric, and non-minimum-norm wording. |
| Sparse-mode Q/economy | Dense-mode and sparse-mode can be compared on product metrics. | Must remain a mode-behavior check and not a performance/backend parity claim. |
| SuiteSparse Q/economy | Corpus controls may exercise shape and product metrics. | Requires corpus availability, support tier, skip behavior, runtime budget, diagnostics, and no product-as-oracle expected values. |

## SuiteSparse Q/Economy Support-Tier Policy

SuiteSparse Q/economy evidence is deferred by default until Day 8-9 unless all
of the following are available before implementation:

1. Matrix path, dimensions, nnz, and support tier.
2. Whether the matrix is checked-in, optional-large, report-only, or absent.
3. Missing-data skip behavior and diagnostics.
4. Runtime budget and platform expectations.
5. Expected Q/R shape, rank claim if any, and metric tolerance.
6. A clear statement that product-observed values are controls, not
   independent oracle values.
7. Focused QR/SuiteSparse validation and full quality gate if `.c` or `.h`
   files change.

## Day 3 Acceptance Gate

Day 3 may implement raw Q-column evidence only if every item is true:

1. The fixture is full-rank, non-degenerate, and not a duplicate of
   `qr_economy_projector_5x3`.
2. Expected full/economy Q shape and storage layout are explicit.
3. Column ordering and permutation interpretation are explicit.
4. Sign normalization is specified before implementation.
5. Value tolerance and failure diagnostics are explicit.
6. The artifact states why raw Q equality adds trust beyond orthogonality,
   reconstruction, projection, or projector metrics.

If any item is missing, Day 3 must explicitly defer raw Q-column evidence.

## No-Reopen Boundary

The following Sprint 128 residual queue items are not Day 2 implementation
candidates:

- compatible zero-residual QR residual evidence;
- wide residual-only QR evidence;
- near-threshold nullspace/subspace evidence;
- SuiteSparse rank-deficient QR corpus evidence;
- additional SuiteSparse or optional-large minimum-norm evidence;
- additional exact underdetermined minimum-norm evidence;
- additional QR-vs-SVD minimum-norm evidence.

They remain end-of-epic queue items unless a later Sprint 129 day proves a
direct Q/economy/helper-specific need and satisfies the promotion gate first.

## Non-Claim Register

Day 2 does not claim:

- raw Q-basis equality, Q-sign, Q-orientation, column ordering, or unique-basis
  parity;
- broad QR factorization, QR solve, compatible solve, wide solve,
  rank-deficient solve, nullspace, minimum-norm, Q-basis, economy, sparse-mode,
  reorder, backend, corpus, optional-data, platform, or performance parity;
- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity;
- SVD-pseudoinverse as a global QR oracle;
- generic helper API or helper consolidation;
- package, ABI, public API, install-header, CMake, Makefile, CI, CTest,
  scalability, memory, or state-of-the-art parity.

## Validation

Day 2 changes documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_129
```

No `.c`, `.h`, Python helper, Matrix Market, build, maintainer guide, public
API, or public wording files changed, so no code quality gate is required.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| No Q/economy candidate can proceed without a metric and tolerance policy. | Complete | Metric table and Day 3 acceptance gate require shape, metric, tolerance, and diagnostics. |
| Raw basis equality is allowed only for fixture-local deterministic cases. | Complete | Raw Q candidates are deferred or gated behind sign, orientation, ordering, layout, and degeneracy requirements. |
| Economy and sparse-mode evidence cannot imply broad QR parity. | Complete | Economy/sparse-mode policy and non-claims fence backend, performance, corpus, and broad parity interpretations. |
