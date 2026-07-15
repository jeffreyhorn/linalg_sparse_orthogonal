# Sprint 124 Day 6 QR Q-Basis and Economy Semantics

## Purpose

Day 6 defines the semantic gates required before Sprint 124 can add QR
Q-basis or economy external evidence. The goal is to prevent raw basis-column
comparisons from being treated as correctness proof when QR factors can differ
by sign, column orientation, valid subspace rotation, economy shape, or
implementation mode.

This is a design artifact only. No C source, header, Python helper, build,
CMake, CTest, workflow, public API, or public wording changes are made by Day
6.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 124 Plan Day 6 | Requires Q-basis/economy inventory, sign/orientation policy, projection/subspace metrics, economy-shape expectations, affected owner map, and non-claims. |
| Sprint 123 Day 7 QR minimum-norm and Q/economy decision | Defers Q/economy external evidence until sign, orientation, projection, subspace, and economy-shape rules are explicit. |
| Sprint 123 Day 5 QR external behavior requirements | Defines Q/economy evidence as distinct from solve residual and minimum-norm evidence. |
| Sprint 122 QR external-lane requirements | Rejects raw Q-basis external comparison without basis semantics. |
| `include/sparse_qr.h` | Defines `sparse_qr_apply_q`, `sparse_qr_form_q`, economy options, and Q shape contracts. |
| `tests/test_qr.c` | Current owner for Q application, orthogonality, economy, sparse-mode, rank-deficient, wide, square, and SuiteSparse QR behavior. |
| `tests/test_qr_solve.c` | Current owner for QR solve external fixtures; should not absorb basis/economy semantics unless a fixture is solve-owned. |
| `tests/qr_external_dense_reference.py` | Current standard-library helper for bounded QR least-squares, rank, and minimum-norm references; not yet a basis/projection helper. |
| `docs/maintainer_guide.md` | Current evidence table and non-claim owner for QR external trust boundaries. |

## Current Evidence Inventory

| Evidence Surface | Current Owner | Current Proof | Externalized? | Day 6 Interpretation |
| --- | --- | --- | --- | --- |
| Full Q reconstruction and orthogonality | `tests/test_qr.c` | Forms Q and checks `Q^T Q ~= I`; reconstruction paths check `A*P = Q*R`. | No | Strong internal invariant evidence, not basis parity. |
| Q application and round trip | `tests/test_qr.c` | Applies `Q^T` then `Q` to basis vectors and arbitrary vectors, including in-place use. | No | Product behavior evidence; suitable future external metric input. |
| Q applied to least-squares RHS | `tests/test_qr.c` | Checks `Q^T*b` round trip and residual-tail interpretation. | No | Solve-adjacent behavior, but still internal Q owner. |
| Tall Q orthogonality | `tests/test_qr.c` | Full Q for an 8x5 matrix is orthogonal. | No | Candidate for projection/orthogonality external check if shape is pinned. |
| Wide Q orthogonality | `tests/test_qr.c` | Wide 3x6 matrix forms `3 x 3` Q and checks orthogonality. | No | Must preserve wide-case full `m x m` shape. |
| Economy solve equivalence | `tests/test_qr.c` | Full and economy solve agree on a 50x10 tall matrix. | No | Solve-equivalence evidence; not a basis-comparison proof. |
| Economy thin-Q orthogonality | `tests/test_qr.c` | Economy Q for tall 20x5 matrix has `Q^T Q ~= I_5`. | No | Best bounded future shape/orthogonality candidate. |
| Economy R shape | `tests/test_qr.c` | Tall economy R is `n x n` for a 30x5 matrix. | No | Shape-only evidence can be externalized without basis equality. |
| Economy square/wide/1x1 behavior | `tests/test_qr.c` | Economy equals full behavior for square and wide cases; 1x1 smoke coverage exists. | No | Shape rules must explicitly distinguish tall, square, wide, and singleton. |
| Economy rank-deficient behavior | `tests/test_qr.c` | Tall 20x4 duplicate-column fixture detects rank deficiency with economy enabled. | No | Must not imply rank-deficient basis uniqueness. |
| Sparse-mode Q orthogonality and mode agreement | `tests/test_qr.c` | Sparse-mode Q is orthogonal and agrees with dense mode across selected fixtures. | No | Backend-mode evidence; external proof must not imply performance or implementation parity. |

## Candidate Table

| Candidate | Evidence Class | Value | Risk | Day 6 Gate |
| --- | --- | --- | --- | --- |
| `qr_economy_shape_tall_5x3` | Shape-only | Confirms economy `Q` is `m x n` and `R` is `n x n` for full-column-rank tall input. | Low | Acceptable only if output protocol reports dimensions, not raw basis values. |
| `qr_economy_orthogonality_tall_5x3` | Orthogonality | Confirms thin-Q columns satisfy `Q^T Q ~= I`. | Low-medium | Needs dimension protocol and tolerance; no external raw-basis reference required. |
| `qr_q_projection_tall_5x3` | Projection | Confirms `Q Q^T b` or residual-tail projection agrees with reference projection. | Medium | Needs projection-vector or residual metric and a non-degenerate full-column-rank fixture. |
| `qr_q_apply_roundtrip_tall_5x3` | Product behavior | Confirms `Q(Q^T b) ~= b` for full Q. | Medium | Product-internal metric; external helper adds little unless it supplies a reference projection. |
| `qr_basis_columns_tall_5x3` | Raw basis comparison | Directly compares Q columns. | High | Reject unless sign, ordering, orientation, and uniqueness are all pinned. |
| `qr_basis_rankdef_duplicate_5x4` | Rank-deficient basis/subspace | Compares basis or nullspace under rank deficiency. | Very high | Defer to a subspace owner using projection/principal-angle metrics. |
| `qr_economy_rankdef_shape_20x4` | Rank-deficient economy shape/rank | Confirms shape and rank under economy with duplicate columns. | Medium-high | Requires rank threshold from Day 2 and must not compare raw bases. |
| `qr_sparse_mode_q_projection` | Backend-mode projection | Compares dense-mode and sparse-mode projection behavior. | Medium-high | Must remain a backend-specific mode check and not claim performance parity. |
| `qr_suite_sparse_q_economy` | Corpus Q/economy evidence | Extends Q/economy proof to named corpus matrices. | High | Needs corpus availability, skip policy, time budget, platform boundaries, and support-tier wording. |

## Sign and Orientation Policy

Raw Q-vector comparison is not the default acceptable metric.

If a future fixture compares raw Q columns, it must define all of these rules:

1. The fixture must be full rank, well separated, and non-degenerate enough that
   the compared column ordering is meaningful.
2. Each compared column must use an explicit sign normalization, such as
   forcing the largest-magnitude entry in that column to be positive.
3. The orientation must name whether `Q` columns are stored in column-major
   order by `sparse_qr_form_q`, whether the comparison is against full `m x m`
   or economy `m x k` Q, and whether column permutations affect only `R`/`P`
   rather than Q columns.
4. Any column mismatch after sign normalization must be diagnosed separately
   from shape, orthogonality, projection, and rank mismatches.
5. Rank-deficient, repeated, clustered, or near-dependent cases must not use
   raw Q-vector comparison.

Preferred future evidence should avoid raw basis equality and use
orthogonality, reconstruction, projection, or subspace metrics.

## Projection and Subspace Metric Policy

Basis-dependent QR external evidence should use metrics that remain valid under
sign flips and valid basis rotations.

| Metric | Formula | Use When | Failure Means | Avoids |
| --- | --- | --- | --- | --- |
| Orthogonality | `||Q^T Q - I||_max` for the formed Q shape | Verifying full/economy Q has orthonormal columns. | Product Q formation or reflector application drift. | Raw column sign and orientation. |
| Reconstruction | `||A*P - Q*R|| / ||A||` | Verifying factorization consistency. | Factorization or permutation mismatch. | Direct Q reference parity. |
| Projection residual | `||Q Q^T b - p_ref||` or residual-tail norm agreement | Verifying the span represented by Q on a vector. | Projection-space mismatch or helper protocol mismatch. | Column sign and basis rotation. |
| Projector distance | `||Q1 Q1^T - Q2 Q2^T||` | Comparing subspaces between product and reference bases. | Subspace mismatch, not sign mismatch. | Raw basis equality. |
| Principal angle bound | `max sin(theta_i)` from a projector or Gram comparison | Rank-deficient or repeated-subspace cases. | Subspace disagreement beyond tolerance. | Basis rotation ambiguity. |

Policy decisions:

- Full-column-rank tall fixtures may use shape plus orthogonality as a first
  external lane because this avoids dense-library basis orientation.
- Projection evidence should use a fixed vector `b` and compare projected
  outputs or residual-tail norms, not individual Q columns.
- Rank-deficient Q/nullspace evidence must wait for projector or
  principal-angle helper ownership.
- Sparse-mode or backend-mode evidence must compare product metrics only and
  remain explicit that it is a mode-behavior proof, not a performance claim.

## Economy Shape Expectations

| Matrix Shape | Option | Expected Formed Q Shape | Expected R Shape | Notes |
| --- | --- | --- | --- | --- |
| Tall full rank, `m > n` | `economy = 0` | `m x m` | `n x n` under current factor storage | Full Q includes orthogonal complement columns. |
| Tall full rank, `m > n` | `economy = 1` | `m x n` | `n x n` | Thin Q columns should be orthonormal; no claim about omitted complement basis. |
| Tall rank-deficient, `m > n` | `economy = 1` | `m x n` | `n x n` | Rank may be `< n`; do not compare basis columns in the deficient subspace. |
| Square, `m == n` | `economy = 0` or `1` | `m x m` | `n x n` | Economy flag should not change Q shape. |
| Wide, `m < n` | `economy = 0` or `1` | `m x m` | `m x n` factor-storage boundary | Economy flag has no thin-Q effect because Q is already `m x m`. |
| Singleton, `m = n = 1` | `economy = 0` or `1` | `1 x 1` | `1 x 1` | Smoke shape only; does not add meaningful external basis evidence. |

Any future external fixture must emit or assert the exact expected shape before
checking values. A shape mismatch is a protocol or product-contract failure,
not a numerical tolerance failure.

## Affected Owners

| Surface | Day 6 Rule |
| --- | --- |
| `tests/test_qr.c` | Remains the primary owner for Q application, Q orthogonality, economy shape, economy solve equivalence, sparse-mode, rank, nullspace, and reconstruction. |
| `tests/test_qr_solve.c` | Continues owning solve-oriented external QR fixtures. It should not host Q-basis/economy tests unless the fixture is explicitly solve-owned. |
| `tests/qr_external_dense_reference.py` | May grow a separate basis/projection protocol only after Day 7 accepts a bounded evidence lane. Do not overload the least-squares output protocol with Q matrices. |
| `tests/test_qr_helpers.h` | May host future behavior-specific measurements such as `tf_qr_projector_distance` or `tf_qr_orthogonality_error`; helper names must preserve the metric meaning. |
| `docs/maintainer_guide.md` | Should only list Q/economy evidence after implementation validates a named fixture; until then, preserve Q-basis/economy non-claims. |
| `docs/solver_selection.md` | No Day 6 basis/economy wording update is justified. User-facing guidance should not mention external Q/economy parity. |

## Day 7 Decision Gates

Day 7 may accept a bounded Q/economy external lane only if it can answer:

1. Is the lane shape-only, orthogonality, reconstruction, projection, or raw
   basis comparison?
2. If raw basis values are compared, what sign, orientation, ordering, and
   degeneracy policy makes that comparison meaningful?
3. If subspaces are compared, what projector or principal-angle metric owns the
   comparison?
4. What exact Q/R dimensions are expected for the matrix shape and economy
   option?
5. Does the fixture avoid duplicating existing deterministic `tests/test_qr.c`
   evidence, or does it add a distinct external trust boundary?
6. Are Windows/helper skip behavior, failure diagnostics, and non-claims
   explicit?

The lowest-risk Day 7 implementation candidate is a shape-plus-orthogonality
economy fixture such as `qr_economy_shape_orthogonality_5x3`, provided the
helper protocol reports dimensions and scalar residuals instead of raw Q
columns. Projection or subspace metrics should remain deferred unless Day 7
adds a dedicated behavior-specific protocol.

## Non-Claim Register

Day 6 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  broad external dense-library parity;
- broad QR factorization parity;
- raw Q-basis equality, Q-sign, Q-orientation, or unique basis parity;
- rank-deficient, repeated, clustered, or near-dependent subspace parity;
- economy-mode external oracle coverage;
- sparse-mode, reorder, backend, or performance parity;
- solve residual, minimum-norm, rank-deficient, nullspace, or pseudoinverse
  coverage beyond previously named fixtures;
- package, ABI, platform, public API, CMake, Makefile, CI, or CTest expansion;
- scalability, memory behavior, or state-of-the-art behavior.

## Validation Notes

Day 6 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_124`

The branch already contains earlier Sprint 124 `.c` and Python helper changes;
Day 5 ran the required full `make format && make lint && make test` gate after
those code changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 3 has explicit semantic gates. | Complete for Day 6 | Sign/orientation, projection/subspace, economy-shape, owner, and Day 7 decision gates are defined. |
| Basis equality is not claimed where only subspace equivalence is justified. | Complete | Raw Q-vector comparison is rejected by default; projection/projector/principal-angle metrics are required for ambiguous cases. |
| Economy-mode expectations are visible before implementation. | Complete | Economy shape table defines tall, rank-deficient tall, square, wide, and singleton expectations. |
