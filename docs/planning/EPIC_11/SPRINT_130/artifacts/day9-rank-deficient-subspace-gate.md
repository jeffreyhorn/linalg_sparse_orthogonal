# Sprint 130 Day 9 - Rank-Deficient Subspace Gate

## Purpose

Day 9 defines the rank, nullity, projection, and tolerance policies required
before Sprint 130 can accept rank-deficient partial-SVD subspace evidence.

Rank-deficient fixtures are not just value fixtures. Once `k` reaches zero or
near-zero singular slots, individual vectors are basis-ambiguous, rank
thresholds become fixture policy, and range/null-space evidence must be kept
separate from reconstruction, minimum-norm, repeated-spectrum, clustered-tail,
and solver-selection claims.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 130 Day 2 metric map | Requires rank/nullity thresholds and projector metrics before rank-deficient evidence. |
| Sprint 130 Day 8 deferral | Shows that projector evidence must fail closed when the current partial-SVD path does not satisfy the subspace contract. |
| Sprint 124 Day 10 residual scenario matrix | Names `partial_svd_rankdef_diag_6x4_k3` as deferred pending rank threshold and range/null-space policy. |
| Sprint 124 Day 11 deferral package | Requires rank threshold, zero singular-value tolerance, and range/null-space projector ownership. |
| `tests/svd_external_dense_reference.py` | Owns full-SVD rank-deficient singular-value fixture `svd_rankdef_duplicate_5x4`; no vector/projector output exists. |
| `tests/test_svd.c` | Owns external full-SVD rank-deficient value evidence, SVD rank threshold tests, and SVD/QR rank consistency. |
| `tests/test_svd_partial_helpers.h` | Owns internal partial-SVD rank-deficient value checks and partial-SVD residual helpers. |
| `tests/test_svd_helpers.h` | Owns reusable exact rank-deficient 5x4 duplicate-column fixture. |
| `docs/maintainer_guide.md` | Evidence wording owner; Day 9 does not add rank-deficient partial-SVD claims. |

## Current Rank-Deficient Coverage

| Evidence | Fixture | Current metric | Boundary |
| --- | --- | --- | --- |
| `svd_rankdef_duplicate_5x4` | External full-SVD 5x4 value fixture | Full-SVD singular values; smallest value expected near zero. | Full-SVD value evidence only; not partial-SVD subspace evidence. |
| `test_svd_rank_deficient` | 5x4 duplicate-column fixture from `tf_svd_make_rank_deficient_colpair_5x4` | `sparse_svd_rank(A, 0.0) == 2`. | Rank API evidence; no partial-SVD vectors or projectors. |
| `test_svd_rank_nearly_singular` | 3x3 diagonal with `1e-14` tail | Rank changes from 3 to 2 under explicit `1e-12` tolerance. | Threshold policy smoke; no subspace evidence. |
| `test_svd_rank_diagonal_threshold_fixture` | 4x4 diagonal `diag(1, 1e-8, 1e-12, 0)` | Expected ranks under `1e-14`, `1e-10`, and `1e-6`. | Rank threshold evidence only. |
| `test_svd_qr_rank_dependent_row_fixture` | 4x3 dependent-row fixture | SVD and QR rank agree at `1e-10`. | Cross-solver rank consistency; no subspace or solver-selection claim. |
| `test_partial_svd_rank_deficient` | 6x4 duplicate-column fixture, requested `k=4` | Product partial values compared to product full-SVD values; trailing values expected zero. | Internal partial-SVD value coverage only; no range/null-space projector. |
| Day 6 near-zero 10x8 tail | Nonsymmetric fixture with values 4-7 near zero | Explicitly deferred from vector residual evidence. | Needs rank threshold and range/null-space split before promotion. |

## Candidate Table

| Candidate | Fixture | Distinct trust value | Required metrics | Oracle | Day 9 decision |
| --- | --- | --- | --- | --- | --- |
| `partial_svd_rankdef_range_projector_5x4_k2` | Exact duplicate-column 5x4 fixture from `tf_svd_make_rank_deficient_colpair_5x4`, `k=2` | Checks the positive-rank range subspaces without entering zero singular slots. | External or analytic positive singular values, rank `2`, left/right range projector errors, triplet residuals, U/V orthogonality, shape checks. | Analytic right range projector from duplicate-column structure plus product/full-SVD or helper value evidence; left projector may need product full-SVD or external projector protocol. | Accept as Day 10 candidate only if oracle ownership is narrowed before implementation. |
| `partial_svd_rankdef_diag6x4_k2_range_projector` | Rectangular diagonal `diag(9, 6, 0, 0)` in 6x4 shape, `k=2` | Lower-risk exact range projector fixture with no zero singular slots requested. | Positive singular values, rank `2`, analytic left/right range projectors, triplet residuals, orthogonality, shape checks. | Fully analytic values and coordinate projectors. | Preferred Day 10 lane if implementation proceeds. |
| `partial_svd_rankdef_diag6x4_k3_zero_crossing` | Same diagonal 6x4 fixture, `k=3` | Would cross into the zero singular subspace. | Rank `2`, zero singular tolerance, range projector, null-space or zero-space policy, residuals, orthogonality. | Analytic, but zero vector/basis semantics must be defined. | Defer; zero-crossing policy is not first-lane evidence. |
| Upgrade `test_partial_svd_rank_deficient` | Existing 6x4 duplicate-column internal fixture, `k=4` | Could add residual/projector diagnostics to existing coverage. | Rank threshold, external/analytic projector ownership, zero-space handling for `k=4`. | Product full-SVD today; no independent projector oracle. | Defer; would silently convert internal value regression into broader subspace evidence. |
| Day 6 near-zero tail `partial_svd_nonsym_rect10x8_k4` | Existing nonsymmetric 10x8 fixture, `k=4` | Would test near-zero rank boundary in non-diagonal fixture. | Threshold, rank/nullity, clustered-tail tolerance, range/null split, convergence semantics. | Dense-reference values only. | Defer to rank/convergence owner after first exact rank-deficient lane. |

## Rank, Nullity, And Threshold Policy

| Policy area | Rule |
| --- | --- |
| Numerical rank | Must be declared per fixture before implementation. A default rank result from product code is not enough. |
| Positive singular values | Must have a positive lower bound or exact analytic values. |
| Zero singular values | Must have an explicit zero tolerance, such as `1e-8` for exact diagonal fixtures, and must not be compared with a positive-rank tolerance accidentally. |
| Nullity | Must state whether nullity is checked. If checked, name left nullity, right nullity, or both. |
| Requested `k` | First accepted lane should keep `k == rank` to prove range subspace before crossing into zero-space behavior. |
| Crossing rank with `k > rank` | Requires zero-space basis policy, null-space projector policy, and failure interpretation before implementation. |
| Near-zero clustered tails | Require rank threshold plus clustered-tail policy; they must not inherit exact-zero fixture tolerances. |

## Projection And Residual Metric Policy

Rank-deficient subspace evidence must state which subspace is being checked:

- left positive range subspace;
- right positive row/range subspace;
- left null space;
- right null space;
- or a two-way projection residual across one of those spaces.

For the first Day 10 lane, prefer range projector evidence with `k == rank`.
That avoids zero-space publication ambiguity and still adds proof value beyond
singular-value-only rank-deficient tests.

Required metrics for an accepted range lane:

1. returned `m`, `n`, `k`, `sigma`, `U`, and `Vt` shape checks;
2. expected numerical rank and positive singular values;
3. left and right range projector Frobenius errors;
4. max `A v_i - sigma_i u_i` and `A^T u_i - sigma_i v_i` residuals;
5. U and V orthogonality errors;
6. explicit statement that null-space basis equality is not tested.

## Preferred Day 10 Path

Day 10 may implement `partial_svd_rankdef_diag6x4_k2_range_projector` if it
keeps this contract:

| Field | Required value |
| --- | --- |
| Fixture | 6x4 diagonal with nonzero diagonal entries `9` and `6`, remaining diagonal entries zero, and two extra zero rows. |
| Expected rank | `2` |
| Requested `k` | `2` |
| Nullity | Right nullity is `2`, but it is not asserted in the first lane. Left nullity is `4`, but it is not asserted in the first lane. |
| Oracle | Analytic positive singular values `[9, 6]` and analytic left/right coordinate range projectors onto coordinates `0..1`. |
| Metrics | Top-2 values, left/right range projector errors, both triplet residual equations, U/V orthogonality, shape checks. |
| Tolerance | `1e-8` for exact diagonal values, range projectors, residuals, and orthogonality. |
| Maintainer wording | May add a bounded rank-deficient range-projector fixture only after validation. |
| Public wording | No public solver-selection update. |

If Day 10 cannot satisfy the projector contract, it should explicitly defer
implementation as Day 8 did.

## Deferrals

| Deferred lane | Reason | Future owner and promotion gate |
| --- | --- | --- |
| `k > rank` zero-crossing evidence | Needs zero singular-value tolerance, zero-vector/basis semantics, and null-space projector policy. | Future rank/null-space owner must define zero-space publication and failure classes. |
| Duplicate-column 5x4 range projector | Useful but needs a clear left projector oracle; product full-SVD alone would be internal consistency. | Future external/projector owner may add helper projector output or analytic derivation. |
| Existing `test_partial_svd_rank_deficient` upgrade | Current test requests `k=4` and crosses into zero slots; upgrading it would mix value, rank, range, and null-space evidence. | Future owner should split range-only and zero-crossing tests before changing it. |
| Day 6 near-zero nonsymmetric tail | Near-zero values are clustered and require rank threshold plus convergence interpretation. | Future rank/convergence owner. |
| Minimum-norm and pseudoinverse behavior | Separate solver behavior, not partial-SVD subspace evidence. | Minimum-norm/pseudoinverse owner. |
| Solver-selection wording | Rank-deficient subspace gate alone does not justify public guidance changes. | Day 14 claim gate. |

## Non-Claim Register

Day 9 does not claim:

- rank-deficient partial-SVD range or null-space projector correctness;
- zero singular-vector or null-space basis stability;
- partial-SVD behavior when `k > rank`;
- near-zero clustered-tail behavior;
- minimum-norm or pseudoinverse correctness;
- broad rank-deficient solver robustness;
- public solver-selection wording readiness;
- LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity.

## Validation

Day 9 changes documentation only. Required validation:

1. `git diff --check`
2. focused Sprint 130 markdown trailing-whitespace scan

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Rank and nullity expectations are explicit before implementation. | Complete | Rank/nullity policy and preferred Day 10 path define rank, `k`, and left/right nullity boundaries. |
| Projection metrics are preferred where bases are non-unique. | Complete | Day 10 candidate requires left/right range projectors, residuals, and orthogonality; raw basis equality is excluded. |
| No rank-deficient evidence implies broad solver robustness. | Complete | Non-claim register and deferral table preserve minimum-norm, pseudoinverse, zero-crossing, near-zero, and solver-selection boundaries. |
| Documentation validation is run. | Complete | `git diff --check` and the focused Sprint 130 markdown whitespace scan passed. |
