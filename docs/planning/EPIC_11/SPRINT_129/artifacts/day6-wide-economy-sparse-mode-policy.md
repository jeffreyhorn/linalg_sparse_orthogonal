# Sprint 129 Day 6 Wide Economy And Sparse-Mode Policy

## Purpose

Day 6 defines the output-shape, metric, and non-claim policy for Sprint 129
wide economy and sparse-mode Q/economy evidence.

The policy keeps Day 7 focused on bounded Q/economy behavior. It does not
reopen residual-only, minimum-norm, near-threshold, SuiteSparse corpus,
optional-large, raw-basis, or broad sparse QR parity work.

This is a documentation-only policy artifact. No C source, header, Python
helper, Matrix Market, build, maintainer guide, public API, or public wording
files are changed on Day 6.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 129 Day 2 Q-basis/economy policy | Defines metric order and states that wide economy must pin `m x m` Q, R shape, projection metric, and non-minimum-norm wording. |
| Sprint 129 Day 3 raw Q decision | Defers wide and sparse-mode raw Q values to the economy/sparse-mode gates and rejects raw equality by default. |
| Sprint 129 Day 4-5 rank-deficient gate and evidence | Keeps wide rank-deficient economy/nullspace interaction separate from completed nullspace projector and Q-application evidence. |
| Sprint 128 Day 5 wide subspace evidence | Completed `qr_rankdef_wide_3x5_nullspace_subspace`; Day 7 must not repeat it as economy or sparse-mode evidence. |
| `tests/test_qr.c` economy tests | Current owner for tall economy solve equivalence, thin-Q orthogonality, `qr_economy_projector_5x3`, square economy, R shape, rank-deficient smoke, wide economy smoke, 1 x 1 economy, and `nos4` economy equivalence. |
| `tests/test_qr.c` sparse-mode tests | Current owner for dense/sparse QR solve agreement across small, SuiteSparse, tall, wide, rank-deficient, 1 x 1, Q orthogonality, reconstruction, reorder, and shape-adjacent scenarios. |

## Current Evidence Fence

Day 6 treats the following evidence as already complete. Day 7 must not
repackage these checks as new Sprint 129 evidence unless it adds a distinct
shape or product-metric claim.

| Completed evidence | Current owner | Existing proof | Duplicate fence |
| --- | --- | --- | --- |
| Wide full-Q orthogonality | `test_q_orthogonality_wide` | Forms a 3 x 3 Q for a 3 x 6 matrix and checks `Q^T Q`. | Do not add raw Q equality or another generic wide orthogonality smoke. |
| Tall economy projector | `qr_economy_projector_5x3` | Checks thin-Q shape, R shape, orthogonality, and `Q Q^T` projector against an external reference. | Do not repeat as raw Q or generic economy projector evidence. |
| Tall economy solve equivalence | `test_economy_solve_tall` | Full and economy QR solve outputs and residuals agree on a 50 x 10 matrix. | Do not treat as Q-basis proof or minimum-norm behavior. |
| Economy shape smoke | `test_economy_square`, `test_economy_r_shape`, `test_economy_rank_deficient`, `test_economy_wide`, `test_economy_1x1`, `test_economy_nos4` | Verifies existing shape/rank/solve expectations for square, tall, rank-deficient, wide, scalar, and checked-in square matrix cases. | Day 7 needs a stronger shape/product metric than another smoke. |
| Sparse-mode dense/sparse solve agreement | `test_sparse_mode_basic`, `test_sparse_mode_nos4`, `test_sparse_mode_tall`, `test_sparse_mode_wide`, `test_sparse_mode_rank_deficient`, `test_sparse_mode_west0067`, `test_sparse_mode_bcsstk04`, `test_sparse_mode_1x1` | Dense and sparse QR modes agree on solution, residual, and rank for named cases. | Do not imply backend, performance, or broad sparse QR parity. |
| Sparse-mode Q orthogonality | `test_sparse_mode_q_ortho` | Forms sparse-mode Q for a 5 x 3 fixture and checks orthogonality. | Do not repeat as raw sparse-mode Q equality. |
| Sparse-mode reconstruction | `test_sparse_mode_reconstruction` and adjacent sparse-mode tests | Dense and sparse modes agree on reconstruction/product metrics for existing controls. | Day 7 must pin a new product metric or defer. |
| Wide rank-deficient subspace | `qr_rankdef_wide_3x5_nullspace_subspace` | External projector comparison for a rank-2/nullity-3 wide fixture. | Not economy, sparse-mode, minimum-norm, or raw basis evidence. |

## Wide Economy Candidate Table

| Candidate | Potential value | Required metric | Day 6 disposition | Rationale |
| --- | --- | --- | --- | --- |
| Wide economy shape plus Q orthogonality on existing 3 x 6 fixture | Low to moderate. Would strengthen that economy mode keeps full row-space Q semantics for wide matrices. | `m x m` Q shape, R shape, rank, `Q^T Q`, and optional reconstruction/product metric. | Candidate for Day 7 only if it adds explicit shape and product diagnostics beyond `test_economy_wide`. | Existing wide economy smoke only checks rank; a bounded shape/orthogonality lane could be useful. |
| Wide economy vs full QR solve equivalence | Moderate if restricted to solve output equivalence. | Dense full/economy solution and residual comparison. | Deferred by default | It risks being read as underdetermined solution-selection or minimum-norm behavior unless scoped very tightly. |
| Wide economy nullspace/subspace projection | Low for Day 7. | Projector or two-way projection residual. | Deferred | Sprint 128 already added wide subspace projector evidence; economy interaction needs a distinct output-shape claim first. |
| Wide raw Q values | Low. | Raw equality with sign/order rules. | Rejected | Raw basis equality is sign and orientation sensitive and adds little beyond orthogonality/product metrics. |
| Wide residual-only solve evidence | Outside Day 7 scope. | Residual target only. | Deferred to end-of-epic queue | This is Sprint 128 residual debt and must not be pulled into economy evidence. |
| Wide minimum-norm evidence | Outside Day 7 scope. | Residual, norm, exact values or pseudoinverse cross-check. | Rejected for Day 7 | Belongs to minimum-norm owner, not economy output semantics. |

## Sparse-Mode Q/Economy Candidate Table

| Candidate | Potential value | Required metric | Day 6 disposition | Rationale |
| --- | --- | --- | --- | --- |
| Sparse-mode plus economy flag on tall full-rank fixture | Moderate. Current sparse-mode tests use `economy = 0`; current economy tests use dense mode. | Dense economy vs sparse economy rank, R shape, Q orthogonality/projector, solution/residual equivalence. | Candidate for Day 7 only if shape and product metrics are pinned. | This is the clearest non-duplicate intersection of sparse-mode and economy behavior. |
| Sparse-mode plus economy flag on wide fixture | Moderate later. | `m x m` Q shape, rank, R shape, dense/sparse product metric, and non-minimum-norm wording. | Secondary Day 7 candidate | Useful only after the primary sparse+economy semantics are pinned. |
| Sparse-mode raw Q comparison | Low. | Raw equality with sign/order rules. | Rejected | Backend mode must compare product metrics, not raw basis orientation. |
| Sparse-mode rank-deficient economy/nullspace | Low for Day 7. | Projector or projection metric plus sparse/dense mode comparison. | Deferred | Would mix rank-deficient subspace, sparse-mode, and economy ownership; current projector lanes already cover nullspace behavior. |
| Sparse-mode SuiteSparse Q/economy | Potentially useful later. | Corpus support tier, skip/runtime policy, Q/R shape, product metric, diagnostics. | Deferred to Days 8-9 | SuiteSparse Q/economy requires corpus metadata and support-tier gates. |
| Sparse-mode performance/fill evidence | Outside Day 7 scope. | Runtime/fill metrics. | Rejected | Sprint 129 is behavior/evidence focused, not performance parity. |

## Output-Shape Policy

| Surface | Required shape interpretation |
| --- | --- |
| Tall full-rank economy | Formed Q is `m x n`; R shape is the product's economy R shape; rank is `n` for full-rank fixtures. |
| Tall rank-deficient economy | Formed Q shape may remain tied to columns while rank is smaller; evidence must avoid unique-basis and minimum-norm claims. |
| Square economy | Economy flag must not imply a different Q shape than full square QR. |
| Wide economy | Formed Q is `m x m`; economy must not imply thin-Q columns based on `n`; rank must be interpreted separately from Q storage shape. |
| Sparse-mode Q/economy | Sparse-mode may be compared to dense-mode only through shape, rank, solve/residual, reconstruction, orthogonality, or projector/product metrics. |
| SuiteSparse Q/economy | Deferred until Day 8-9 support-tier, skip, runtime, corpus, and diagnostics policy is complete. |

## Metric Policy

Day 7 evidence must use product metrics and shape checks, not raw basis
equality.

| Metric | Accepted use | Tolerance guidance | Diagnostics |
| --- | --- | --- | --- |
| Shape and rank | Required for all accepted wide economy or sparse+economy candidates. | Exact integer checks. | Print or assert expected rows, columns, R shape, rank, and mode flags. |
| Q orthogonality | Accepted for formed Q or thin Q. | Small deterministic fixtures should use `< 1e-10`. | Report maximum `Q^T Q - I` error. |
| Projector distance | Accepted when proving column-space span for economy Q. | Small external/projector fixtures should use `< 1e-8`. | Report maximum projector difference. |
| Reconstruction/product residual | Accepted for mode comparison and wide Q/economy product behavior. | Fixture-local; small deterministic fixtures should use `1e-10` class tolerances. | Report dense/economy/sparse maxima separately. |
| Solve and residual equivalence | Accepted only as solve-adjacent mode behavior. | Match existing QR solve tolerances unless a fixture pins looser bounds. | Report full/economy/sparse residuals and max solution difference. |
| Raw Q equality | Disallowed by default. | Not applicable. | Do not use for Day 7. |

## Day 7 Acceptance Checklist

Day 7 may implement one wide economy or sparse-mode Q/economy lane only if all
of the following are true before code edits:

1. The fixture and test name are non-duplicate relative to the current evidence
   fence.
2. The matrix shape, rank expectation, economy flag, sparse-mode flag, and R
   shape expectation are explicit.
3. The formed Q shape is explicit: `m x n`, `m x rank`, `m x m`, or other
   product-specific shape.
4. The primary metric is shape, orthogonality, reconstruction/product,
   projector distance, or solve/residual equivalence.
5. The metric has fixture-local tolerances and diagnostics.
6. The artifact states why the evidence cannot be misread as residual-only
   solve, minimum-norm, raw basis, unique basis, SuiteSparse corpus, backend,
   platform, performance, or broad sparse QR parity.
7. Focused QR/economy validation is planned, and the full C quality gate is
   required if `.c` or `.h` files change.

If any checklist item is missing, Day 7 should explicitly defer the candidate.

## Recommended Day 7 Order

1. Evaluate a tall sparse-mode plus economy fixture first, because it is the
   clearest non-duplicate intersection of economy and sparse-mode behavior.
2. Evaluate a wide economy shape/orthogonality fixture only if it proves
   stronger output-shape semantics than the existing `test_economy_wide`.
3. Explicitly defer wide residual-only, wide minimum-norm, rank-deficient
   sparse/economy nullspace, raw Q, SuiteSparse, and performance lanes unless
   their owner-specific gates are already satisfied.

## No-Reopen Boundary

Day 6 does not reopen Sprint 128 residual QR debt. The following remain
end-of-epic queue items:

- compatible zero-residual QR residual evidence;
- wide residual-only QR evidence;
- near-threshold nullspace/subspace evidence;
- SuiteSparse rank-deficient QR corpus evidence;
- additional SuiteSparse or optional-large minimum-norm evidence;
- additional exact underdetermined minimum-norm evidence;
- additional QR-vs-SVD minimum-norm evidence.

Wide economy and sparse-mode evidence may mention these only as explicit
non-claims or deferrals.

## Validation

Day 6 changes documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_129
```

No `.c`, `.h`, Python helper, Matrix Market, build, maintainer guide, public
API, or public wording files changed on Day 6, so no additional code quality
gate is required for the Day 6 artifact itself.

## Non-Claims Preserved

- No new wide economy or sparse-mode implementation is accepted on Day 6.
- No raw Q-basis, Q-sign, Q-orientation, raw nullspace basis, column ordering,
  unique basis, or basis parity claim.
- No residual-only solve, compatible solve, wide solve, minimum-norm,
  pseudoinverse, SVD-oracle, SuiteSparse corpus, optional-data, platform,
  backend, performance, or broad sparse QR parity claim.
- No global QR rank-threshold, default-threshold, or numerical-rank policy.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity claim.
- No public API, package, ABI, CMake, Makefile, CI, CTest, helper API,
  scalability, memory, or state-of-the-art claim.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Wide economy and sparse-mode semantics are pinned before implementation. | Complete | Output-shape policy, metric policy, and Day 7 checklist define shape, rank, Q/R, metric, tolerance, and diagnostics. |
| No candidate can imply minimum-norm or residual-only behavior. | Complete | Candidate tables reject/defer wide residual-only and minimum-norm lanes and require non-claim wording for accepted evidence. |
| Accepted metrics are compatible with existing QR/economy APIs. | Complete | Metrics use existing shape/rank, `sparse_qr_form_q`, solve/residual, reconstruction, orthogonality, and projector-style checks. |
