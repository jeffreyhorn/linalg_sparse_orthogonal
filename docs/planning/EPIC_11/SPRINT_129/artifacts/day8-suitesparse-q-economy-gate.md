# Sprint 129 Day 8 SuiteSparse Q/Economy Gate

## Purpose

Day 8 decides whether SuiteSparse-backed Q/economy evidence has enough
metadata to proceed on Day 9. The gate is intentionally conservative: checked
in Matrix Market files can be good corpus controls, but product-observed Q,
R, rank, residual, solve, fill, or timing values are not independent oracle
values.

Day 8 does not implement a new test. It defines what Day 9 may accept,
defer, or reject.

## Checked-In Corpus Inventory

All current SuiteSparse fixtures are checked in under
`tests/data/suitesparse`, so missing-data behavior for these paths is a test
failure rather than an optional-data skip.

| Matrix | Shape | nnz | Current QR-adjacent coverage | Day 8 support tier |
| --- | --- | ---: | --- | --- |
| `west0067.mtx` | 67 x 67 | 294 | QR solve, sparse-mode dense/sparse agreement, COLAMD/LU, minimum-norm submatrix | Checked-in smoke/control |
| `nos4.mtx` | 100 x 100 | 347 | QR solve, QR reorder fill, economy/full solve equivalence, sparse-mode dense/sparse agreement, reconstruction, timing/fill, refinement | Checked-in smoke/control |
| `bcsstk04.mtx` | 132 x 132 | 1890 | QR solve/reconstruction, sparse-mode dense/sparse agreement, other direct/eigs coverage | Checked-in smoke/control |
| `steam1.mtx` | 240 x 240 | 3762 | COLAMD/LU and iterative controls, no direct QR Q/economy owner | Checked-in non-default candidate |
| `fs_541_1.mtx` | 541 x 541 | 4285 | COLAMD/LU control, no direct QR Q/economy owner | Checked-in non-default candidate |
| `orsirr_1.mtx` | 1030 x 1030 | 6858 | COLAMD/LU and iterative controls, no direct QR Q/economy owner | Checked-in non-default candidate |
| `bcsstk14.mtx` | 1806 x 1806 | 32630 | direct/eigs/reorder controls, no direct QR Q/economy owner | Checked-in large smoke candidate |
| `s3rmt3m3.mtx` | 5357 x 5357 | 106526 | reorder/postorder corpus control, no direct QR Q/economy owner | Checked-in report-only for Q/economy |
| `Kuu.mtx` | 7102 x 7102 | 173651 | reorder corpus control, no direct QR Q/economy owner | Checked-in report-only for Q/economy |
| `bloweybq.mtx` | 10001 x 10001 | 39996 | no current QR Q/economy owner | Checked-in report-only for Q/economy |
| `Pres_Poisson.mtx` | 14822 x 14822 | 365313 | reorder corpus control, no direct QR Q/economy owner | Checked-in report-only for Q/economy |
| `tuma1.mtx` | 22967 x 22967 | 50560 | no current QR Q/economy owner | Checked-in report-only for Q/economy |

Day 9 should prefer `nos4`, `west0067`, or `bcsstk04` if it implements a
bounded SuiteSparse Q/economy lane. They already have QR-adjacent ownership,
small runtime, and established diagnostics. Larger matrices may be used only
as report-only controls unless a separate runtime and support-tier decision is
recorded first.

## Existing Coverage Fence

| Existing lane | File | What it proves | Day 8 fence |
| --- | --- | --- | --- |
| `test_economy_nos4` | `tests/test_qr.c` | Full and economy QR solve outputs agree on checked-in `nos4`. | Solve-equivalence control, not Q-basis or projector evidence. |
| `test_sparse_mode_nos4` | `tests/test_qr.c` | Dense-mode and sparse-mode solutions agree on checked-in `nos4`. | Sparse-mode solve control, not economy evidence. |
| `test_sparse_mode_west0067` | `tests/test_qr.c` | Dense-mode and sparse-mode solutions agree on checked-in `west0067`. | Product-observed mode control only. |
| `test_sparse_mode_bcsstk04` | `tests/test_qr.c` | Dense-mode and sparse-mode solutions agree on checked-in `bcsstk04`. | Product-observed mode control only. |
| `test_sparse_mode_reconstruction` | `tests/test_qr.c` | Dense and sparse QR reconstruction errors are small on checked-in `nos4`. | Reconstruction control, not independent Q/economy oracle. |
| `test_sparse_mode_timing` | `tests/test_qr.c` | Prints `nos4` R nnz in dense and sparse modes. | Diagnostic/fill smoke only; no performance or backend parity claim. |
| `test_qr_bcsstk04` | `tests/test_qr_solve.c` | QR rank, reconstruction, and solve residual on checked-in `bcsstk04`. | QR solve owner, not Q/economy owner. |
| `test_qr_west0067` | `tests/test_qr_solve.c` | QR solve residual on checked-in `west0067`. | QR solve owner, not Q/economy owner. |

Day 9 must not duplicate these lanes as new evidence unless it adds a distinct
Q/economy metric, such as formed-Q shape plus orthogonality or an external
projector that is not derived from the implementation under test.

## Candidate Disposition

| Candidate | Metadata status | Day 8 decision | Promotion requirement |
| --- | --- | --- | --- |
| `nos4` square economy formed-Q shape and orthogonality | Checked-in, small, existing economy owner, known square shape | Tentatively promotable | Pin Q shape 100 x 100, R shape 100 x 100, metric `Q^T Q`, tolerance, diagnostics, focused runtime, and non-oracle wording. |
| `nos4` sparse-mode economy formed-Q shape and orthogonality | Checked-in, small, extends Day 7 crossed-mode evidence to corpus | Tentatively promotable | Pin dense/sparse/economy options, Q/R shapes, orthogonality metric, solve/residual comparison only as control, and no broad sparse QR parity. |
| `west0067` sparse-mode economy shape | Checked-in, small, nonsymmetric QR-adjacent control | Deferred | Existing sparse-mode solve lane already covers mode agreement; Day 9 needs an independent Q/economy metric before promotion. |
| `bcsstk04` economy or sparse-mode Q/economy | Checked-in, moderate, existing solve/reconstruction owner | Deferred | Runtime and tolerance should be proved with a focused diagnostic before adding to required CI. |
| Large checked-in matrices (`bcsstk14`, `s3rmt3m3`, `Kuu`, `Pres_Poisson`, `bloweybq`, `tuma1`) | Checked-in but larger runtime/memory footprint | Deferred/report-only | Needs explicit runtime budget, memory posture, skip/report tier, and failure interpretation before required tests. |
| Raw SuiteSparse Q-column values | No independent sign/orientation oracle | Rejected | Would need external basis metadata and sign/order policy that does not come from product output. |
| SuiteSparse rank-deficient QR corpus evidence | No rank-deficient corpus metadata in this gate | Deferred to end-of-epic queue | Needs independent rank/nullity metadata and separate residual/nullspace owner. |
| SuiteSparse minimum-norm evidence | Owner is COLAMD/minimum-norm, not Q/economy | Deferred to Days 10-11 or end-of-epic queue | Needs minimum-norm helper/owner gate and independent expected behavior. |

## Day 9 Acceptance Checklist

Day 9 may implement at most one SuiteSparse Q/economy lane, and only if every
item below is true before editing code:

1. Matrix path, dimensions, nnz, and checked-in support tier are recorded.
2. Missing-data behavior is explicit. For checked-in paths, missing data is a
   failure; optional-data paths would need `SKIP_TEST` with diagnostic text.
3. Runtime expectation is bounded to the QR focused suite and full quality
   gate if C code changes.
4. Full/economy/sparse-mode options are explicit.
5. Expected Q shape and R shape are pinned from matrix dimensions and mode
   semantics, not from product-observed output.
6. Metric and tolerance are basis-invariant, preferably Q orthogonality,
   reconstruction, projection, or projector distance.
7. Any dense-versus-sparse, full-versus-economy, solve, residual, fill, or
   timing comparison is labeled a control, not an independent oracle.
8. Diagnostics print matrix name, shape, rank, selected mode flags, Q/R shape
   interpretation, metric value, and tolerance.
9. The evidence explicitly fences raw basis, sign, ordering, unique basis,
   rank-deficient corpus, minimum-norm, backend, platform, performance, and
   broad SuiteSparse parity claims.

If any item is missing, Day 9 must explicitly defer the lane.

## Support-Tier And Runtime Policy

- Checked-in small controls: `west0067`, `nos4`, and `bcsstk04` may be
  required tests after a focused runtime check.
- Checked-in non-default candidates: `steam1`, `fs_541_1`, and `orsirr_1`
  require a stronger runtime and failure-interpretation note before promotion.
- Checked-in large/report-only candidates: `bcsstk14`, `s3rmt3m3`, `Kuu`,
  `Pres_Poisson`, `bloweybq`, and `tuma1` should not become required
  Q/economy tests in Sprint 129 without a separate runtime budget.
- Optional or absent SuiteSparse data is not part of Day 8. Future optional
  lanes must use explicit skip diagnostics and must not be required for normal
  CI.

## Files Changed

| File | Change |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_129/WORKING_NOTES.md` | Added Day 8 gate notes. |
| `docs/planning/EPIC_11/SPRINT_129/artifacts/day8-suitesparse-q-economy-gate.md` | Recorded corpus inventory, candidate disposition, support-tier policy, and Day 9 acceptance checklist. |

No C source, header, Python helper, Matrix Market data, build file, maintainer
guide, public API, or public wording file changed.

## Maintainer Guide Decision

No maintainer-guide update is required on Day 8. The day defines an internal
promotion gate and does not add a new accepted evidence lane, public support
tier, helper protocol, external fixture key, or user-visible claim.

## Validation

Day 8 changes documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_129
```

No code quality gate is required for Day 8 because no `.c` or `.h` file
changed for this day.

## Non-Claims Preserved

- No raw SuiteSparse Q-basis, Q-sign, Q-orientation, column-ordering,
  unique-basis, rank-deficient corpus, nullspace, residual-only,
  minimum-norm, pseudoinverse, or SVD-oracle claim.
- No broad QR, Q/economy, sparse-mode, SuiteSparse corpus, optional-data,
  platform, backend, performance, fill, timing, scalability, or memory claim.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity claim.
- No public API, package, ABI, CMake, Makefile, CI, CTest, helper API, or
  state-of-the-art claim.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| SuiteSparse evidence cannot proceed without support-tier and diagnostic metadata. | Complete | Corpus inventory, support tiers, runtime policy, and Day 9 checklist require matrix metadata, shape, diagnostics, support tier, and runtime posture. |
| Product output is not treated as an independent oracle. | Complete | Candidate table and acceptance checklist label product-observed solve/residual/fill/timing values as controls, not independent expected values. |
| Runtime and optional-data expectations are explicit. | Complete | Support-tier policy distinguishes checked-in small controls, non-default candidates, report-only large matrices, and future optional-data skip behavior. |
