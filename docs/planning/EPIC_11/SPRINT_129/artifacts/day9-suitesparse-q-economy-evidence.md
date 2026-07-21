# Sprint 129 Day 9 SuiteSparse Q/Economy Evidence

## Purpose

Day 9 applies the Day 8 SuiteSparse Q/economy gate and accepts one bounded
checked-in corpus lane:
`test_suitesparse_nos4_sparse_mode_economy_q_orthogonality`.

The lane extends Day 7's sparse-mode plus economy shape/orthogonality evidence
from a synthetic tall fixture to a checked-in SuiteSparse square fixture. It
does not use raw Q values, product-observed rank as an independent oracle, or
solve residuals as expected external truth.

## Accepted Evidence

| Field | Value |
| --- | --- |
| Test | `test_suitesparse_nos4_sparse_mode_economy_q_orthogonality` |
| Matrix | `tests/data/suitesparse/nos4.mtx` |
| Support tier | Checked-in small smoke/control |
| Matrix metadata | 100 x 100, 347 nnz |
| Dense control options | `reorder = SPARSE_REORDER_NONE`, `economy = 1` |
| Accepted options | `reorder = SPARSE_REORDER_NONE`, `economy = 1`, `sparse_mode = 1` |
| Expected Q shape | 100 x 100, pinned from checked-in matrix metadata and square-economy semantics |
| Expected R shape | 100 x 100, pinned from checked-in matrix metadata and square-economy semantics |
| Primary metric | Sparse-mode economy formed-Q orthogonality, `Q^T Q ~= I` |
| Control metric | Dense-economy and sparse-mode-economy solve/residual agreement |
| Tolerances | `Q^T Q` max error `< 1e-10`, max solution difference `< 1e-8`, residual difference `<= 1e-8` |
| Diagnostics | Prints matrix shape, dense and sparse ranks as controls, Q orthogonality error, max solution difference, and residual difference |

## Why This Is Non-Duplicate

Existing `nos4` QR lanes already cover solve residual, economy/full solve
agreement, sparse-mode solve agreement, reconstruction, fill diagnostics, and
refinement. Day 9 adds a distinct Q/economy metric: formed-Q orthogonality
with both `economy = 1` and `sparse_mode = 1` enabled on the checked-in
`nos4` fixture.

The lane also differs from Day 7 because Day 7 uses a synthetic tall 24 x 6
fixture with thin-Q semantics. Day 9 uses a checked-in square SuiteSparse
fixture where economy mode should preserve full 100 x 100 Q/R shape.

## Day 8 Gate Results

| Gate item | Result |
| --- | --- |
| Matrix path, dimensions, nnz, and support tier recorded | Passed: checked-in `nos4.mtx`, 100 x 100, 347 nnz, small smoke/control tier. |
| Missing-data behavior explicit | Passed: checked-in fixture load failure remains a test failure. |
| Runtime bounded | Passed in focused QR suite; full quality gate required because C changed. |
| Options explicit | Passed: dense economy control and sparse-mode economy options are recorded. |
| Q/R shape independent of product output | Passed: expected 100 x 100 shapes come from fixture metadata and square-economy semantics. |
| Basis-invariant metric | Passed: primary assertion is `Q^T Q` orthogonality. |
| Product-observed comparisons labeled controls | Passed: rank, solve, and residual comparisons are controls only. |
| Diagnostics present | Passed: diagnostic prints shape, ranks, Q metric, solve diff, and residual diff. |
| Non-claims fenced | Passed: raw basis, rank-deficient corpus, minimum-norm, backend, platform, performance, and broad SuiteSparse parity remain non-claims. |

## Candidate Disposition

| Candidate | Day 9 decision | Rationale |
| --- | --- | --- |
| `nos4` sparse-mode economy formed-Q shape and orthogonality | Accepted and implemented | Adds non-duplicate checked-in corpus Q/economy evidence with small runtime and pinned shape/metric/tolerance. |
| `nos4` square economy formed-Q shape without sparse mode | Deferred | Useful but less distinct than the crossed sparse-mode plus economy lane after Day 7. |
| `west0067` sparse-mode economy shape | Deferred | Existing sparse-mode solve lane covers mode agreement; needs a separate Q/economy metric before promotion. |
| `bcsstk04` economy or sparse-mode Q/economy | Deferred | Moderate fixture should get focused runtime/tolerance proof before becoming required Q/economy coverage. |
| Large checked-in SuiteSparse Q/economy lanes | Deferred/report-only | Need explicit runtime, memory, and failure-interpretation gates. |
| Raw SuiteSparse Q-column values | Rejected | No independent sign, orientation, or ordering oracle exists. |
| SuiteSparse rank-deficient QR corpus evidence | Deferred to end-of-epic queue | Needs independent rank/nullity metadata and separate nullspace/residual owner. |
| SuiteSparse minimum-norm evidence | Deferred to minimum-norm owner | Belongs to Days 10-11 or end-of-epic minimum-norm queue. |

## Files Changed

| File | Change |
| --- | --- |
| `tests/test_qr.c` | Added and registered `test_suitesparse_nos4_sparse_mode_economy_q_orthogonality`. |
| `docs/planning/EPIC_11/SPRINT_129/WORKING_NOTES.md` | Added Day 9 implementation notes. |
| `docs/planning/EPIC_11/SPRINT_129/artifacts/day9-suitesparse-q-economy-evidence.md` | Recorded evidence, validation, and non-claims. |

No Python helper, Matrix Market data, build file, public API, or public
wording file changed.

## Maintainer Guide Decision

No maintainer-guide update is required on Day 9. The accepted lane strengthens
the internal QR evidence corpus but does not add a public support tier, helper
protocol, external fixture key, or user-facing SuiteSparse parity claim.
Sprint closeout can decide whether to roll up Sprint 129 evidence into the
maintainer guide as a single bounded summary.

## Validation

Focused validation completed:

```text
make build/test_qr && ./build/test_qr
```

The focused QR suite passed with:

```text
nos4 sparse-mode economy Q: shape=100x100, rank_dense=100, rank_sparse=100,
Q_ortho=1.887e-15, max_sol_diff=0.000e+00, res_diff=0.000e+00
```

Because Day 9 changes a C test file, the required full gate is:

```text
make format && make lint && make test
```

The full C quality gate passed after the Day 9 test edit.

Documentation hygiene:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_129
```

## Non-Claims Preserved

- No raw SuiteSparse Q-basis, Q-sign, Q-orientation, column ordering,
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
| SuiteSparse Q/economy work is either bounded and validated or explicitly deferred. | Complete | `make build/test_qr && ./build/test_qr` and `make format && make lint && make test` passed after the C test edit. |
| No broad SuiteSparse corpus or platform claim is introduced. | Complete | Non-claims fence corpus, platform, backend, performance, and broad SuiteSparse parity. |
| Every deferred item has blocker and future-owner notes. | Complete | Candidate table records blocker and owner for remaining SuiteSparse Q/economy, raw basis, rank-deficient corpus, and minimum-norm lanes. |
