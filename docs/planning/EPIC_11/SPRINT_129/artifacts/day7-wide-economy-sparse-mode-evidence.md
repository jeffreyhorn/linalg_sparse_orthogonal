# Sprint 129 Day 7 Wide Economy And Sparse-Mode Evidence

## Purpose

Day 7 applies the Day 6 wide economy and sparse-mode Q/economy policy. One
bounded evidence lane is accepted and implemented:
`test_sparse_mode_economy_tall_q_shape`.

The new lane covers the intersection that was not already tested: sparse-mode
QR with the economy flag enabled on a tall full-rank fixture. It checks shape,
rank, thin-Q orthogonality, and dense-economy versus sparse-economy solve
equivalence without making raw-basis, residual-only, minimum-norm, backend, or
performance claims.

## Accepted Evidence

| Field | Value |
| --- | --- |
| Test | `test_sparse_mode_economy_tall_q_shape` |
| Fixture source | `tf_qr_make_tall_diagonal_dominant(24, 6, 8.0, 0.25, 1)` |
| Matrix shape | 24 x 6 |
| Expected rank | 6 |
| Dense-mode options | `reorder = SPARSE_REORDER_NONE`, `economy = 1` |
| Sparse-mode options | `reorder = SPARSE_REORDER_NONE`, `economy = 1`, `sparse_mode = 1` |
| R shape | 6 x 6 for both dense economy and sparse economy QR |
| Formed-Q shape | 24 x 6 thin Q for sparse-mode economy QR |
| Primary metric | Sparse-mode economy thin-Q orthogonality |
| Secondary metric | Dense economy and sparse economy solution/residual equivalence |
| Tolerances | `Q^T Q` max error `< 1e-10`, max solution difference `< 1e-10`, residual difference `<= 1e-10` |
| Diagnostics | Prints product rank, sparse-mode economy Q orthogonality error, max solution difference, and residual difference |

## Why This Is Non-Duplicate

Existing economy tests cover dense-mode economy behavior. Existing sparse-mode
tests cover sparse-mode behavior with `economy = 0`. Day 7 adds only the
crossed mode: `sparse_mode = 1` with `economy = 1`.

The test does not duplicate `qr_economy_projector_5x3`, because it does not
add an external projector or raw Q comparison. It also does not duplicate
`test_sparse_mode_tall`, because that lane checks sparse-mode full QR solve
agreement rather than sparse-mode thin-Q shape and economy R shape.

## Candidate Disposition

| Candidate | Day 7 decision | Rationale |
| --- | --- | --- |
| Tall sparse-mode plus economy fixture | Accepted and implemented | Non-duplicate intersection of sparse-mode and economy behavior with pinned shape, rank, metric, tolerances, and diagnostics. |
| Wide economy shape plus Q orthogonality | Deferred | Useful later, but Day 7 already accepted the cleaner sparse+economy intersection and should not add another lane. |
| Sparse-mode plus economy on wide fixture | Deferred | Requires more careful `m x m` wide semantics and non-minimum-norm wording. |
| Wide economy nullspace/subspace projection | Deferred | Sprint 128 already owns wide subspace projector evidence; economy interaction needs a distinct future shape claim. |
| Wide residual-only evidence | Deferred to end-of-epic queue | Residual-only behavior is outside Q/economy output semantics. |
| Wide or sparse-mode minimum-norm evidence | Rejected for Day 7 | Minimum-norm behavior belongs to the minimum-norm owner. |
| Raw Q or raw nullspace basis equality | Rejected | Basis sign, ordering, and orientation are not stable sparse/economy evidence. |
| SuiteSparse sparse-mode Q/economy | Deferred to Days 8-9 | Requires support-tier, skip, runtime, corpus, and diagnostics gates. |
| Sparse-mode performance or fill evidence | Rejected | Day 7 is behavior evidence, not performance or backend parity. |

## Files Changed

| File | Change |
| --- | --- |
| `tests/test_qr.c` | Added `test_sparse_mode_economy_tall_q_shape` and registered it in the QR suite. |
| `docs/planning/EPIC_11/SPRINT_129/WORKING_NOTES.md` | Added Day 7 implementation notes. |
| `docs/planning/EPIC_11/SPRINT_129/artifacts/day7-wide-economy-sparse-mode-evidence.md` | Recorded evidence, validation, and non-claims. |

No Python helper, Matrix Market data, build file, maintainer guide, public API,
or public wording file changed.

## Maintainer Guide Decision

No maintainer-guide update is required on Day 7. The accepted lane strengthens
the existing QR economy/sparse-mode evidence in `tests/test_qr.c`, but it does
not introduce a new external fixture key, helper protocol, public API behavior,
or user-visible support tier.

## Validation

Because Day 7 changes a C test file, the required quality gate is:

```text
make format && make lint && make test
```

Focused validation should also include:

```text
make build/test_qr && ./build/test_qr
```

Documentation hygiene:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_129
```

Completed validation:

```text
make build/test_qr && ./build/test_qr
make format && make lint && make test
```

Both commands passed after the Day 7 C test edit.

## Non-Claims Preserved

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
| Accepted evidence has stable shape, metric, tolerance, and diagnostics. | Complete | The test pins 24 x 6 shape, rank 6, 6 x 6 R shape, 24 x 6 thin Q, orthogonality/solution/residual metrics, tolerances, and diagnostic output. |
| Sparse-mode evidence does not imply broad sparse QR parity. | Complete | Non-claims fence backend, platform, performance, SuiteSparse, and broad sparse QR parity. |
| Touched files have focused validation and full quality gate when required. | Complete | `make build/test_qr && ./build/test_qr` and `make format && make lint && make test` passed after the C test edit. |
