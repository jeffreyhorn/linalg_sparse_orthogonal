# Sprint 129 Day 5 Rank-Deficient Q/Nullspace Evidence

## Purpose

Day 5 applies the Day 4 rank-deficient Q/nullspace gate. One bounded
Q-application evidence lane is accepted and implemented:
`test_qr_dependent_row_q_transpose_column_space_rhs`.

The new test is intentionally not another nullspace projector fixture. It
checks that product QR applies `Q^T` coherently on a rank-deficient
dependent-row matrix when the right-hand side is known to be in the matrix
column space.

## Accepted Evidence

| Field | Value |
| --- | --- |
| Test | `test_qr_dependent_row_q_transpose_column_space_rhs` |
| Fixture source | `tf_qr_make_dependent_row_4x3()` |
| Matrix shape | 4 x 3 |
| Expected rank | 2 |
| Expected nullity | 1, already covered by existing projector evidence |
| Rank threshold | `0.0` product QR rank check |
| RHS | `b = 2*A(:,0) - A(:,1) = [2, -1, 1, 5]^T` |
| Primary metric | `Q^T b` residual-tail norm over entries `rank:m` |
| Secondary metric | `Q * (Q^T b)` round-trip maximum absolute error |
| Tolerances | `tail_norm < 1e-10`, `roundtrip_err < 1e-10` |
| Diagnostics | Prints rank, residual-tail norm, and round-trip error |

## Why This Is Non-Duplicate

Existing dependent-row evidence already checks rank, reconstruction,
nullspace extraction, null residual, and an external nullspace projector. Day
5 does not repeat those claims.

The accepted lane is Q-specific: it checks the solve-adjacent interpretation
that a column-space RHS has negligible `Q^T b` tail after the product rank.
That behavior is not established by comparing nullspace projectors alone.

## Candidate Disposition

| Candidate | Day 5 decision | Rationale |
| --- | --- | --- |
| Dependent-row Q-application projection | Accepted and implemented | The RHS, rank, threshold, metric, tolerances, and diagnostics are pinned, and the claim is Q-specific rather than nullspace-projector-specific. |
| Additional duplicate-column projector | Deferred | Existing duplicate-column projector already covers nullity-1 projector behavior. |
| Additional rank-1/nullity-2 projector | Deferred | Existing rank-1 projector already covers multi-dimensional nullity and local orthonormalization behavior. |
| Raw rank-deficient Q/nullspace basis equality | Rejected | Basis sign, ordering, and rotations are not stable evidence for deficient subspaces. |
| Wide rank-deficient economy/nullspace interaction | Deferred to Days 6-7 | Requires economy/wide output semantics before promotion. |
| Sparse-mode rank-deficient Q/nullspace interaction | Deferred to Days 6-7 | Requires mode-specific product metrics and backend non-claims. |
| Near-threshold nullspace/subspace | Deferred to end-of-epic queue | Requires threshold-specific rank/nullity and projection metadata. |
| SuiteSparse rank-deficient Q/nullspace evidence | Deferred to Days 8-9 or end-of-epic corpus owner | Requires independent rank/nullity metadata, support tier, skip behavior, runtime budget, and diagnostics. |
| Rank-deficient minimum-norm behavior | Rejected for Day 5 | Owned by minimum-norm tests and must not be blended with Q/nullspace evidence. |

## Files Changed

| File | Change |
| --- | --- |
| `tests/test_qr.c` | Added `test_qr_dependent_row_q_transpose_column_space_rhs` and registered it in the QR suite. |
| `docs/planning/EPIC_11/SPRINT_129/WORKING_NOTES.md` | Added Day 5 implementation notes. |
| `docs/planning/EPIC_11/SPRINT_129/artifacts/day5-rankdef-q-nullspace-evidence.md` | Recorded this evidence package, validation, and non-claims. |

No Python helper, Matrix Market data, build file, maintainer guide, public API,
or public wording file changed.

## Maintainer Guide Decision

No maintainer-guide update is required on Day 5. The accepted lane strengthens
the existing QR Q-application/rank-deficient evidence owner in
`tests/test_qr.c`, but it does not introduce a new external fixture key, public
behavior row, helper protocol, or user-visible support claim.

## Validation

Because Day 5 changes a C test file, the required quality gate is:

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

Both commands passed.

## Non-Claims Preserved

- No raw Q-basis, raw nullspace basis, Q-sign, Q-orientation, column ordering,
  or unique-basis parity claim.
- No new nullspace projector, broad nullspace, broad subspace, or
  rank-deficient QR parity claim.
- No residual-only solve, compatible solve, wide solve, minimum-norm,
  SVD-pseudoinverse, economy, sparse-mode, SuiteSparse, corpus, optional-data,
  platform, performance, or backend parity claim.
- No global QR rank-threshold, default-threshold, or numerical-rank policy.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity claim.
- No public API, package, ABI, CMake, Makefile, CI, CTest, helper API,
  scalability, memory, or state-of-the-art claim.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Accepted evidence is behavior-specific and non-duplicative, or the lane is explicitly deferred. | Complete | One Q-application residual-tail lane is accepted; overlapping projector, wide, sparse-mode, near-threshold, SuiteSparse, and minimum-norm candidates are deferred or rejected. |
| Validation is complete for every touched code or helper file. | Complete | `make build/test_qr && ./build/test_qr` and `make format && make lint && make test` passed after the C test edit. |
| No broad rank-deficient QR, nullspace, or Q-basis parity claim is added. | Complete | Non-claims fence raw basis, broad rank-deficient QR, nullspace/subspace parity, threshold policy, and external ecosystem parity. |
