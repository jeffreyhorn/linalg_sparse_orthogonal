# Sprint 129 Day 3 Raw Q-Column Evidence Decision

## Purpose

Day 3 applies the Day 2 Q-basis/economy policy to raw QR Q-column evidence.

No raw Q-column fixture is accepted for implementation on Day 3. Current
candidate lanes either duplicate stronger basis-invariant evidence, require
implementation-specific sign/orientation rules that do not support a durable
claim, or belong to later Sprint 129 economy/sparse-mode and SuiteSparse gates.

This is a documentation-only decision artifact. No C source, header, Python
helper, Matrix Market, build, maintainer guide, public API, or public wording
files are changed on Day 3.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 129 Day 2 policy artifact | Defines the raw Q acceptance gate and preferred metric order. |
| Sprint 124 Day 6 Q-basis/economy semantics | Defines raw Q sign/orientation and projection/subspace policies. |
| Sprint 124 Day 7 Q-basis/economy decision | Accepted `qr_economy_projector_5x3`, the bounded economy projector baseline. |
| `tests/test_qr.c` | Current owner for Q formation, application, orthogonality, economy, sparse-mode, nullspace, and reconstruction behavior. |
| `tests/qr_external_dense_reference.py` | Current external helper for bounded QR references; already owns projector-style Q/economy output for `qr_economy_projector_5x3`. |
| Sprint 128 end-of-epic residual queue | Prevents raw Q decisions from reopening residual, threshold, SuiteSparse corpus, or minimum-norm debt. |

## Candidate Disposition

| Candidate | Existing evidence | Day 3 decision | Rationale |
| --- | --- | --- | --- |
| Full-rank tall raw Q-column fixture | `test_q_orthogonality_tall`, reconstruction checks, Q apply tests, and full Q formation coverage. | Deferred | A raw column test would mostly lock down Householder sign/orientation conventions. It does not add enough trust beyond orthogonality, reconstruction, and application metrics. |
| Economy raw Q-column fixture based on `qr_economy_projector_5x3` | `qr_economy_projector_5x3` already compares `Q Q^T` against an external projector and checks thin-Q shape and orthogonality. | Deferred | Raw values would duplicate the accepted economy projector lane while adding basis-orientation risk. |
| Rank-deficient raw Q-column fixture | Sprint 125-128 projector/subspace fixtures cover named rank-deficient nullspace/subspace behavior. | Rejected for raw equality | Rank-deficient bases can validly rotate; use projector, projection, or principal-angle metrics instead. |
| Wide raw Q-column fixture | Existing wide Q orthogonality and wide economy shape smoke coverage. | Deferred to Days 6-7 | Wide/economy semantics need shape and projection gates first; raw equality is not accepted. |
| Sparse-mode raw Q-column fixture | Existing sparse-mode dense-mode product comparisons and sparse-mode Q orthogonality. | Deferred to Days 6-7 | Sparse-mode should compare product metrics only and must not imply backend or performance parity. |
| SuiteSparse raw Q-column fixture | Existing SuiteSparse QR controls for rank/solve/sparse-mode behavior. | Rejected for Day 3 | Corpus-backed raw basis evidence lacks support-tier, skip, runtime, and independent expected-basis metadata. |

## Decision

Day 3 explicitly defers raw Q-column implementation.

The accepted Sprint 129 direction is:

1. Keep raw Q equality out of default Q/economy evidence.
2. Prefer shape, orthogonality, reconstruction, projection, projector distance,
   or principal-angle metrics.
3. Reserve raw Q-column values for a future fixture only if the fixture has
   stable sign normalization, column ordering, storage layout, permutation
   interpretation, tolerance, diagnostics, and a clear claim that is not better
   served by a basis-invariant metric.

## Future Promotion Gate

A future raw Q-column fixture may proceed only when all of the following are
available before code edits:

1. A non-duplicate fixture key and owner-local test name.
2. A full-rank, non-degenerate matrix with a documented reason raw Q values are
   meaningful.
3. Expected full or economy Q dimensions and storage layout.
4. Column ordering and permutation interpretation.
5. Sign normalization rule for each compared column.
6. Exact value tolerance and separate diagnostics for shape, sign, ordering,
   value, orthogonality, and reconstruction failures.
7. A statement explaining why raw Q equality adds trust beyond orthogonality,
   reconstruction, projection, or projector metrics.
8. Focused QR/helper validation and full quality validation if `.c` or `.h`
   files change.

If any requirement is missing, use a basis-invariant metric or defer.

## No-Reopen Boundary

Day 3 does not reopen Sprint 128 residual QR debt. The following remain
end-of-epic queue items:

- compatible zero-residual QR residual evidence;
- wide residual-only QR evidence;
- near-threshold nullspace/subspace evidence;
- SuiteSparse rank-deficient QR corpus evidence;
- additional SuiteSparse or optional-large minimum-norm evidence;
- additional exact underdetermined minimum-norm evidence;
- additional QR-vs-SVD minimum-norm evidence.

Raw Q-column evidence is a Q-basis topic, but none of the Day 3 candidates
requires pulling those residual items back into Sprint 129.

## Validation

Day 3 changes documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_129
```

No `.c`, `.h`, Python helper, Matrix Market, build, maintainer guide, public
API, or public wording files changed, so no code quality gate is required.

## Non-Claims Preserved

- No raw Q-column fixture is accepted on Day 3.
- No raw Q-basis equality, Q-sign, Q-orientation, column ordering, or
  unique-basis parity claim.
- No broad QR, Q-basis, economy, sparse-mode, nullspace, SuiteSparse, corpus,
  optional-data, platform, performance, or backend parity claim.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity claim.
- No SVD-pseudoinverse oracle claim.
- No public API, package, ABI, CMake, Makefile, CI, CTest, helper API,
  scalability, memory, or state-of-the-art claim.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Raw Q evidence either passes with fixture-local orientation rules or is explicitly deferred. | Complete | Every Day 3 candidate is deferred or rejected for raw equality. |
| No raw Q result is described as unique-basis or broad Q parity evidence. | Complete | No raw Q implementation was added; non-claims preserve basis and parity boundaries. |
| Touched code/scripts have appropriate focused validation. | Complete | No code or scripts changed on Day 3; documentation hygiene is sufficient. |
