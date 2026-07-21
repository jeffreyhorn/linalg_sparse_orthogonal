# Sprint 129 Day 4 Rank-Deficient Q/Nullspace Policy Gate

## Purpose

Day 4 defines the Sprint 129 policy for rank-deficient Q/nullspace evidence.
The gate decides whether any Q-specific or nullspace/subspace candidate can add
trust beyond the projector and threshold work completed in Sprints 125-128.

The result is a conservative Day 5 gate: implementation may proceed only for a
bounded, non-duplicate candidate with explicit rank, nullity, metric,
tolerance, diagnostics, and claim boundaries. Otherwise Day 5 should record an
explicit deferral.

This is a documentation-only policy artifact. No C source, header, Python
helper, Matrix Market, build, maintainer guide, public API, or public wording
files are changed on Day 4.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 129 Day 2 Q-basis/economy policy | Establishes basis-invariant metrics as the default and rejects rank-deficient raw Q equality. |
| Sprint 129 Day 3 raw Q decision | Confirms raw Q-column evidence is deferred or rejected and rank-deficient evidence must use projector, projection, or principal-angle metrics. |
| Sprint 125-128 QR nullspace/subspace artifacts | Provide the completed duplicate-column, rank-1, dependent-row, wide, and threshold-family evidence lanes. |
| Sprint 128 residual queue | Keeps near-threshold, SuiteSparse rank-deficient corpus, optional-large, and extra minimum-norm work in the end-of-epic queue unless directly promoted. |
| `tests/test_qr.c` | Current owner for product QR rank, Q formation/application, nullspace extraction, external projector checks, wide subspace evidence, and threshold-family checks. |
| `tests/qr_external_dense_reference.py` | Current owner for small external QR projector and threshold reference values. |

## Completed Evidence Fence

Day 4 treats the following rank-deficient nullspace/subspace evidence as
complete for Sprint 129 planning purposes. New evidence must not repackage
these fixtures under a different name.

| Completed evidence | Current owner | Existing proof | Duplicate fence |
| --- | --- | --- | --- |
| `qr_rankdef_duplicate_5x4_nullspace_projector` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py` | Rank 3, nullity 1, product null residual, normalized basis, and 4 x 4 projector comparison. | Do not add another duplicate-column nullity-1 projector unless the new claim is not projector/subspace behavior. |
| `qr_rank1_4x3_nullspace_projector` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py` | Rank 1, nullity 2, locally orthonormalized basis, null residual, orthogonality, and projector comparison. | Do not repeat as generic multi-dimensional nullspace coverage. |
| `qr_rankdef_dependent_row_4x3_nullspace_projector` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py` | Rank 2, nullity 1, dependent-row construction, null residual, norm, and projector comparison. | Do not repeat as dependent-row, raw-basis, minimum-norm, sparse-mode, or economy evidence. |
| `qr_rankdef_wide_3x5_nullspace_subspace` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py` | Rank 2, nullity 3, wide-shape product null residual, orthogonality, and 5 x 5 projector comparison. | Do not repeat as wide nullspace/subspace evidence unless the new claim is strictly Q/economy/sparse-mode output semantics. |
| Threshold-family rank evidence | `tests/test_qr.c`, `tests/qr_external_dense_reference.py` | Fixture-local expected ranks under explicit thresholds and rank-info diagnostics. | Does not by itself prove threshold-specific nullspace/subspace; future subspace evidence must pin nullity and projection metrics per threshold. |

## Candidate Table

| Candidate | Possible proof value | Required metric | Day 4 disposition | Rationale |
| --- | --- | --- | --- | --- |
| Raw rank-deficient Q basis | Low. Could expose one product basis orientation. | Raw equality would be required, but is not accepted. | Rejected | Rank-deficient bases are not unique; raw equality would overfit sign, ordering, and rotations. |
| Additional duplicate-column nullity-1 projector | Low. Existing 5 x 4 duplicate-column projector already covers this lane. | Full projector | Deferred as duplicate | No distinct Q-specific claim is available for Day 5. |
| Additional rank-1/nullity-2 projector | Low. Existing rank-1 4 x 3 fixture already covers multi-dimensional nullity handling. | Full projector or two-way projection | Deferred as duplicate | Another small rank-1 projector would mostly repeat completed orthonormalization/projector behavior. |
| Dependent-row Q-application projection | Moderate only if it proves Q application on a rank-deficient row-space vector, not nullspace extraction. | Projection plus orthogonality/reconstruction diagnostics | Candidate for Day 5 only if vector, expected projection, tolerance, and non-nullspace claim are pinned. | This could be Q-specific, but must not become another nullspace projector. |
| Wide rank-deficient economy/nullspace interaction | Potentially useful later for wide/economy semantics. | Shape plus projector/projection metric | Deferred to Days 6-7 | Belongs to the wide economy and sparse-mode owners, not Day 5 rank-deficient nullspace. |
| Near-threshold nullspace/subspace | Potentially useful later after threshold ranks are fixed. | Threshold-specific projector or two-way projection | Deferred | Requires threshold-specific rank/nullity metadata and would reopen end-of-epic debt without a Day 5 need. |
| SuiteSparse rank-deficient nullspace/subspace | Potentially high later. | Two-way projection with corpus support metadata | Deferred to Days 8-9 or end-of-epic corpus owner | Missing independent rank/nullity metadata, support tier, skip behavior, runtime budget, and diagnostics. |
| Minimum-norm rank-deficient behavior | Owned by minimum-norm tests, not nullspace policy. | Residual, solution norm, optional exact values | Rejected for Day 5 | Would blur nullspace evidence with solution-selection behavior. |

## Metric And Tolerance Policy

Rank-deficient Q/nullspace evidence must use the least basis-dependent metric
that proves the intended behavior.

| Metric | Accepted use | Tolerance policy | Diagnostics |
| --- | --- | --- | --- |
| Rank and nullity | Required for every nullspace/subspace candidate. | Fixture-local expected rank and nullity must be exact integer checks. | Print or report expected rank, product rank, expected nullity, and product nullity. |
| Null residual `||A Z||` | Required when checking product nullspace basis vectors. | Small exact fixtures should use `<= 1e-10` unless the fixture pins a looser threshold. | Report maximum residual across basis vectors. |
| Orthonormality `||Z^T Z - I||` | Required before projector or two-way projection metrics. | Small exact fixtures should use `<= 1e-10`; larger/corpus fixtures need fixture-local tolerances. | Report maximum diagonal and off-diagonal error or combined max error. |
| Full projector distance | Accepted for tiny fixtures where the full projector is readable and non-noisy. | Small exact fixtures should use `<= 1e-8` unless justified otherwise. | Report maximum absolute projector difference. |
| Two-way projection residual | Preferred for wide, multi-dimensional, near-threshold, or corpus fixtures. | Candidate-specific; must separately bound product-to-reference and reference-to-product residuals. | Report both directions and the maximum. |
| Principal-angle bound | Allowed only if projector/projection metrics are insufficient. | Candidate-specific with basis construction and angle tolerance pinned before implementation. | Report maximum sine/cosine-derived subspace error. |
| Raw basis equality | Disallowed by default. | Not applicable unless a future deterministic-basis owner satisfies the raw Q promotion gate. | Must not be used for Day 5 rank-deficient nullspace evidence. |

## Required Metadata For Day 5

Day 5 may implement rank-deficient Q/nullspace evidence only if the candidate
has all of the following before code edits:

1. A fixture key and owner-local test name that do not duplicate completed
   projector evidence.
2. Matrix shape, nonzero pattern, and fixture construction source.
3. Expected rank, expected nullity, and explicit rank threshold.
4. The intended claim, stated as Q-specific, nullspace/subspace-specific, or
   explicit deferral.
5. Expected Q/nullspace output shape and storage layout if Q or basis arrays
   are inspected.
6. Metric choice: projection, full projector, two-way projection residual, or
   principal-angle bound.
7. Tolerances for rank metadata, null residual, orthogonality, projector or
   projection metric, and any Q application or reconstruction metric.
8. Failure diagnostics listing shape, rank, nullity, threshold, residual, and
   metric maxima.
9. Non-claims fencing residual-only solve, minimum-norm, raw basis equality,
   economy, sparse-mode, SuiteSparse corpus, platform, performance, and broad
   external-library parity.

If any item is missing, Day 5 should explicitly defer the candidate and record
which prerequisite is absent.

## Day 5 Candidate Order

Day 5 should evaluate candidates in this order:

1. Dependent-row Q-application projection, only if it can prove a Q-specific
   projection/application behavior without duplicating the existing
   dependent-row nullspace projector.
2. A new non-duplicate exact rank-deficient fixture, only if it introduces a
   distinct shape or Q/nullspace behavior not already covered by the completed
   evidence fence.
3. Explicit deferral of additional rank-deficient Q/nullspace evidence if the
   first two candidates cannot satisfy the metadata gate.

Day 5 should not promote near-threshold, SuiteSparse, wide economy,
sparse-mode, or minimum-norm candidates from the end-of-epic residual queue
unless they are directly required for a Sprint 129 Q/economy/helper claim and
satisfy their owner-specific gates first.

## No-Reopen Boundary

Day 4 does not reopen Sprint 128 residual QR debt. The following remain
end-of-epic queue items:

- compatible zero-residual QR residual evidence;
- wide residual-only QR evidence;
- near-threshold nullspace/subspace evidence;
- SuiteSparse rank-deficient QR corpus evidence;
- additional SuiteSparse or optional-large minimum-norm evidence;
- additional exact underdetermined minimum-norm evidence;
- additional QR-vs-SVD minimum-norm evidence.

The Day 5 gate may refer to these items only as explicit deferrals or duplicate
fences unless a candidate directly supports the Sprint 129 Q/economy/helper
surface.

## Validation

Day 4 changes documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_129
```

No `.c`, `.h`, Python helper, Matrix Market, build, maintainer guide, public
API, or public wording files changed, so no code quality gate is required.

## Non-Claims Preserved

- No new rank-deficient Q/nullspace implementation is accepted on Day 4.
- No raw Q-basis, raw nullspace basis, Q-sign, Q-orientation, column ordering,
  or unique-basis parity claim.
- No broad QR, rank-deficient solve, nullspace, subspace, Q-basis, economy,
  sparse-mode, minimum-norm, SuiteSparse, corpus, optional-data, platform,
  performance, or backend parity claim.
- No global QR rank-threshold, default-threshold, or numerical-rank policy.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity claim.
- No SVD-pseudoinverse oracle claim.
- No public API, package, ABI, CMake, Makefile, CI, CTest, helper API,
  scalability, memory, or state-of-the-art claim.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| No rank-deficient Q/nullspace candidate duplicates completed projector work. | Complete | Completed evidence fence and candidate table mark existing duplicate-column, rank-1, dependent-row, and wide projector lanes as complete. |
| Q-specific proof value is explicit before implementation. | Complete | Day 5 metadata gate requires intended claim, metric, output shape, tolerances, diagnostics, and non-claims before code edits. |
| Sprint 128 residual debt remains in the end-of-epic queue unless directly required. | Complete | No-reopen boundary keeps near-threshold, SuiteSparse corpus, wide residual, optional-large, extra exact, and QR-vs-SVD work deferred. |
