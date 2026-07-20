# Sprint 128 Day 3 Compatible and Wide Residual Evidence Decision

## Purpose

Day 3 applies the Day 2 compatible zero-residual and wide residual-only
semantics policy. The accepted evidence, if any, must add distinct residual
trust without duplicating completed Sprint 125-127 residual, compatible,
projector, threshold, or minimum-norm lanes or implying solution selection,
minimum-norm behavior, nullspace/subspace behavior, Q/economy behavior,
sparse-mode behavior, SuiteSparse corpus behavior, or broad QR parity.

## Day 3 Decision

Day 3 explicitly defers all compatible zero-residual and wide residual-only QR
evidence candidates.

No candidate satisfies the Day 2 trust-value rules without weakening the proof
boundary. The strongest compatible candidate,
`qr_rankdef_duplicate_5x4_compatible_zero_residual`, would reuse the same
duplicate-column matrix family already covered by rank-only and residual-only
external fixtures, while a zero-residual assertion would be easy to misread as
solution-selection or minimum-norm evidence. The wide candidates still depend
on underdetermined output semantics, rank/nullity boundaries, and
Q/economy/sparse-mode boundaries that belong to later Sprint 128 and Sprint
129 owners.

Day 3 does not change C tests, headers, Python helpers, Matrix Market data,
build metadata, maintainer wording, public solver wording, or public claims.

## Candidate Review

| Candidate | Disposition | Reason |
| --- | --- | --- |
| `qr_rankdef_duplicate_5x4_compatible_zero_residual` | Deferred | Reuses the completed duplicate-column fixture family and adds only another zero-residual classification unless a future owner proves a distinct diagnostic beyond existing full-rank compatible solve, duplicate-column rank-only, duplicate-column residual-only, and rank-deficient solve-smoke evidence. |
| `qr_rankdef_dependent_row_4x3_compatible_zero_residual` | Deferred | Overlaps Sprint 126's accepted dependent-row residual-only lane and Sprint 127's dependent-row projector lane; no distinct residual-only proof value is pinned today. |
| New synthetic compatible rank-deficient fixture | Deferred | A new matrix family is unjustified while existing fixture builders already cover duplicate-column and dependent-row structures; no non-duplicate proof value is pinned. |
| SuiteSparse compatible zero-residual fixture | Deferred | Requires independent expected rank/nullity metadata, threshold semantics, support tier, optional-data behavior, runtime budget, skip behavior, and diagnostics owned by Days 8-11. |
| `qr_rankdef_wide_3x5_residual_only` | Deferred | Wide residual behavior cannot be interpreted safely until rank/nullity, underdetermined output semantics, solution-selection boundaries, and Q/economy/sparse-mode fences are explicit. |
| `qr_wide_compatible_zero_residual_only` | Deferred | Near-zero residual for a wide compatible system risks implying solution selection or minimum-norm behavior and adds little beyond existing wide and minimum-norm evidence. |
| `qr_wide_incompatible_residual_only` | Deferred | Many wide systems are compatible for all RHS when full row rank; a meaningful incompatible residual-only claim needs rank, row-space, and RHS feasibility metadata that is not pinned. |
| Wide sparse-mode residual-only fixture | Deferred | Requires sparse-mode output semantics, Q/economy boundaries, and residual-only proof value owned by Sprint 129 Q/economy/sparse-mode follow-through. |

## Existing Evidence Preserved

| Existing Lane | Owner | Day 3 Handling |
| --- | --- | --- |
| `qr_rankdef_duplicate_5x4_rank_only` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | Preserved as completed rank-only evidence; not relabeled as compatible zero-residual proof. |
| `qr_rankdef_duplicate_5x4_residual_only` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | Preserved as completed non-zero residual-only evidence; not duplicated with a zero-residual variant today. |
| `qr_rankdef_dependent_row_4x3_residual_only` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | Preserved as completed second-structure residual-only evidence; not duplicated with a compatible residual variant. |
| `qr_rankdef_dependent_row_4x3_nullspace_projector` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py` | Preserved as completed projector evidence; not used as compatible residual proof. |
| `qr_overdetermined_compatible_5x3` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | Preserved as full-rank compatible external-reference evidence; not treated as rank-deficient proof. |
| `test_qr_solve_rank_deficient` | `tests/test_qr_solve.c` | Preserved as product rank-deficient solve smoke; not converted into external compatible zero-residual evidence. |
| `test_qr_wide`, `test_economy_wide`, `test_sparse_mode_wide` | `tests/test_qr.c` | Preserved as wide/Q/economy/sparse-mode coverage; not converted into residual-only proof. |
| Exact and owner-local minimum-norm lanes | `tests/test_qr_solve.c`, `tests/test_colamd.c` | Preserved as minimum-norm evidence; not used to justify residual-only wide claims. |

## Why No Code Was Added

Adding a Day 3 fixture would have required at least one of the following
unsupported interpretations:

1. Treating another zero-residual fixture as distinct trust without proving
   why existing compatible solve and rank-deficient residual evidence are
   insufficient.
2. Treating a returned wide-system solution as residual-only while leaving
   solution-selection behavior unspecified.
3. Letting a residual-only wide test imply minimum-norm, nullspace,
   Q/economy, or sparse-mode behavior.
4. Registering SuiteSparse or optional-large residual evidence before corpus
   metadata, support tier, runtime, and skip behavior are pinned.

The Sprint 128 Day 3 action is to preserve the proof boundary and hand each
candidate to the owner that can satisfy its missing semantics.

## Future Promotion Gates

| Deferred Candidate | Future Owner | Promotion Gate |
| --- | --- | --- |
| Duplicate-column compatible zero-residual fixture | Future QR residual owner | Prove a named zero-residual diagnostic adds trust beyond existing full-rank compatible solve, duplicate-column rank-only, duplicate-column residual-only, and rank-deficient solve-smoke evidence; assert residual only. |
| Dependent-row compatible zero-residual fixture | Future QR residual owner | Show why compatible dependent-row zero residual adds a new diagnostic after Sprint 126's dependent-row residual-only lane and Sprint 127's dependent-row projector lane; assert residual only. |
| New synthetic compatible fixture | Future QR residual owner | Justify the new family, fixture key, rank-deficient structure, compatible RHS, exact residual class, diagnostics, and duplicate fence. |
| SuiteSparse compatible zero-residual fixture | Days 8-11 corpus owners | Pin matrix path, extraction rule if any, expected rank/nullity, threshold semantics, support tier, optional-data behavior, runtime budget, skip behavior, diagnostics, and validation. |
| Wide residual-only fixture | Day 4-5 nullspace/subspace owner, Day 10-13 minimum-norm owner, or Sprint 129 Q/economy owner | Define rank/nullity, underdetermined output semantics, solution-selection policy, Q/economy boundaries, residual-only proof value, diagnostics, and non-claims. |
| Wide sparse-mode residual fixture | Sprint 129 Q/economy/sparse-mode owner | Define sparse-mode output semantics, economy/Q boundaries, residual-only proof value, and validation requirements. |

## Residual Non-Claim Register

Day 3 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad dense-library parity;
- broad QR factorization, QR solve, compatible solve, wide solve,
  rank-deficient solve, nullspace, minimum-norm, Q-basis, economy,
  sparse-mode, reorder, backend, corpus, optional-data, platform, or
  performance parity;
- new compatible zero-residual or wide residual-only evidence;
- new rank, nullity, nullspace, projector, subspace, Q-basis, economy, or
  sparse-mode evidence;
- solution-vector uniqueness, solution equality, solution-selection policy,
  solution-norm optimality, or minimum-norm optimality;
- QR-vs-SVD-pseudoinverse oracle behavior or dense-library parity;
- SuiteSparse corpus, optional-data, runtime, platform, or performance
  support;
- global near-rank-deficient threshold, default-threshold, or numerical-rank
  policy;
- generic QR/SVD helper API or helper consolidation behavior;
- package, ABI, public API, CMake, Makefile, CI, CTest, scalability, memory,
  or state-of-the-art behavior.

## Validation Notes

Day 3 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_128`

No `.c`, `.h`, Python helper, build, public API, maintainer, or public wording
files changed, so no code quality gate is required for Day 3.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 2 is complete or explicitly deferred. | Complete | All compatible zero-residual and wide residual-only candidates are explicitly deferred with future-owner gates. |
| Accepted evidence proves only documented residual behavior. | Complete | No evidence was accepted; existing residual lanes remain bounded by their original artifacts. |
| Focused validation evidence is recorded for code or script changes. | Not applicable | Day 3 made no code or script changes. |
