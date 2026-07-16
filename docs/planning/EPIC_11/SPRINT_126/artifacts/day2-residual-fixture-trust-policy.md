# Sprint 126 Day 2 Residual Fixture Trust Policy

## Purpose

Day 2 decides which compatible zero-residual, dependent-row, and wide
rank-deficient QR residual fixtures can add distinct trust after Sprint 125's
completed residual-only duplicate-column evidence. The policy keeps residual
fixtures separate from nullspace, minimum-norm, pseudoinverse, Q-basis,
economy, SuiteSparse, backend, and broad QR claims.

This is a policy artifact only. No C source, header, Python helper, build,
CMake, CTest, workflow, public API, maintainer, or public wording files are
changed by Day 2.

## Inputs Reviewed

| Input | Trust-Policy Use |
| --- | --- |
| Sprint 126 Plan Day 2 | Requires candidate compatible/dependent-row/wide residual fixtures, trust-value analysis, proof boundaries, diagnostics, tolerances, skips, and Day 3 checklist. |
| Sprint 126 Day 1 artifact | Provides duplicate fences and day-level owners for Sprint 125 carry-forward debt. |
| Sprint 125 Day 2-3 artifacts | Define residual-only proof boundaries and completed `qr_rankdef_duplicate_5x4_residual_only` evidence. |
| Sprint 125 Day 4-5 artifacts | Define nullspace/subspace policy that Day 2 residual fixtures must not absorb. |
| Sprint 125 Day 10-12 artifacts | Define minimum-norm owner-local evidence and QR-vs-SVD bounded cross-check rules that Day 2 residual fixtures must not absorb. |
| `tests/test_qr.c` | Owns deterministic QR rank, dependent-row, wide, nullspace, Q/economy, sparse-mode, reorder, and refinement evidence. |
| `tests/test_qr_solve.c` | Owns QR solve residual evidence and bounded external QR solve fixtures. |
| `tests/test_colamd.c` | Owns QR minimum-norm, COLAMD, fallback, refinement, QR-vs-SVD-pseudoinverse, and SuiteSparse minimum-norm scenarios. |
| `tests/test_qr_helpers.h` | Owns shared QR fixture builders, including duplicate-column and dependent-row builders. |
| `tests/qr_external_dense_reference.py` | Owns current Python standard-library external QR reference protocols. |

## Current Evidence Inventory

| Evidence Class | Current Owner | Evidence Summary | Day 2 Interpretation |
| --- | --- | --- | --- |
| Rank-deficient duplicate-column residual-only external fixture | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_rankdef_duplicate_5x4_residual_only` compares returned product residual against a standard-library column-space residual reference. | Completed Sprint 125 residual-only baseline; do not duplicate it as compatible, dependent-row, wide, or SuiteSparse evidence. |
| Rank-deficient duplicate-column rank-only external fixture | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_rankdef_duplicate_5x4_rank_only` checks product rank `3`. | Completed rank evidence; not a residual fixture. |
| Full-rank compatible external residual fixture | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_overdetermined_compatible_5x3` checks solution and near-zero residual for a full-rank compatible system. | Completed compatible full-rank solve evidence; not rank-deficient. |
| Full-rank incompatible external residual fixture | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_overdetermined_incompatible_4x2` checks solution and non-zero residual for a full-rank least-squares system. | Completed full-rank residual evidence; not rank-deficient. |
| Dependent-row deterministic rank/nullspace fixture | `tests/test_qr.c`, `tests/test_qr_helpers.h` | `tf_qr_make_dependent_row_4x3` and `test_qr_rank_dependent_row_fixture` check rank, reconstruction, and null residual. | Good structural candidate, but deterministic rank/nullspace behavior is already covered. A residual-only lane must add external residual trust. |
| Wide QR deterministic fixtures | `tests/test_qr.c`, `tests/test_qr_solve.c` | `test_qr_wide`, `test_economy_wide`, `test_sparse_mode_wide`, and underdetermined solve checks cover wide behavior. | Existing wide evidence is close to minimum-norm/nullspace semantics; residual-only expansion needs a narrow contract. |
| Minimum-norm owner-local fixtures | `tests/test_colamd.c`, `tests/test_qr_solve.c` | Exact 2x4, COLAMD, fallback, rank-deficient, refinement, zero-row, QR-vs-SVD, and `west0067` submatrix checks cover minimum-norm behavior. | Separate owner; Day 2 residual fixtures must not assert solution norm or minimum-norm optimality. |

## Residual Fixture Candidate Table

| Candidate | Candidate Shape | Trust Value | Duplicate/Claim Risk | Day 2 Disposition |
| --- | --- | --- | --- | --- |
| `qr_rankdef_dependent_row_4x3_residual_only` | Existing dependent-row 4x3 fixture with an incompatible RHS and non-zero least-squares residual | Moderate. Adds a second structural rank-deficient family beyond duplicate columns, while keeping the fixture small enough for a standard-library external residual reference. | May duplicate deterministic dependent-row rank/nullspace coverage if it asserts rank, nullity, basis, or reconstruction. | Preferred Day 3 candidate if it asserts residual agreement only and treats rank/dependence as fixture context. |
| `qr_rankdef_duplicate_5x4_compatible_zero_residual` | Existing duplicate-column 5x4 matrix with RHS in the column space | Low. Confirms compatible consistency for a completed fixture, but adds little beyond deterministic compatible solve checks. | High minimum-norm and solution-selection confusion because many rank-deficient solutions can produce zero residual. | Defer by default unless Day 3 proves a new diagnostic that is not already covered. |
| `qr_rankdef_dependent_row_4x3_compatible_zero_residual` | Existing dependent-row 4x3 fixture with compatible RHS | Low to moderate. Could show compatibility for a second rank-deficient family. | Same zero-residual confusion risk, plus overlap with deterministic dependent-row null residual checks. | Defer unless the incompatible dependent-row lane is rejected and this one has a sharper diagnostic purpose. |
| `qr_rankdef_wide_3x5_residual_only` | Wide 3x5 rank-deficient or underdetermined fixture with residual-only comparison | Low for Day 3. Wide residual behavior is tightly coupled to solution-selection, nullspace, and minimum-norm interpretation. | High risk of implying underdetermined minimum-norm or nullspace behavior. | Defer to Days 10-13 or Day 4-5 owners unless Day 3 can fence solution-selection completely. |
| `qr_rankdef_wide_sparse_mode_residual_only` | Wide sparse-mode QR fixture with residual-only comparison | Low. Sparse-mode coverage already exists and residual-only external trust would require extra shape and output semantics. | Risks drifting into sparse-mode Q/economy and Sprint 127 scope. | Defer to Sprint 127 Q/economy/sparse-mode owners. |
| SuiteSparse rank-deficient residual-only fixture | Checked-in or optional SuiteSparse rank-deficient matrix/submatrix | Potentially high later. | Requires corpus support tier, expected-rank metadata, optional-data skip behavior, diagnostics, and validation. | Defer to Days 8-9 SuiteSparse corpus gate. |

## Trust-Value Decision Rules

A compatible, dependent-row, or wide residual fixture can be accepted only if it
adds all of the following:

1. A named fixture with explicit structural rank-deficient context.
2. A residual expected value from a standard-library external helper or an
   independently documented exact residual derivation.
3. A residual behavior not already covered by the completed duplicate-column
   residual-only fixture or deterministic solve tests.
4. A narrow assertion set: residual agreement and, if necessary, non-zero or
   near-zero residual classification.
5. Explicit refusal to assert solution equality, solution uniqueness, solution
   norm, minimum-norm optimality, nullspace basis, Q-basis, economy behavior,
   sparse-mode behavior, SuiteSparse corpus behavior, or external-library
   parity.

A candidate should be deferred when the only new signal is another zero
residual, when the shape cannot be interpreted without solution-selection
policy, when it depends on unstated rank/nullity metadata, or when it needs
SuiteSparse support-tier policy before failures can be understood.

## Proof Boundary

| Topic | Residual Fixture Rule |
| --- | --- |
| Rank | May use an explicit fixture rank as context, but residual agreement is not new rank proof. |
| Residual | May compare product residual against an independent fixture-local expected residual. |
| Compatible zero residual | May assert near-zero residual only if the fixture adds distinct trust beyond existing compatible solve checks. |
| Solution vector | Do not assert solution-vector equality for rank-deficient residual fixtures. |
| Minimum-norm | Do not assert solution norm, minimum-norm optimality, QR-vs-SVD agreement, or pseudoinverse behavior. |
| Nullspace/subspace | Do not assert nullity, basis vectors, projector agreement, principal angles, or raw basis orientation. |
| Q/economy/sparse mode | Do not assert Q columns, Q signs, economy shape, sparse-mode Q behavior, or Sprint 127 helper behavior. |
| SuiteSparse/corpus | Do not assert SuiteSparse support, optional-corpus behavior, or platform support from Day 2 residual policy. |

## Residual Tolerance and Diagnostic Policy

| Policy Point | Requirement |
| --- | --- |
| Reference implementation | Use Python standard library only unless a future artifact explicitly accepts a closed-form residual derivation. No NumPy, SciPy, LAPACK, BLAS, SuiteSparse, or external package dependency. |
| Fixture preference | Prefer existing helper builders, especially `tf_qr_make_dependent_row_4x3`, before adding new matrix families. |
| RHS preference | Prefer an incompatible RHS with a non-zero least-squares residual for residual-only evidence; zero residual requires stronger duplicate-risk justification. |
| Output protocol | Use explicit fixture keys and fixed output counts. Residual-only fixtures should emit expected residual first and avoid solution-vector oracle fields. |
| Residual comparison | Compare absolute residual difference for tiny fixtures; add relative residual only if Day 3 defines scale-sensitive behavior. |
| Product diagnostics | Print returned residual, recomputed true residual, expected residual, and absolute diff. Do not assert diagnostic solution values. |
| Candidate tolerance | Default absolute residual-difference tolerance is `1e-8` unless Day 3 derives a tighter fixture-local threshold. |
| Failure diagnostics | Include fixture key, helper status, expected residual, returned residual, recomputed true residual, diff, and failure class. |
| Windows behavior | Preserve existing external-helper skip pattern unless a platform sprint promotes the helper. |
| Missing Python behavior | Preserve existing external-reference helper skip behavior. |
| Helper `ERROR` behavior | Treat as test failure for any accepted fixture. |

## Day 3 Implementation or Deferral Checklist

Day 3 should evaluate candidates in this order:

1. Start with `qr_rankdef_dependent_row_4x3_residual_only` because it is the
   only Day 2 candidate with moderate trust value and bounded duplicate risk.
2. Confirm the sparse fixture can reuse `tf_qr_make_dependent_row_4x3` or a
   clearly equivalent helper/reference matrix.
3. Choose an RHS with a non-zero least-squares residual and no solution
   uniqueness or minimum-norm assertion.
4. Define helper protocol, expected output count, tolerance, skip behavior, and
   diagnostics before editing code.
5. Assert residual agreement only; do not assert solution equality, solution
   norm, rank, nullity, nullspace vectors, pseudoinverse agreement, Q-basis,
   economy, sparse-mode, or SuiteSparse behavior.
6. If accepted and C/Python changes are made, run helper validation, focused
   `test_qr_solve` or `test_qr`, and full `make format && make lint && make
   test`.
7. If deferred, name the exact blocker, future owner, and promotion gate in the
   Day 3 decision artifact.

## Deferred Candidate Promotion Gates

| Deferred Candidate | Future Owner | Promotion Gate |
| --- | --- | --- |
| Duplicate-column compatible zero-residual fixture | Future QR residual owner | Prove zero residual adds trust beyond existing compatible solves and cannot be read as minimum-norm or solution-selection evidence. |
| Dependent-row compatible zero-residual fixture | Future QR residual owner | Show a compatible dependent-row residual has a distinct diagnostic purpose after the incompatible dependent-row lane is accepted or rejected. |
| Wide residual-only fixture | Minimum-norm or nullspace/subspace owner | Define underdetermined solution-selection boundaries, expected residual, and proof wording that excludes minimum-norm and nullspace claims. |
| Wide sparse-mode residual fixture | Sprint 127 Q/economy/sparse-mode owner | Define sparse-mode output semantics, economy/Q boundaries, and residual-only proof value. |
| SuiteSparse rank-deficient residual fixture | Days 8-9 SuiteSparse corpus owner | Define expected-rank metadata, support tier, optional-data behavior, diagnostics, skip behavior, and validation requirements. |

## Non-Claim Register

Day 2 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad dense-library parity;
- broad QR factorization, QR solve, least-squares, rank-deficient solve,
  nullspace, minimum-norm, Q-basis, economy, sparse-mode, reorder, backend,
  corpus, or performance parity;
- new residual evidence beyond completed Sprint 125 fixtures;
- new rank, nullity, nullspace, projector, subspace, Q-basis, economy, or
  sparse-mode evidence;
- solution-vector uniqueness, solution equality, solution-norm optimality, or
  minimum-norm optimality;
- QR-vs-SVD-pseudoinverse oracle behavior or dense-library parity;
- SuiteSparse corpus, optional-data, platform, or performance support;
- global near-rank-deficient threshold policy;
- package, ABI, platform, public API, CMake, Makefile, CI, CTest,
  performance, scalability, memory, or state-of-the-art behavior.

## Validation Notes

Day 2 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_126`

No `.c`, `.h`, Python helper, build, public API, maintainer, or public wording
files changed, so no code quality gate is required for Day 2.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Each accepted candidate has distinct trust value. | Complete | See residual fixture candidate table and trust-value decision rules. |
| Residual-only evidence cannot imply nullspace or minimum-norm behavior. | Complete | See proof boundary and non-claim register. |
| Deferred candidates have explicit blockers and promotion gates. | Complete | See deferred candidate promotion gates and Day 3 checklist. |
