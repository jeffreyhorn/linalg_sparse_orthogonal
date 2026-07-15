# Sprint 125 Day 2 Residual-Only Rank-Deficient QR Trust Gate

## Purpose

Day 2 defines when residual-only rank-deficient QR evidence is worth adding.
The gate exists because Sprint 124 already added a bounded rank-only external
fixture, and the project still has deterministic rank-deficient solve
coverage. Sprint 125 must prove any new residual-only lane adds distinct trust
without implying nullspace, minimum-norm, pseudoinverse, Q-basis, economy,
SuiteSparse, backend, or broad QR parity.

This is a policy artifact only. No C source, header, Python helper, build,
CMake, CTest, workflow, public API, or public wording changes are made by Day
2.

## Inputs Reviewed

| Input | Trust-Gate Use |
| --- | --- |
| Sprint 125 Plan Day 2 | Requires candidate residual-only fixtures, trust-beyond-deterministic rationale, proof boundaries, diagnostics, tolerances, skips, and Day 3 checklist. |
| Sprint 125 Day 1 artifact | Provides duplicate fences and day-level owners for Sprint 124 carry-forward debt. |
| Sprint 124 Day 2 rank policy | Defines rank-threshold, nullspace, minimum-norm, tolerance, skip, and failure-interpretation boundaries. |
| Sprint 124 Day 3 rank decision | Completed `qr_rankdef_duplicate_5x4_rank_only` and deferred residual-only rank-deficient QR evidence. |
| Sprint 124 Day 4-5 minimum-norm artifacts | Define minimum-norm owner boundaries that residual-only evidence must not absorb. |
| Sprint 124 Day 13-14 artifacts | Provide validation baseline, maintainer evidence, solver-selection no-update rationale, and future-owner queue. |
| `tests/test_qr.c` | Owns deterministic QR rank, nullspace, diagonal-threshold, reconstruction, Q/economy, sparse-mode, reorder, and refinement evidence. |
| `tests/test_qr_solve.c` | Owns QR solve residual evidence and bounded external QR solve fixtures. |
| `tests/test_colamd.c` | Owns QR minimum-norm, COLAMD, fallback, refinement, QR-vs-SVD-pseudoinverse, and optional SuiteSparse minimum-norm scenarios. |
| `tests/qr_external_dense_reference.py` | Owns the current Python standard-library external QR reference protocols. |

## Current Residual Evidence Inventory

| Evidence Class | Current Owner | Evidence Summary | Day 2 Interpretation |
| --- | --- | --- | --- |
| Full-rank incompatible overdetermined external residual | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_overdetermined_incompatible_4x2` compares solution and non-zero residual against a bounded external helper. | Completed full-rank least-squares residual evidence; do not duplicate as rank-deficient evidence. |
| Full-rank compatible overdetermined external residual | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_overdetermined_compatible_5x3` compares solution and near-zero residual. | Completed compatible solve evidence; not rank-deficient. |
| Rank-only duplicate-column external fixture | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_rankdef_duplicate_5x4_rank_only` checks product rank `3` at threshold `0.0`. | Completed rank evidence; it does not check a solve residual. |
| Deterministic rank-deficient solve residual | `tests/test_qr_solve.c` | `test_qr_solve_rank_deficient` checks product residual behavior for a local rank-deficient solve. | Internal baseline; a new external lane must add independent reference trust. |
| Deterministic null residual handling | `tests/test_qr_solve.c` | `test_qr_solve_null_residual` verifies QR solve behavior when the residual pointer is null. | API behavior, not external residual evidence. |
| Deterministic rank/nullspace tests | `tests/test_qr.c` | Rank-deficient, known-nullspace, rectangular nullspace, and dependent-row fixtures validate internal QR behavior. | Useful baseline; residual-only external evidence must not claim nullspace or basis behavior. |
| Minimum-norm residual/norm checks | `tests/test_qr_solve.c`, `tests/test_colamd.c` | Exact 2x4 and broader COLAMD/fallback/refinement/rank-deficient minimum-norm tests check residual and norm behavior. | Separate owner; residual-only rank-deficient QR must not imply minimum-norm optimality. |

## Residual-Only Candidate Table

| Candidate | Candidate Shape | Trust Value | Risk | Day 2 Disposition |
| --- | --- | --- | --- | --- |
| `qr_rankdef_duplicate_5x4_residual_only` | Existing 5x4 duplicate-column matrix with a deliberately incompatible RHS and non-zero least-squares residual | High enough for Day 3 consideration because it reuses the completed rank-only structural fixture while adding independent solve-residual reference evidence. | Must avoid comparing solution uniqueness or minimum-norm; rank-deficient systems may have non-unique minimizers. | Preferred Day 3 candidate if output protocol reports residual only, or residual plus diagnostic solution values that are not asserted as unique. |
| `qr_rankdef_duplicate_5x4_compatible_zero_residual` | Existing 5x4 duplicate-column matrix with RHS in the column space | Low. A zero residual mostly repeats deterministic compatible solve behavior and can be misread as minimum-norm proof. | High claim-confusion risk because many solutions can produce zero residual. | Defer unless Day 3 proves a distinct diagnostic value. |
| `qr_rankdef_dependent_row_4x3_residual_only` | Local dependent-row style fixture adapted to external helper | Moderate. Could cover a second structural rank-deficient family. | Duplicates deterministic dependent-row coverage and may blur rank/nullspace evidence. | Defer until the duplicate-column lane is accepted or rejected. |
| `qr_rankdef_wide_residual_only` | Wide rank-deficient system | Low for Sprint 125 Day 3. | Too close to minimum-norm, nullspace, and underdetermined solve semantics. | Defer to minimum-norm or nullspace/subspace owners. |
| SuiteSparse rank-deficient residual-only fixture | Optional corpus matrix | Potentially high later. | Requires optional corpus, platform skip, support tier, diagnostics, and claim boundaries. | Defer to Days 8-9. |

## Trust-Beyond-Deterministic Rationale

Residual-only rank-deficient QR evidence is worth accepting only if it adds all
of the following beyond existing tests:

1. It uses a named rank-deficient fixture whose structural rank model is
   already explicit.
2. It computes the least-squares residual through the external helper rather
   than from the product implementation.
3. It chooses an RHS that makes the residual meaningful; a non-zero residual is
   preferred because it exercises the least-squares residual path instead of
   merely confirming consistency.
4. It compares only residual quantities unless a separate policy justifies
   asserting solution values.
5. It preserves existing rank-only evidence as rank evidence and does not
   convert it into solve, nullspace, or minimum-norm proof.

A candidate should be deferred if it only repeats deterministic solve behavior,
depends on an unstated rank threshold, requires raw nullspace or basis
comparison, or needs SuiteSparse/platform policy before it can be interpreted.

## Proof Boundary

| Topic | Residual-Only Rule |
| --- | --- |
| Rank | Use the existing fixture rank as input context only. Do not treat residual agreement as a new rank proof. |
| Solve residual | May compare product residual against a standard-library external reference for a named fixture. |
| Solution vector | Do not assert solution-vector equality for a rank-deficient least-squares fixture unless Day 3 explicitly proves a unique selected solution policy. |
| Nullspace | No nullity, nullspace vector, projection, subspace, or basis orientation claim. |
| Minimum-norm | No solution-norm or minimum-norm optimality claim. Those lanes belong to Days 10-12. |
| Pseudoinverse | No QR-vs-SVD-pseudoinverse claim. That comparison belongs to Day 12. |
| Q-basis/economy | No Q column, sign, orientation, projector, economy, or sparse-mode claim. |
| Corpus/backend | No SuiteSparse, backend, reorder, performance, or platform claim. |

## Residual Tolerance and Diagnostic Policy

| Policy Point | Requirement |
| --- | --- |
| Reference implementation | Use Python standard library only; no NumPy, SciPy, LAPACK, BLAS, SuiteSparse, or external package dependency. |
| Fixture size | Prefer the existing 5x4 duplicate-column fixture to avoid adding a new matrix family on Day 3. |
| RHS | Prefer an incompatible RHS that produces a non-zero least-squares residual. |
| Output protocol | Prefer `OK 1` plus residual norm for residual-only evidence. If diagnostic values are emitted, each must be named and non-asserted unless Day 3 accepts its semantics. |
| Residual comparison | Compare absolute residual difference for tiny exact fixtures; Day 3 may add relative residual only if the fixture defines scale-sensitive behavior. |
| Product residual | Compare the returned QR solve residual and, where useful, independently recomputed true residual as diagnostics. |
| Tolerance | Candidate default is `1e-8` absolute residual difference, unless Day 3 derives a tighter fixture-local threshold. |
| Failure diagnostics | Identify fixture key, helper status, expected residual, product returned residual, recomputed true residual, and whether failure is helper protocol, solve status, residual mismatch, unsupported platform, or optional helper unavailability. |
| Windows behavior | Preserve the existing explicit external-helper skip pattern unless a platform sprint promotes it. |
| Missing Python behavior | Preserve existing external-reference helper skip behavior. |
| Helper `ERROR` output | Treat as test failure for an accepted residual-only fixture. |

## Day 3 Implementation or Deferral Checklist

Day 3 should evaluate residual-only evidence in this order:

1. Start from the completed `qr_rankdef_duplicate_5x4_rank_only` fixture and
   reuse its structural matrix if possible.
2. Choose an RHS that creates a meaningful non-zero residual while avoiding
   solution uniqueness and minimum-norm claims.
3. Define the external helper protocol, expected output count, tolerance,
   diagnostics, skip behavior, and affected test owner before editing code.
4. Assert residual agreement only; do not assert raw solution equality,
   solution norm, nullspace vectors, pseudoinverse agreement, or Q-basis
   behavior.
5. If accepted, run focused helper validation, focused `test_qr_solve`, and
   full `make format && make lint && make test` because `.c` or helper files
   will likely change.
6. If deferred, name the future owner, blocker, and promotion gate in a Day 3
   decision artifact.
7. Update maintainer evidence only if the accepted lane is implemented and
   validated; otherwise preserve the current public wording.

## Non-Claim Register

Day 2 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad dense-library parity;
- broad QR factorization, QR solve, least-squares, rank-deficient solve,
  nullspace, minimum-norm, Q-basis, economy, sparse-mode, reorder, backend,
  corpus, or performance parity;
- new rank evidence beyond the completed `qr_rankdef_duplicate_5x4_rank_only`
  fixture;
- raw nullspace basis equality, sign/orientation, unique-basis, projection, or
  subspace external parity;
- minimum-norm optimality, solution-norm optimality, COLAMD, fallback,
  refinement, QR-vs-SVD-pseudoinverse, or SuiteSparse minimum-norm behavior;
- global near-rank-deficient threshold policy;
- package, ABI, platform, public API, CMake, Makefile, CI, CTest, performance,
  scalability, memory, or state-of-the-art behavior.

## Validation Notes

Day 2 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_125`

No `.c`, `.h`, Python helper, build, public API, or public wording files
changed, so no code quality gate is required for Day 2.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Residual-only evidence has a clear proof boundary. | Complete | See proof boundary and residual tolerance/diagnostic policy. |
| No accepted candidate implies nullspace, minimum-norm, or pseudoinverse behavior. | Complete | See proof boundary and non-claim register. |
| Deferred candidates have explicit blockers and promotion gates. | Complete | See residual-only candidate table and Day 3 checklist. |
