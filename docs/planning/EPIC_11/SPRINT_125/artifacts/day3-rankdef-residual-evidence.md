# Sprint 125 Day 3 Rank-Deficient Residual Evidence Batch

## Purpose

Day 3 applies the Day 2 residual-only trust gate. The accepted lane must add
rank-deficient QR residual evidence without asserting rank, nullspace,
minimum-norm, pseudoinverse, Q-basis, economy, SuiteSparse, backend, or broad
QR parity.

## Inputs Reviewed

| Input | Decision Use |
| --- | --- |
| Sprint 125 Plan Day 3 | Requires accepted residual evidence or explicit deferral, fixture/reference protocol, focused validation, and updated non-claims. |
| Sprint 125 Day 2 trust gate | Defines residual-only proof boundaries, preferred duplicate-column candidate, diagnostics, tolerance, and validation checklist. |
| Sprint 124 Day 3 rank decision | Provides the completed 5x4 duplicate-column rank-only fixture and duplicate fence. |
| `tests/qr_external_dense_reference.py` | Existing Python standard-library QR external reference helper. |
| `tests/test_qr_solve.c` | Existing QR solve and external QR fixture test owner. |
| `docs/maintainer_guide.md` | Maintainer evidence table and QR non-claim owner. |

## Decision Summary

Accepted one bounded residual-only rank-deficient QR fixture:

`qr_rankdef_duplicate_5x4_residual_only`

The fixture reuses the exact 5x4 duplicate-column matrix from
`qr_rankdef_duplicate_5x4_rank_only`, adds an incompatible RHS, and compares
only the least-squares residual norm against a standard-library external
column-space projection reference.

This lane proves only that the product QR solve residual agrees with the
external reference residual for one named rank-deficient fixture. It does not
assert solution-vector equality, rank, nullity, nullspace vectors, subspace
projection, minimum-norm optimality, SVD-pseudoinverse agreement, Q-basis
orientation, economy behavior, SuiteSparse behavior, backend behavior, or
broad QR parity.

## Accepted Fixture Contract

| Field | Decision |
| --- | --- |
| Fixture key | `qr_rankdef_duplicate_5x4_residual_only` |
| Matrix shape | 5x4 dense tall matrix |
| Structural model | Column 4 duplicates column 2 exactly; the same matrix as the completed rank-only fixture. |
| RHS | `[1.0, -2.0, 2.0, 5.0, -1.0]` |
| Expected residual | Computed by projecting `b` onto the column space with a standard-library Gram-Schmidt basis and reporting `||b - P_col(A)b||_2`. |
| Reference path | `tests/qr_external_dense_reference.py` emits one residual value through `column_space_residual_reference()`. |
| Product path | `tests/test_qr_solve.c` factors the matching sparse fixture, calls `sparse_qr_solve`, and compares the returned residual to the external reference. |
| Output protocol | `OK 1`, then the expected residual norm. |
| Tolerance | Absolute residual difference below `1e-8`. |
| Asserted quantities | Returned residual norm and non-zero reference residual. |
| Diagnostic quantities | Recomputed true residual from the product solution, printed but not used as a separate oracle claim. |
| Windows behavior | Preserves existing external QR helper skip behavior. |
| Missing Python behavior | Preserves existing external-reference helper skip behavior. |
| Helper `ERROR` behavior | Test failure. |
| Build membership impact | None. The test is registered inside existing `test_qr_solve`; no new executable, Makefile entry, CMake entry, or CTest member is added. |

## Implemented Changes

| Surface | Change |
| --- | --- |
| `tests/qr_external_dense_reference.py` | Added `qr_rankdef_duplicate_5x4_residual_only`, an incompatible RHS, and `column_space_residual_reference()` for residual-only output. |
| `tests/test_qr_solve.c` | Added a shared 5x4 duplicate-column fixture builder and `test_qr_external_dense_reference_rankdef_duplicate_5x4_residual_only`. |
| `docs/maintainer_guide.md` | Added the bounded residual-only fixture to the QR maintained evidence row while preserving non-claims. |
| Makefile | No change. |
| CMake / CTest | No change. |
| Public API / public solver-selection docs | No change. |

## Deferred Rank-Deficient Residual Work

| Deferred Work | Future Owner | Promotion Gate |
| --- | --- | --- |
| Compatible zero-residual rank-deficient external fixture | Future QR solve oracle owner | Prove zero-residual evidence adds distinct trust beyond deterministic compatible solve behavior and cannot be misread as minimum-norm proof. |
| Dependent-row rank-deficient residual external fixture | Future QR residual owner | Show a second structural rank-deficient family adds trust beyond duplicate-column evidence without duplicating deterministic dependent-row tests. |
| Wide rank-deficient residual fixture | Future minimum-norm or nullspace/subspace owner | Define underdetermined, solution-selection, minimum-norm, and nullspace boundaries before accepting residual evidence. |
| SuiteSparse rank-deficient residual fixture | Sprint 125 Days 8-9 corpus/platform owner | Define optional corpus, platform skip behavior, support tier, diagnostics, validation, and claim boundaries. |

## Validation Checklist

Day 3 touched `.c` and Python helper files, so the required validation was:

1. `python3 -m py_compile tests/qr_external_dense_reference.py`
2. `python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_residual_only`
3. `make build/test_qr_solve && ./build/test_qr_solve`
4. `make format`
5. `make lint`
6. `make test`
7. `git diff --check`
8. Focused trailing-whitespace scan over Sprint 125 files and touched
   maintainer/test/helper files

## Validation Notes

Focused validation passed:

1. `python3 -m py_compile tests/qr_external_dense_reference.py`
2. `python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_residual_only`
   emitted `OK 1` and residual `3.7886027630095733`.
3. `make build/test_qr_solve && ./build/test_qr_solve` passed with 18 tests,
   0 failures, 0 skips, and 1089 assertions. The new fixture reported
   residual `3.789e+00`, expected `3.789e+00`, and diff `8.882e-16`.

Full required quality validation passed:

1. `make format`
2. `make lint`
3. `make test`

The final `make test` phase ended with `All tests passed.`

## Non-Claim Register

Day 3 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad dense-library parity;
- broad QR factorization, QR solve, least-squares, rank-deficient solve,
  nullspace, minimum-norm, Q-basis, economy, sparse-mode, reorder, backend,
  corpus, or performance parity;
- new rank evidence beyond the already completed
  `qr_rankdef_duplicate_5x4_rank_only` fixture;
- solution-vector uniqueness or solution equality for the accepted
  rank-deficient residual fixture;
- raw nullspace basis equality, sign/orientation, unique-basis, projection, or
  subspace external parity;
- minimum-norm optimality, solution-norm optimality, COLAMD, fallback,
  refinement, QR-vs-SVD-pseudoinverse, or SuiteSparse minimum-norm behavior;
- global near-rank-deficient threshold policy;
- package, ABI, platform, public API, CMake, Makefile, CI, CTest, performance,
  scalability, memory, or state-of-the-art behavior.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 2 is complete or explicitly deferred. | Complete | Bounded residual-only fixture accepted and implemented; other candidate lanes deferred with promotion gates. |
| Accepted code or script changes have focused validation evidence. | Complete | See focused validation notes and full quality validation. |
| Residual-only proof boundaries remain documented. | Complete | See accepted fixture contract and non-claim register. |
