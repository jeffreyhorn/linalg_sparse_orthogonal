# Sprint 124 Day 5 QR Minimum-Norm Decision and Bounded Batch

## Purpose

Day 5 decides whether Sprint 124 can add bounded QR minimum-norm external
evidence under the Day 4 behavior contract. Any accepted evidence must remain
scenario-specific and must not hide COLAMD, fallback, refinement,
rank-deficient, SVD-pseudoinverse, or SuiteSparse ownership behind a generic
minimum-norm label.

## Inputs Reviewed

| Input | Decision Use |
| --- | --- |
| Sprint 124 Plan Day 5 | Requires accepted/deferred QR minimum-norm decision, fixture/reference protocol, focused validation plan, and non-claim update. |
| Sprint 124 Day 4 minimum-norm behavior contract | Defines behavior-specific acceptance criteria, residual/norm policy, helper ownership, optional skip policy, and Day 5 decision criteria. |
| Sprint 123 Day 7 minimum-norm decision | Defers minimum-norm external evidence until a behavior-specific owner can preserve QR/COLAMD/SVD/fallback/refinement/SuiteSparse boundaries. |
| Sprint 123 Day 11 helper migration decision | Forbids generic minimum-norm helpers that hide scenario-local assertions. |
| `tests/qr_external_dense_reference.py` | Existing Python standard-library external QR helper. |
| `tests/test_qr_solve.c` | Focused QR solve scenario owner. |
| `docs/maintainer_guide.md` | Maintainer evidence table and QR non-claim owner. |

## Decision Summary

Accepted one bounded exact underdetermined minimum-norm fixture:

`qr_underdetermined_minnorm_2x4`

The fixture proves only that the product QR minimum-norm solve agrees with a
tiny standard-library reference for one exact 2x4 underdetermined system. It
compares the solution vector, residual norm, and solution 2-norm. It does not
claim broad minimum-norm parity, COLAMD/reorder behavior, fallback behavior,
refinement behavior, rank-deficient minimum-norm behavior, SuiteSparse corpus
support, or SVD-pseudoinverse oracle parity.

## Accepted Fixture Contract

| Field | Decision |
| --- | --- |
| Fixture key | `qr_underdetermined_minnorm_2x4` |
| Matrix shape | 2x4 dense underdetermined matrix |
| Matrix | `[1 1 0 0; 0 0 1 1]` |
| RHS | `[1, 1]` |
| Expected solution | `[0.5, 0.5, 0.5, 0.5]` |
| Expected residual norm | `0.0` |
| Expected solution norm | `1.0` |
| Reference path | `tests/qr_external_dense_reference.py` computes `x = A^T (A A^T)^{-1} b` with Python standard-library arithmetic. |
| Product path | `tests/test_qr_solve.c` calls `sparse_qr_solve_minnorm` on the matching sparse fixture. |
| Output protocol | `OK 6`, then four solution values, residual norm, and solution norm. |
| Tolerance | Absolute max solution difference, residual difference, and norm difference below `1e-8`. |
| Windows behavior | Preserve existing external QR helper skip behavior on Windows. |
| Missing Python behavior | Preserve existing external-reference helper skip behavior. |
| Helper `ERROR` behavior | Test failure. |
| Build membership impact | None. The test is registered inside existing `test_qr_solve`; no new executable, Makefile entry, CMake entry, or CTest member is added. |

## Implemented Changes

| Surface | Change |
| --- | --- |
| `tests/qr_external_dense_reference.py` | Added `qr_underdetermined_minnorm_2x4` and a standard-library minimum-norm reference path that emits solution, residual, and norm. |
| `tests/test_qr_solve.c` | Added fixture allow-list entry, sparse fixture construction, external reference read for six values, solution/residual/norm comparison, and existing-suite registration. |
| `docs/maintainer_guide.md` | Updated the QR trust-boundary row to include the bounded exact minimum-norm fixture while preserving broad minimum-norm non-claims. |
| Makefile | No change. |
| CMake / CTest | No change. |
| Public API / public docs | No change. |

## Deferred Minimum-Norm Work

| Deferred Work | Future Owner | Promotion Gate |
| --- | --- | --- |
| COLAMD/reordered external minimum-norm evidence | Future QR minimum-norm/COLAMD owner | Define ordering options, expected residual/norm behavior, and non-superiority wording. |
| Overdetermined and square fallback external evidence | Future QR fallback owner | Define whether the fixture asserts ordinary QR solve fallback rather than underdetermined minimum-norm behavior. |
| Rank-deficient minimum-norm external evidence | Future rank-deficient/minimum-norm owner | Combine Day 2 rank policy with solution norm, residual, and nullspace boundaries. |
| Refinement external minimum-norm evidence | Future QR refinement owner | Define before/after residual expectations and iteration-budget semantics. |
| QR-vs-SVD-pseudoinverse external evidence | Future QR/SVD cross-solver owner | Define whether SVD pseudoinverse is an oracle, cross-check, or independent behavior owner. |
| SuiteSparse minimum-norm external evidence | Future corpus/platform owner | Define corpus availability, skip behavior, platform implications, and support-tier wording. |

## Validation Checklist

Day 5 touched `.c` and Python helper files, so the required code gate is:

1. `python3 tests/qr_external_dense_reference.py qr_underdetermined_minnorm_2x4`
2. `make build/test_qr_solve && ./build/test_qr_solve`
3. `make format`
4. `make lint`
5. `make test`
6. `git diff --check`
7. Focused trailing-whitespace scan over Sprint 124 files and touched QR files

The helper must emit `OK 6`, four `0.5` solution values, residual `0`, and
norm `1`. A different output count is a fixture protocol failure.

## Failure Diagnostics

The accepted test identifies:

- fixture key;
- reference helper status;
- solution max difference;
- residual difference;
- solution norm difference;
- whether the failure is helper protocol, QR minimum-norm solve, residual,
  solution norm, unsupported platform, or optional-helper unavailability.

## Non-Claim Register

Day 5 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  broad external dense-library parity;
- broad QR minimum-norm external oracle parity;
- global minimum-norm optimality beyond the named 2x4 fixture;
- COLAMD, reorder, fallback, refinement, rank-deficient, SuiteSparse, or
  SVD-pseudoinverse parity;
- rank-deficient solve, nullspace, Q-basis, economy-mode, sparse-mode, or
  backend parity;
- package, ABI, platform, public API, CMake, Makefile, CI, or CTest expansion;
- performance, scalability, memory behavior, or state-of-the-art behavior.

## Validation Notes

Focused validation passed before full quality:

1. `python3 tests/qr_external_dense_reference.py qr_underdetermined_minnorm_2x4`
   emitted `OK 6`, four `0.5` solution values, residual `0`, and norm `1`.
2. `make build/test_qr_solve && ./build/test_qr_solve` passed with 17 tests, 0
   failures, 0 skips, and 1069 assertions.

Full required quality validation passed:

1. `make format`
2. `make lint`
3. `make test`

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 2 is complete or explicitly deferred. | Complete | Bounded exact 2x4 minimum-norm fixture accepted and implemented; broader work is explicitly deferred. |
| Accepted evidence remains behavior-specific. | Complete | Fixture compares only exact solution, residual, and norm for one named underdetermined system. |
| Deferred evidence has clear future ownership. | Complete | See deferred minimum-norm work table. |
