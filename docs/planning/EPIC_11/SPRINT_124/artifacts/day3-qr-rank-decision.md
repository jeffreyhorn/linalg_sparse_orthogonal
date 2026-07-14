# Sprint 124 Day 3 Rank-Deficient QR Decision and Bounded Batch

## Purpose

Day 3 decides the rank-deficient QR external evidence lane from the Day 2
policy. The selected work must either add a bounded rank-only fixture or
explicitly defer the lane with future owners and promotion gates. The decision
must not turn rank-only evidence into nullspace, minimum-norm, pseudoinverse,
Q-basis, economy, dense-library, or broad QR parity claims.

## Inputs Reviewed

| Input | Decision Use |
| --- | --- |
| Sprint 124 Plan Day 3 | Requires accepted/deferred rank-deficient QR decision, fixture/reference protocol, validation checklist, and non-claim update. |
| Sprint 124 Day 2 rank policy | Defines rank-threshold, nullspace, pseudoinverse, minimum-norm, tolerance, skip, and failure-interpretation boundaries. |
| Sprint 123 Day 6 QR decision | Defers rank-deficient external QR until rank, nullspace, and minimum-norm semantics are separated. |
| `tests/qr_external_dense_reference.py` | Existing Python standard-library external QR helper and output protocol. |
| `tests/test_qr_solve.c` | Existing external QR fixture owner and QR solve scenario executable. |
| `docs/maintainer_guide.md` | Maintainer evidence table that records bounded QR evidence and non-claims. |

## Decision Summary

Accepted one bounded rank-only external QR fixture:

`qr_rankdef_duplicate_5x4_rank_only`

The fixture proves only that the product QR rank agrees with a tiny
standard-library dense reference rank for one exact duplicate-column 5x4
matrix at threshold `0.0`. It does not assert least-squares residual behavior,
nullspace basis/subspace behavior, minimum-norm optimality, pseudoinverse
agreement, Q-basis orientation, economy-mode behavior, SuiteSparse behavior, or
broad QR parity.

## Accepted Fixture Contract

| Field | Decision |
| --- | --- |
| Fixture key | `qr_rankdef_duplicate_5x4_rank_only` |
| Matrix shape | 5x4 dense tall matrix |
| Structural model | Column 3 duplicates column 1 exactly. |
| Expected rank | `3` |
| Threshold | `0.0` for the product `sparse_qr_rank` comparison and Python reference elimination. |
| Reference path | `tests/qr_external_dense_reference.py` computes a standard-library Gaussian-elimination rank. |
| Product path | `tests/test_qr_solve.c` builds the matching sparse fixture and compares `sparse_qr_rank(&qr, 0.0)` and `qr.rank` to the reference rank. |
| Output protocol | `OK 1`, then expected rank as a numeric value. |
| Tolerance | None beyond exact rank equality; no solution, residual, nullspace, or basis vector is compared. |
| Windows behavior | Preserve existing external QR helper skip behavior on Windows. |
| Missing Python behavior | Preserve existing external-reference helper skip behavior. |
| Helper `ERROR` behavior | Test failure. |
| Build membership impact | None. The test is registered inside existing `test_qr_solve`; no new executable, Makefile entry, CMake entry, or CTest member is added. |

## Fixture Matrix

```text
[ 1.0,  0.0,  2.0,  0.0 ]
[ 0.0,  1.0, -1.0,  1.0 ]
[ 2.0, -1.0,  0.0, -1.0 ]
[ 1.0,  1.0,  1.0,  1.0 ]
[ 3.0,  0.0, -2.0,  0.0 ]
```

The fourth column duplicates the second column. The first three columns are
independent for this fixture, so the expected rank is `3`.

## Implemented Changes

| Surface | Change |
| --- | --- |
| `tests/qr_external_dense_reference.py` | Added `qr_rankdef_duplicate_5x4_rank_only`, a rank-fixture dispatcher, and a standard-library rank routine. |
| `tests/test_qr_solve.c` | Added fixture allow-list entry, sparse fixture construction, external rank read, product rank comparison, and existing-suite registration. |
| `docs/maintainer_guide.md` | Updated the QR trust-boundary row to include the bounded rank-only fixture while preserving rank-deficient solve/nullspace/minimum-norm non-claims. |
| Makefile | No change. |
| CMake / CTest | No change. |
| Public API / public docs | No change. |

## Deferred Rank-Adjacent Work

| Deferred Work | Future Owner | Promotion Gate |
| --- | --- | --- |
| Rank-deficient residual-only external QR evidence | Future QR solve oracle owner | Prove residual evidence adds trust beyond deterministic rank-deficient solve checks without implying nullspace or minimum-norm behavior. |
| Rank-deficient nullspace external evidence | Future QR basis/subspace owner | Define sign, basis ordering, projection/subspace metric, nullity, and null residual semantics. |
| Rank-deficient minimum-norm external evidence | Sprint 124 Days 4-5 minimum-norm owner | Define QR solve, COLAMD, SVD-pseudoinverse, fallback, refinement, optional SuiteSparse, norm, and residual policies. |
| Near-rank-deficient threshold external evidence | Future numerical-rank owner | Define threshold family, expected rank at each threshold, reference rank stability, and non-global interpretation. |
| SuiteSparse rank-deficient QR external evidence | Future corpus/platform owner | Define optional corpus, platform availability, support tier, skip behavior, and claim boundaries. |

## Validation Checklist

Day 3 touched `.c` and Python helper files, so the required code gate is:

1. `python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_rank_only`
2. `make build/test_qr_solve && ./build/test_qr_solve`
3. `make format`
4. `make lint`
5. `make test`
6. `git diff --check`
7. Focused trailing-whitespace scan over Sprint 124 files and touched QR files

The helper must emit `OK 1` followed by `3`. A different output count or rank
is a fixture protocol failure.

## Failure Diagnostics

The accepted test identifies:

- fixture key;
- reference helper status;
- expected rank;
- product `sparse_qr_rank(&qr, 0.0)`;
- stored factorization `qr.rank`;
- whether the failure is helper protocol, reference rank, product QR factor,
  rank comparison, unsupported platform, or optional-helper unavailability.

## Non-Claim Register

Day 3 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  broad external dense-library parity;
- broad QR factorization, QR solve, least-squares, or rank-deficient parity;
- global QR rank-threshold policy;
- rank-deficient residual-only behavior;
- raw nullspace basis, nullspace subspace, Q-basis, Q-sign, Q-orientation,
  economy-mode, sparse-mode, reorder, or backend parity;
- underdetermined or rank-deficient minimum-norm global optimality;
- SVD-pseudoinverse parity;
- package, ABI, platform, public API, CMake, Makefile, CI, or CTest expansion;
- performance, scalability, memory behavior, or state-of-the-art behavior.

## Validation Notes

Focused validation passed before full quality:

1. `python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_rank_only`
   emitted `OK 1` and `3`.
2. `make build/test_qr_solve && ./build/test_qr_solve` passed with 16 tests, 0
   failures, 0 skips, and 1060 assertions.

Full required quality validation passed:

1. `make format`
2. `make lint`
3. `make test`

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 1 is complete or explicitly deferred. | Complete | Bounded rank-only fixture accepted and implemented; rank-adjacent work is explicitly deferred. |
| Accepted evidence is bounded and testable. | Complete | Fixture protocol, output count, validation commands, and affected owners are defined. |
| Deferred work has clear promotion gates and owners. | Complete | See deferred rank-adjacent work table. |
