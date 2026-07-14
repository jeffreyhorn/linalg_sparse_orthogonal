# Sprint 123 Day 6 QR Compatible and Rank-Deficient Evidence Decision

## Purpose

Day 6 decides the compatible and rank-deficient QR external evidence lanes from
the Day 5 requirements. The decision must either define bounded fixture
protocols for Day 8 implementation or explicitly defer lanes with future
owners and promotion gates.

This is a decision artifact only. No C source, header, Python helper, build,
CMake, CTest, workflow, public API, or public wording changes are made by Day
6.

## Inputs Reviewed

| Input | Decision Use |
| --- | --- |
| Sprint 123 Plan Day 6 | Requires compatible and rank-deficient evidence decisions, fixture protocol for accepted lanes, affected owners, promotion gates, and validation plan. |
| Sprint 123 Day 5 QR requirements | Defines candidate classes, basis rules, tolerance policy, skip behavior, and duplicate fences. |
| Sprint 122 Day 6 QR lane design | Provides completed `qr_overdetermined_incompatible_4x2` external lane and duplicate fence. |
| `tests/qr_external_dense_reference.py` | Current Python standard-library QR reference helper with one incompatible 4x2 fixture. |
| `tests/test_qr_solve.c` | Current QR solve scenario owner and external QR fixture reader. |
| `tests/test_qr.c` | Current QR rank, nullspace, Q basis, economy, sparse-mode, and reconstruction owner. |

## Decision Summary

Accepted one bounded compatible tall least-squares external fixture for Day 8
implementation:

`qr_overdetermined_compatible_5x3`

Deferred rank-deficient QR external least-squares evidence. The rank-deficient
lane is valuable, but it must not be implemented as a residual-only fixture
until the future owner explicitly separates rank threshold, nullspace,
minimum-norm, and reference-solver behavior.

## Duplicate Fence

`qr_overdetermined_incompatible_4x2` remains complete and must not be reopened
by Day 8. It already proves one bounded incompatible tall least-squares lane
with external solution and residual-norm evidence.

The accepted Day 6 compatible fixture must add only a different behavior class:
a full-column-rank compatible tall system with a near-zero residual. It must
not claim broader least-squares parity or direct-solver parity.

## Accepted Compatible Fixture Contract

| Field | Decision |
| --- | --- |
| Fixture key | `qr_overdetermined_compatible_5x3` |
| Matrix shape | 5x3 dense tall full-column-rank matrix |
| Matrix | See proposed matrix below. |
| RHS construction | `b = A * [1.0, -2.0, 0.5]` |
| Expected solution | `[1.0, -2.0, 0.5]` |
| Expected residual norm | `0.0`, with comparison tolerance below `1e-8` |
| Reference path | Extend `tests/qr_external_dense_reference.py` with a small fixed fixture and standard-library dense least-squares reference. |
| Product path | Extend `tests/test_qr_solve.c` with a matching sparse fixture and full QR solve comparison. |
| Output protocol | `OK 4`, then `x0`, `x1`, `x2`, and residual norm. |
| Solution tolerance | Max absolute solution difference below `1e-8`. |
| Residual tolerance | Absolute residual-norm difference below `1e-8`; product residual must also remain near zero. |
| Dependency policy | Python standard library only; no NumPy, SciPy, LAPACK, BLAS, SuiteSparse, package, or external-data dependency. |
| Windows behavior | Preserve explicit Windows skip for external QR helper lanes. |
| Missing Python behavior | Preserve existing external-reference helper skip behavior. |
| Helper `ERROR` behavior | Test failure. |
| Build membership impact | None expected. The test should remain inside existing `test_qr_solve`; no new Makefile, CMake, or CTest member should be added. |

## Proposed Compatible Matrix

Day 8 should use this matrix unless implementation discovers a concrete
conditioning or reference-protocol problem:

```text
[ 1.0,  0.0,  2.0 ]
[ 0.0,  1.0, -1.0 ]
[ 2.0, -1.0,  0.0 ]
[ 1.0,  1.0,  1.0 ]
[ 3.0,  0.0, -2.0 ]
```

With `x = [1.0, -2.0, 0.5]`, the right-hand side is:

```text
[ 2.0, -2.5, 4.0, -0.5, 2.0 ]
```

The fixture intentionally proves only compatible tall least-squares solution
and near-zero residual behavior for this small fixed matrix.

## Affected Surface Matrix for Day 8

| Surface | Day 8 Action |
| --- | --- |
| `tests/qr_external_dense_reference.py` | Add `build_qr_overdetermined_compatible_5x3`, fixture-key dispatch, and a small dense solve path capable of the 3-value solution output. |
| `tests/test_qr_solve.c` | Add fixture-key allow-list entry, test fixture construction, external reference read for four values, solution/residual comparison, and existing-suite registration. |
| `tests/test_qr.c` | No change expected. Rank, nullspace, Q basis, economy, and sparse-mode evidence stay there. |
| Makefile | No change expected. |
| CMake / CTest | No change expected. |
| Public docs / API | No change expected. |

## Rank-Deficient QR Decision

Rank-deficient external QR least-squares is deferred.

| Candidate | Disposition | Future Owner | Promotion Gate |
| --- | --- | --- | --- |
| `qr_rankdef_duplicate_5x4_ls` | Deferred | Future QR rank-deficient/minimum-norm oracle owner | Define whether the fixture asserts residual-only behavior, rank threshold, nullspace, minimum-norm solution, or pseudoinverse agreement before implementing. |
| rank-deficient external residual-only fixture | Deferred | Future QR solve owner | Prove that residual-only evidence does not imply minimum-norm or global rank policy. |
| rank-deficient external minimum-norm fixture | Deferred | Future minimum-norm helper owner | Wait for Day 7/Day 11 ownership decisions so QR, COLAMD, SVD-pseudoinverse, refinement, fallback, and SuiteSparse owners remain visible. |

The current Python QR helper solves a tiny full-rank normal-equation system.
Using that pattern for a rank-deficient fixture would either fail on singular
normal equations or silently require a pseudoinverse/minimum-norm policy. That
policy belongs to a separate owner, not a Day 6 compatible/rank-deficient
residual decision.

## Day 8 Validation Checklist

If Day 8 implements the accepted compatible fixture, run:

1. `python3 tests/qr_external_dense_reference.py qr_overdetermined_compatible_5x3`
2. `make format`
3. `make build/test_qr_solve && ./build/test_qr_solve`
4. `make lint`
5. `make test`
6. `git diff --check`
7. Focused trailing-whitespace scan over Sprint 123 docs and touched files

The helper check must emit `OK 4`. A different output count is a fixture
protocol failure.

## Failure Diagnostics Required for Day 8

Any accepted implementation must identify:

- fixture key;
- reference helper status;
- product QR factor/solve status;
- solution max difference;
- residual norm difference;
- whether the failure is reference generation, QR solve, output-count
  protocol, solution mismatch, residual mismatch, unsupported platform, or
  optional-helper unavailability.

## Non-Claim Register

Day 6 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, or broad
  external dense-library parity;
- broad QR factorization or least-squares parity;
- direct-solver parity;
- rank-deficient QR external parity;
- underdetermined or minimum-norm global optimality;
- Q-basis, Q-sign, Q-orientation, economy-mode, sparse-mode, reorder, or
  backend parity;
- package, ABI, platform, public API, CMake, Makefile, CI, or CTest expansion;
- performance, scalability, memory behavior, or state-of-the-art behavior.

## Validation Notes

Day 6 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_123`

The branch already contains Day 4 `.c` and Python helper changes; Day 4 ran
the full `make format && make lint && make test` gate after those changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Compatible and rank-deficient QR lanes are accepted or explicitly deferred. | Complete | Compatible `qr_overdetermined_compatible_5x3` accepted; rank-deficient external QR deferred with owners and gates. |
| Basis/tolerance rules are visible. | Complete | Accepted fixture compares solution/residual only; rank/Q/minimum-norm basis-sensitive evidence remains separate. |
| No completed Sprint 121 or Sprint 122 QR work is duplicated. | Complete | `qr_overdetermined_incompatible_4x2` is fenced; deterministic rank and compatible fixtures are not relabeled as external parity. |
