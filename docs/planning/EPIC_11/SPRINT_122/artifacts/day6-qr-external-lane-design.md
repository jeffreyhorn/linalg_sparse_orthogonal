# Sprint 122 Day 6 QR External Lane Design

## Purpose

Day 6 completes Sprint 122 Item 3 by selecting and implementing one bounded QR
external dense-reference lane. The selected lane validates a small incompatible
overdetermined least-squares fixture without claiming broad QR, LAPACK, SciPy,
NumPy, direct-solver, minimum-norm, rank-deficient, or performance parity.

## Decision

Accepted and implemented one bounded QR external least-squares lane:

`qr_overdetermined_incompatible_4x2`

The lane stays inside existing `test_qr_solve` test membership.

## Fixture and Reference Protocol

| Field | Decision |
| --- | --- |
| Fixture key | `qr_overdetermined_incompatible_4x2` |
| Matrix shape | 4x2 dense tall full-column-rank matrix |
| Matrix | `[[1, 0], [0, 1], [1, 1], [2, -1]]` |
| RHS construction | `A * [2, -1] + [-1, -1, 1, 0]`, where the added vector is orthogonal to the columns of `A` |
| Expected solution | `[2, -1]` |
| Expected residual norm | `sqrt(3)` |
| Reference path | `tests/qr_external_dense_reference.py` computes `A^T A`, `A^T b`, solves the 2x2 normal-equation system, and emits solution plus residual norm. |
| Product path | `tests/test_qr_solve.c` factors the fixture with `sparse_qr_factor` and solves with `sparse_qr_solve`. |
| Output protocol | `OK 3`, then `x0`, `x1`, and residual norm. |
| Solution tolerance | Max absolute solution difference below `1e-8`. |
| Residual tolerance | Absolute residual-norm difference below `1e-8`. |
| Optional dependency policy | Python standard library only; no NumPy, SciPy, LAPACK, BLAS, or external package dependency. |
| Windows behavior | Explicit skip, matching the existing external-reference lane policy. |
| Missing Python behavior | Existing external-reference helper skip. |
| Helper `ERROR` behavior | Test failure. |

The Python reference uses normal equations only for this tiny, fixed,
well-conditioned fixture. That is a bounded reference implementation, not a
recommendation or claim about production least-squares algorithms.

## Affected Surfaces

| Surface | Change |
| --- | --- |
| `tests/qr_external_dense_reference.py` | New Python standard-library helper for the selected QR fixture. |
| `tests/test_qr_solve.c` | Added external-reference helper inclusion, reference reader, one test case, and existing-executable registration. |
| Makefile | No change. |
| CMake / CTest | No change. |
| Production source | No change. |
| Public docs / API | No change. |

## Diagnostics and Failure Semantics

| Failure | Expected Interpretation |
| --- | --- |
| Unknown fixture key | Reference-helper contract error. |
| Missing `python3` | Optional external-reference skip. |
| Python helper `ERROR` | Reference generation failure and test failure. |
| QR factor or solve error | Product QR solve failure. |
| Solution diff >= `1e-8` | QR least-squares solution mismatch against bounded reference. |
| Residual diff >= `1e-8` | QR reported residual mismatch against bounded reference residual norm. |
| Windows execution | Explicit skip, not product evidence. |

Focused output included:

```text
external QR dense ref overdetermined_incompatible_4x2: solution diff = 4.441e-16, residual diff = 2.220e-16
```

Focused `test_qr_solve` result:

- 14 tests
- 0 failures
- 0 skips
- 1025 assertions

## Rejected or Deferred Day 5 Candidates

| Candidate | Disposition | Reason |
| --- | --- | --- |
| `qr_overdetermined_compatible_4x2_external_solve` | Deferred | Generated-RHS compatible solve coverage already exists; this would add less evidence than the incompatible residual lane. |
| `qr_square_3x3_external_solve` | Rejected | Square QR solve already has deterministic and QR-vs-LU coverage and could imply direct-solver parity. |
| `qr_rankdef_duplicate_5x4_external_ls` | Deferred | Rank-deficient external least-squares overlaps rank threshold and minimum-norm ownership. |
| `qr_underdetermined_minnorm_2x4_external` | Deferred | Minimum-norm ownership is handled by later helper-boundary work. |
| `qr_q_factor_external_basis_check` | Rejected | Q-basis comparisons require sign, basis, and economy/full semantics not designed here. |
| `qr_suite_sparse_external_ls` | Rejected | Optional corpus fixtures would broaden runtime, platform, and external-corpus claims. |

## Non-Claim Register

This Day 6 lane does not claim:

- LAPACK, SciPy, NumPy, SuiteSparse, PETSc, Trilinos, Eigen, or broad external
  dense-library parity;
- direct-solver or broad least-squares parity;
- rank-deficient QR external parity;
- underdetermined or minimum-norm global optimality;
- Q-basis, Q-sign, economy-mode, sparse-mode, or reorder parity;
- performance, scalability, package, platform, ABI, public API, or
  state-of-the-art behavior.

## Rollback Path

If validation fails:

1. Remove `tests/qr_external_dense_reference.py`.
2. Remove the external-reference include, reader, test, and registration from
   `tests/test_qr_solve.c`.
3. Re-run `make format && make lint && make test`.
4. Record the failed lane and reason in this artifact and the Sprint 122
   residual queue.

## Validation Plan

Because `.c` changed, the branch-level validation gate is:

1. `python3 tests/qr_external_dense_reference.py qr_overdetermined_incompatible_4x2`
2. `make format`
3. `make build/test_qr_solve && ./build/test_qr_solve`
4. `make lint`
5. `make test`
6. `git diff --check`
7. Focused trailing-whitespace scan over Sprint 122 docs and touched files

## Validation Results

| Command | Result |
| --- | --- |
| `python3 tests/qr_external_dense_reference.py qr_overdetermined_incompatible_4x2` | Passed; emitted `[2, -1, sqrt(3)]`. |
| `make format` | Passed. |
| `make build/test_qr_solve && ./build/test_qr_solve` | Passed: 14 tests, 0 failures, 0 skips, 1025 assertions. |
| `make lint` | Passed. |
| `make test` | Passed. |

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 3 is complete. | Complete | One bounded QR external lane was implemented and other candidates were rejected or deferred. |
| QR fixture size, tolerance, skip behavior, and failure semantics are explicit. | Complete | See fixture protocol and diagnostics tables. |
| No implementation is implied without a validation and support boundary. | Complete | Implementation scope, validation plan, rollback path, and non-claims are recorded. |
