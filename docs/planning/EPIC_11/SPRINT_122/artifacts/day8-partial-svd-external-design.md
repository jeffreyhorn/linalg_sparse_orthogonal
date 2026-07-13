# Sprint 122 Day 8 Partial-SVD External Design

## Purpose

Day 8 completes Sprint 122 Item 4 by deciding whether partial-SVD external
comparison is viable in this sprint. The accepted scope is a bounded top-k
singular-value lane only; vector, subspace, convergence-budget, SuiteSparse, and
low-rank optimality external claims remain unsupported.

## Decision

Accepted and implemented one bounded partial-SVD external top-k value lane:

`partial_svd_diag6_k2`

The lane validates the top two singular values of a fixed diagonal 6x6 fixture
against the existing Python standard-library dense reference helper. It stays
inside existing `test_svd` membership.

## Fixture and Reference Protocol

| Field | Decision |
| --- | --- |
| Fixture key | `partial_svd_diag6_k2` |
| Matrix shape | 6x6 diagonal |
| Singular values | `{9, 6, 3, 1, 0.5, 0.25}` |
| Partial rank | `k = 2` |
| Compared quantity | Top two singular values only |
| Product path | `sparse_svd_partial(A, 2, NULL, &partial)` |
| Reference path | `tests/svd_external_dense_reference.py` computes full singular values, sorts descending, and emits the top two values for this fixture key. |
| Output protocol | `OK 2`, then `sigma_0`, `sigma_1` |
| Value tolerance | Max absolute top-k singular-value difference below `1e-8` |
| Vector/subspace handling | Not compared; returned `U` and `Vt` are asserted null for default options. |
| Convergence budget | Default partial-SVD options only; no convergence-budget parity claim. |
| Optional dependency policy | Python standard library only; no NumPy, SciPy, LAPACK, BLAS, or external package dependency. |
| Windows behavior | Explicit skip, matching existing external-reference lane policy. |
| Build membership impact | None; the test remains inside existing `test_svd`. |

This fixture is intentionally simple. Its value is protocol and top-k external
boundary proof, not a claim that diagonal fixtures alone establish broad
partial-SVD parity.

## Implemented Surfaces

| Surface | Change |
| --- | --- |
| `tests/svd_external_dense_reference.py` | Added `build_partial_svd_diag6_k2` and top-two output for the selected fixture key. |
| `tests/test_svd.c` | Allowed the partial-SVD fixture key and registered the partial-SVD external test in existing `test_svd`. |
| `tests/test_svd_partial_helpers.h` | Added `test_partial_svd_external_dense_reference_diag6_k2`. |
| Makefile | No change. |
| CMake / CTest | No change. |
| Production source | No change. |
| Public docs / API | No change. |

## Ordering, Tolerance, and Diagnostics

| Topic | Decision |
| --- | --- |
| Ordering | Reference and product top-k singular values must be descending. The selected fixture has separated values, so no tie handling is exercised. |
| Tolerance | `1e-8` max absolute difference across the two values. |
| Degenerate spectra | Not exercised; repeated and clustered spectra remain deferred. |
| Failure interpretation | Failure means reference read failure, partial-SVD compute failure, top-k value mismatch, or unsupported optional helper/platform. |
| Diagnostic output | Test prints `external partial-SVD dense ref diag6_k2: max |sigma-sigma_ref| = ...`. |

Focused output included:

```text
external partial-SVD dense ref diag6_k2: max |sigma-sigma_ref| = 1.776e-15
```

Focused `test_svd` result:

- 106 tests
- 0 failures
- 0 skips
- 1729 assertions

## Deferred or Rejected Day 7 Candidates

| Candidate | Disposition | Reason |
| --- | --- | --- |
| `partial_svd_rect_lowrank_6x4_k2_external_sigma` | Deferred | Rectangular top-k evidence overlaps low-rank reconstruction and tail-error semantics. |
| `partial_svd_rankdef_duplicate_k2_external_sigma` | Deferred | Rank threshold and zero-tail semantics should wait for a rank-threshold external owner. |
| `partial_svd_vectors_external_av_residual` | Deferred | Value reference plus vector residual is useful but still not vector parity; needs explicit vector owner. |
| `partial_svd_subspace_external_projection` | Deferred | Requires basis-invariant projector or principal-angle semantics. |
| `partial_svd_suite_sparse_external_values` | Rejected for Sprint 122 | Optional corpus fixtures would broaden runtime, platform, and SuiteSparse interpretation. |
| `partial_svd_convergence_budget_external` | Deferred | Needs deterministic budget, residual target, and nonconvergence diagnostics. |

## Non-Claim Register

This Day 8 lane does not claim:

- partial-SVD external parity beyond one bounded top-k value fixture;
- singular-vector external parity;
- subspace external parity;
- convergence-budget parity;
- repeated-spectrum or clustered-spectrum behavior;
- SuiteSparse, LAPACK, SciPy, NumPy, PETSc, Trilinos, Eigen, or broad external
  dense-library parity;
- arbitrary low-rank or pseudoinverse optimality;
- performance, scalability, package, platform, ABI, public API, or
  state-of-the-art behavior.

## Rollback Path

If validation fails:

1. Remove `build_partial_svd_diag6_k2` and top-k output handling from
   `tests/svd_external_dense_reference.py`.
2. Remove the fixture-key allowance from `tests/test_svd.c`.
3. Remove `test_partial_svd_external_dense_reference_diag6_k2` from
   `tests/test_svd_partial_helpers.h`.
4. Remove the test registration from `tests/test_svd.c`.
5. Re-run `make format && make lint && make test`.
6. Record the failed lane and reason in this artifact and the Sprint 122
   residual queue.

## Validation Plan

Because `.h` and `.c` files changed, the branch-level validation gate is:

1. `python3 tests/svd_external_dense_reference.py partial_svd_diag6_k2`
2. `make format`
3. `make build/test_svd && ./build/test_svd`
4. `make lint`
5. `make test`
6. `git diff --check`
7. Focused trailing-whitespace scan over Sprint 122 docs and touched files

## Validation Results

| Command | Result |
| --- | --- |
| `python3 tests/svd_external_dense_reference.py partial_svd_diag6_k2` | Passed; emitted `[9, 6]`. |
| `make format` | Passed. |
| `make build/test_svd && ./build/test_svd` | Passed: 106 tests, 0 failures, 0 skips, 1729 assertions. |
| `make lint` | Passed. |
| `make test` | Passed. |

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 4 is complete. | Complete | One bounded partial-SVD top-k external lane was implemented and other candidate classes are rejected or deferred. |
| Partial-SVD design does not depend on unresolved QR or SVD decisions. | Complete | The lane uses the existing SVD reference helper and does not depend on Day 6 QR work. |
| Every unsupported partial-SVD claim remains explicitly fenced. | Complete | See non-claim register and deferred/rejected candidate table. |
