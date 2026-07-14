# Day 12 Helper Ownership Follow-Through

## Purpose

Revisit minimum-norm and Bidiagonal/Golub-Kahan helper movement after Sprint
124 added bounded QR rank, QR minimum-norm, QR economy-projector, and
partial-SVD vector-residual evidence. The goal is to decide whether any helper
can move now without hiding behavior-specific assertions, tolerances,
diagnostics, or future owners.

## Decision Summary

No helper movement is accepted on Sprint 124 Day 12.

Minimum-norm helper migration remains deferred. Bidiagonal/Golub-Kahan helper
extraction remains deferred. The branch has added new bounded evidence, but it
has not changed the key ownership constraint: helper names and call sites must
preserve the behavior being proven.

## Inputs Reviewed

| Input | Relevance |
| --- | --- |
| Sprint 123 Day 11 minimum-norm helper migration decision | Defers generic minimum-norm helper migration and defines behavior-specific naming gates. |
| Sprint 123 Day 12 Bidiagonal/Golub-Kahan helper decision | Defers GK/Bidiag extraction except through a dedicated semantic owner. |
| Sprint 124 Day 4 QR minimum-norm behavior contract | Defines owner boundaries across QR solve, COLAMD, SVD pseudoinverse, fallback, refinement, rank-deficient, zero-row, and SuiteSparse scenarios. |
| Sprint 124 Day 5 QR minimum-norm decision | Adds one bounded exact external QR minimum-norm lane while preserving broad minimum-norm non-claims. |
| `tests/test_qr_solve.c` | Focused QR solve and bounded external QR minimum-norm owner. |
| `tests/test_colamd.c` | Broad minimum-norm, COLAMD, fallback, refinement, rank-deficient, zero-row, QR-vs-pinv, and SuiteSparse submatrix owner. |
| `tests/test_svd.c` | SVD pseudoinverse, Moore-Penrose, Golub-Kahan extraction, and bidiagonal QR iteration owner. |
| `tests/test_bidiag.c` | Bidiagonal reduction, implicit Householder reconstruction, wide transpose, and bidiag lifecycle owner. |

## Minimum-Norm Ownership Inventory

| Surface | Current responsibility | Day 12 decision |
| --- | --- | --- |
| `tests/test_qr_solve.c` | Focused QR solve scenarios plus bounded external `qr_underdetermined_minnorm_2x4` solution/residual/norm proof | Keep in place. This file is the clearest owner for the accepted bounded QR solve lane. |
| `tests/qr_external_dense_reference.py` | Standard-library external reference protocol for named QR fixtures | Keep in place. It may emit fixture-specific values, but it should not own generic minimum-norm semantics. |
| `tests/test_colamd.c` | Broad minimum-norm behavior across COLAMD, fallback, rank-deficient, refinement, zero-row, QR-vs-pinv, and SuiteSparse submatrix cases | Keep in place. Moving assertions would hide behavior-specific ownership. |
| `tests/test_svd.c` | SVD pseudoinverse and Moore-Penrose minimum-norm evidence | Keep in place. SVD remains a bounded cross-check only when explicitly named. |
| `tests/test_qr_helpers.h` | QR fixture builders and measurements | Do not add minimum-norm assertion helpers now. Future fixture builders must use QR/minimum-norm-specific names. |
| `tests/test_solver_helpers.h` | External-process plumbing | Do not absorb minimum-norm behavior semantics. |

## Minimum-Norm Helper Candidate Assessment

| Candidate | Decision | Reason |
| --- | --- | --- |
| Generic `assert_minnorm` or `check_minnorm` helper | Reject | Would hide whether the owner is QR solve, COLAMD, SVD pseudoinverse, fallback, refinement, rank-deficient, or corpus behavior. |
| Move 2x4 fixture construction into a shared helper | Defer | The same shape appears in different behavior contexts; a builder would be safe only with a behavior-specific name and caller-visible tolerances. |
| Move solution, residual, and norm assertions into a shared helper | Reject | Assertion wrappers would hide the fixture's exact claim and threshold. |
| Add `tf_qr_minnorm_residual_norm` or `tf_qr_minnorm_solution_norm2` measurement helpers | Defer | Measurement-only helpers could be safe later, but Sprint 124 does not need them because the current call sites are readable and behavior-specific. |
| Share QR-vs-SVD pseudoinverse comparison helpers | Defer | Cross-solver semantics must remain explicit; SVD must not become a generic QR oracle. |
| Share SuiteSparse submatrix minimum-norm setup | Defer | Corpus availability, skip behavior, matrix shape, and support wording must remain scenario-local. |

## Bidiagonal/Golub-Kahan Ownership Inventory

| Surface | Current responsibility | Day 12 decision |
| --- | --- | --- |
| `tests/test_bidiag.c` | Bidiagonal reduction, implicit Householder reconstruction, wide transpose, single-row, diagonal, `nos4`, null, and free-zeroed behavior | Keep in place. This is the Bidiagonal reduction owner. |
| `tests/test_svd.c` `gk_reconstruction_error` | Explicit extracted `U`/`V` Golub-Kahan reconstruction for square/tall cases | Keep in place. It is not the same as full-SVD reconstruction. |
| `tests/test_svd.c` `validate_gk` and `test_gk_*` | GK reconstruction, orthogonality, and wide-transpose skip semantics | Keep in place. Wide reconstruction skip behavior must remain visible. |
| `tests/test_svd.c` `test_bidiag_svd_*` | Bidiagonal QR iteration on raw bidiagonal arrays and optional accumulators | Keep in place. These are algorithm-kernel proof owners. |
| `tests/test_svd_helpers.h` | Full-SVD reconstruction, orthogonality, pseudoinverse, and low-rank measurement helpers | Do not expand into Bidiagonal/GK internals now. |
| Potential `tests/test_bidiag_helpers.h` | Dedicated future Bidiagonal/GK helper owner | Defer until a future sprint can validate focused bidiag/SVD behavior plus full quality gates. |

## Bidiagonal/Golub-Kahan Helper Candidate Assessment

| Candidate | Decision | Reason |
| --- | --- | --- |
| Move `bidiag_reconstruction_error` to `tests/test_svd_helpers.h` | Reject | It owns implicit Householder reconstruction and wide transpose semantics, not generic SVD reconstruction. |
| Move `gk_reconstruction_error` to `tests/test_svd_helpers.h` | Reject | It assumes explicit extracted `U`/`V` and upper-bidiagonal arrays, not full-SVD result objects. |
| Extract a dedicated `tests/test_bidiag_helpers.h` now | Defer | This can be safe only with a dedicated owner, behavior-specific names, and focused validation. |
| Share square/tall/wide GK matrix builders | Defer | Builders must encode GK/Bidiag intent and leave assertions/tolerances at call sites. |
| Move bidiagonal QR iteration assertions into helper wrappers | Reject | The raw-kernel iteration tests must keep algorithm semantics visible. |
| Create generic `assert_svd_reconstructs` wrappers | Reject | Would blur full SVD, GK extraction, and Bidiagonal reduction layout/tolerance differences. |

## Helper Naming Policy

Future helper names must encode behavior ownership. Acceptable future patterns
include:

- `tf_qr_minnorm_make_split_2x4`
- `tf_qr_minnorm_residual_norm2`
- `tf_qr_minnorm_solution_norm2`
- `tf_qr_minnorm_make_rankdef_2x4`
- `tf_bidiag_reconstruction_error_from_reflectors`
- `tf_bidiag_reconstruction_error_with_transpose`
- `tf_gk_reconstruction_error_from_extracted_uv`
- `tf_bidiag_qr_reconstruct_from_accumulated_uv`

Names to avoid:

- `assert_minnorm`
- `check_minnorm`
- `minnorm_oracle`
- `assert_svd_reconstructs`
- `gk_oracle`
- `bidiag_validated`

## Maintainer Evidence Update

No maintainer-guide evidence update is made on Day 12 because no helper moved
and no new test evidence was added. The existing maintainer guide already
documents the current helper policy: keep helpers header-only unless there is
a measured reason to compile them, and use helper names that include family or
workflow intent.

## Future Promotion Gates

| Future movement | Required validation |
| --- | --- |
| Minimum-norm measurement-helper movement | `make build/test_qr_solve && ./build/test_qr_solve`, `make build/test_colamd && ./build/test_colamd`, `make build/test_svd && ./build/test_svd`, then `make format && make lint && make test`. |
| Bidiagonal/GK helper extraction | `make build/test_bidiag && ./build/test_bidiag`, `make build/test_svd && ./build/test_svd`, then `make format && make lint && make test`. |
| Python helper protocol change | Direct helper invocation, affected test executable, skip-path proof, and failure-interpretation note. |
| Makefile, CMake, or CTest membership change | Source-list inspection plus relevant CMake/CTest surface proof, including platform count implications. |

## Non-Claim Register

Day 12 does not claim:

- broad minimum-norm optimality beyond named fixtures;
- QR/SVD pseudoinverse oracle parity;
- external dense-library parity for minimum-norm, Bidiagonal, or Golub-Kahan
  behavior;
- broad SVD helper consolidation;
- package, platform, ABI, public API, CMake/CTest, performance, scalability,
  memory, or state-of-the-art behavior;
- any new test membership, Windows CTest count behavior, or helper API surface.

## Validation

Day 12 changes documentation only. Validation is limited to `git diff --check`
and a focused trailing-whitespace scan of Sprint 124 documentation files.
