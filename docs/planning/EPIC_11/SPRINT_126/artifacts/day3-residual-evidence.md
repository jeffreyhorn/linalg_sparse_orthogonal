# Sprint 126 Day 3 Residual Evidence

## Purpose

Day 3 implements the bounded residual evidence accepted by the Day 2 trust
policy and explicitly defers the remaining compatible and wide residual
candidates. The accepted evidence adds one structural rank-deficient family
beyond Sprint 125's duplicate-column residual-only fixture without claiming
solution uniqueness, minimum-norm behavior, nullspace behavior, Q-basis
behavior, economy behavior, sparse-mode behavior, SuiteSparse behavior, or
broad QR parity.

## Day 3 Decision

Day 3 accepts exactly one fixture:

| Fixture | Disposition | Reason |
| --- | --- | --- |
| `qr_rankdef_dependent_row_4x3_residual_only` | Implemented | Adds a dependent-row rank-deficient residual-only lane with a non-zero external residual reference and low solution-selection ambiguity. |
| `qr_rankdef_duplicate_5x4_compatible_zero_residual` | Deferred | Adds little beyond completed compatible solve evidence and risks being read as solution-selection or minimum-norm evidence. |
| `qr_rankdef_dependent_row_4x3_compatible_zero_residual` | Deferred | Overlaps the accepted dependent-row incompatible residual lane and existing deterministic dependent-row null-residual checks. |
| `qr_rankdef_wide_3x5_residual_only` | Deferred | Wide residual behavior is coupled to underdetermined solution-selection, nullspace, and minimum-norm boundaries. |
| `qr_rankdef_wide_sparse_mode_residual_only` | Deferred | Requires Sprint 127 Q/economy and sparse-mode output semantics before residual-only proof value is clear. |
| SuiteSparse rank-deficient residual-only fixture | Deferred | Requires Days 8-9 corpus metadata, support tier, optional-data behavior, skip rules, and diagnostics. |

## Implemented Fixture

The accepted fixture reuses the existing dependent-row 4x3 QR helper shape:

```text
A =
  [1.0,  0.0, 1.0]
  [0.0,  1.0, 2.0]
  [1.0,  1.0, 3.0]
  [2.0, -1.0, 0.0]

b = [1.0, -2.0, 5.0, 0.0]
```

The matrix shape is rank-deficient context only. The new test does not assert
rank, nullity, nullspace vectors, projector equality, reconstruction quality,
or basis orientation.

## External Reference

`tests/qr_external_dense_reference.py` now exposes
`qr_rankdef_dependent_row_4x3_residual_only` through the existing
`column_space_residual_reference()` path.

Focused helper output:

```text
$ python3 tests/qr_external_dense_reference.py qr_rankdef_dependent_row_4x3_residual_only
OK 1
4.2840332837724997
```

The helper remains Python standard-library only. It does not introduce NumPy,
SciPy, LAPACK, BLAS, SuiteSparse, or external package dependencies.

## C Test Behavior

`tests/test_qr_solve.c` now:

1. Allows the new fixture key in the external QR reference helper allowlist.
2. Builds the sparse matrix through `tf_qr_make_dependent_row_4x3()`.
3. Solves with the existing QR solve path.
4. Compares the returned residual against the external expected residual.
5. Prints returned residual, recomputed product residual, expected residual,
   and absolute difference.

The assertion set is intentionally narrow:

- `ref[0] > 0.0`
- `fabs(residual - ref[0]) < 1e-8`

The recomputed true residual is diagnostic only.

## Proof Boundary

Day 3 proves only this bounded statement:

> For the accepted dependent-row 4x3 fixture and RHS, the QR solve returned
> residual agrees with a Python standard-library column-space residual
> reference within `1e-8`.

Day 3 does not prove:

- broad QR, rank-deficient QR, least-squares, or residual parity;
- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, or ecosystem parity;
- solution-vector equality, solution uniqueness, solution-norm optimality, or
  minimum-norm optimality;
- rank, nullity, nullspace basis, projector, subspace, or raw-basis behavior;
- QR-vs-SVD-pseudoinverse oracle behavior;
- Q-basis, economy, sparse-mode, reorder, backend, corpus, package, ABI,
  platform, performance, scalability, memory, public API, CMake, Makefile, CI,
  CTest, or state-of-the-art behavior.

## Deferred Promotion Gates

| Deferred Candidate | Future Owner | Promotion Gate |
| --- | --- | --- |
| Duplicate-column compatible zero-residual fixture | Future QR residual owner | Show a zero-residual duplicate-column fixture adds distinct trust beyond completed compatible solves and cannot imply minimum-norm or solution-selection behavior. |
| Dependent-row compatible zero-residual fixture | Future QR residual owner | Show why compatible dependent-row zero residual adds a new diagnostic after the incompatible dependent-row residual lane. |
| Wide residual-only fixture | Minimum-norm or nullspace/subspace owner | Define underdetermined solution-selection boundaries, expected residual semantics, and wording that excludes minimum-norm and nullspace claims. |
| Wide sparse-mode residual fixture | Sprint 127 Q/economy/sparse-mode owner | Define sparse-mode output semantics, Q/economy boundaries, and residual-only proof value. |
| SuiteSparse rank-deficient residual fixture | Days 8-9 SuiteSparse corpus owner | Define expected-rank metadata, support tier, optional-data behavior, skip behavior, diagnostics, and validation requirements. |

## Validation Notes

Day 3 changed C test code and the Python external-reference helper, so the
required validation is:

1. `python3 -m py_compile tests/qr_external_dense_reference.py`
2. `python3 tests/qr_external_dense_reference.py qr_rankdef_dependent_row_4x3_residual_only`
3. `make build/test_qr_solve && ./build/test_qr_solve`
4. `make format && make lint && make test`
5. `git diff --check`

Focused validation already completed:

```text
$ python3 -m py_compile tests/qr_external_dense_reference.py
$ python3 tests/qr_external_dense_reference.py qr_rankdef_dependent_row_4x3_residual_only
OK 1
4.2840332837724997

$ make build/test_qr_solve && ./build/test_qr_solve
Tests run:    19
Tests failed: 0
Tests skipped: 0
ALL TESTS PASSED
```

Full quality gate completed:

```text
$ make format && make lint && make test
All tests passed.
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 2 is complete or explicitly deferred. | Complete | One dependent-row residual-only fixture implemented; compatible, wide, sparse-mode, and SuiteSparse candidates explicitly deferred. |
| Accepted fixtures prove only documented residual behavior. | Complete | Assertion set checks only non-zero expected residual and residual agreement within `1e-8`. |
| Focused validation evidence is recorded for code changes. | Complete | Helper and focused `test_qr_solve` validation output recorded above. |
