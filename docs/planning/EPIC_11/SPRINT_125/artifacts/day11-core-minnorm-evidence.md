# Sprint 125 Day 11 Core Minimum-Norm Evidence

## Decision

Accepted one bounded core minimum-norm evidence batch in
`tests/test_colamd.c`.

The batch strengthens existing behavior-specific tests instead of adding a
generic minimum-norm helper or a new external-reference protocol. It keeps
COLAMD, fallback, rank-deficient, refinement, and zero-row behavior visible at
their current owner and adds explicit value, residual, or norm assertions where
the previous tests mostly printed diagnostics.

## Accepted Lanes

| Owner key | Test | Added evidence | Claim boundary |
| --- | --- | --- | --- |
| `qr_minnorm_colamd_behavior` | `test_minnorm_with_colamd` | Asserts the exact COLAMD-path minimum-norm solution `[0.75, 0.75, 1.5, 0.75, 0.75]` and norm `sqrt(4.5)`. | Minimum-norm solve under one named COLAMD option; no COLAMD superiority or reorder parity claim. |
| `qr_minnorm_fallback_behavior` | `test_minnorm_fallback_overdetermined` | Asserts exact fallback solution `[1, 2, 3]` in addition to residual. | Ordinary QR solve fallback for one compatible overdetermined fixture; no underdetermined optimality claim. |
| `qr_minnorm_rankdef_behavior` | `test_minnorm_rank_deficient` | Asserts expected `[0.5, 0.5, 0.5, 0.5]` and norm `1.0` for a consistent rank-1 fixture. | Fixture-local rank-deficient minimum-norm behavior; no nullspace basis or global rank-threshold claim. |
| `qr_minnorm_refinement_behavior` | `test_refine_minnorm` | Asserts post-refinement residual below `1e-10` and solution norm `sqrt(4.5)`. | Residual non-increase and bounded residual for one refinement fixture; no convergence-rate claim. |
| `qr_minnorm_zero_row_behavior` | `test_minnorm_zero_row` | Asserts zero-row residual, expected `[1, 0, 1, 0]`, and norm `sqrt(2)`. | Consistent zero-row minimum-norm fixture; no broad inconsistent-system claim. |

## Deferred Lanes

| Lane | Disposition | Reason |
| --- | --- | --- |
| Larger underdetermined shapes | Deferred | Existing `3x6`, `5x10`, and `1x5` controls remain useful, but Day 11 did not add new expected-value contracts for every shape. |
| QR-vs-SVD-pseudoinverse | Deferred to Day 12 | Cross-solver semantics require separate oracle/cross-check wording and SVD validation. |
| SuiteSparse submatrix minimum-norm | Deferred to Day 12 | Corpus evidence must apply Day 8-9 support-tier, skip, and non-claim rules. |
| Generic minimum-norm helper movement | Deferred | Assertion helpers would hide behavior ownership; measurement-only helpers require a future helper-owner decision. |

## Implemented Changes

| Surface | Change |
| --- | --- |
| `tests/test_colamd.c` | Added exact fallback assertions, exact COLAMD-path assertions, rank-deficient value/norm assertions, refinement residual/norm assertions, and zero-row value/norm assertions. |
| `tests/test_qr_solve.c` | No change; used as focused companion validation because it owns the exact external 2x4 minimum-norm lane. |
| `tests/qr_external_dense_reference.py` | No change; Day 11 does not add external helper output. |
| `docs/maintainer_guide.md` | No Day 11 update; Day 13 owns final evidence-table and claim-gate refresh. |

## Diagnostics And Tolerances

| Lane | Diagnostics retained | Tolerance |
| --- | --- | --- |
| COLAMD | solution norm print | exact values and norm within `1e-10` |
| Fallback | solution values and max residual print | exact values within `1e-10`; max residual below `1e-10` |
| Rank-deficient | solution norm print | exact values and norm within `1e-8`; residual rows within `1e-8` |
| Refinement | before/after residual print | residual non-increase, residual below `1e-10`, norm within `1e-10` |
| Zero row | active-row residual and norm print | exact values and norm within `1e-10` |

## Non-Claims Preserved

- No broad QR minimum-norm parity.
- No global minimum-norm optimality beyond named fixtures.
- No SVD pseudoinverse as a global QR oracle.
- No COLAMD, reorder, fallback, refinement, rank-deficient, zero-row, or
  SuiteSparse superiority claim.
- No broad SuiteSparse corpus support or platform parity.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK, dense-library,
  backend, package, ABI, public API, CMake, CTest, performance, scalability,
  memory, or state-of-the-art claim.

## Validation

Focused validation passed:

```text
make build/test_colamd && ./build/test_colamd
make build/test_qr_solve && ./build/test_qr_solve
```

Focused results:

| Command | Result |
| --- | --- |
| `./build/test_colamd` | 70 tests, 0 failures, 0 skips, 282 assertions |
| `./build/test_qr_solve` | 18 tests, 0 failures, 0 skips, 1089 assertions |

Because Day 11 changed a C file, full required validation passed:

```text
make format && make lint && make test
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_125 docs/maintainer_guide.md tests/qr_external_dense_reference.py tests/test_qr.c tests/test_qr_solve.c tests/test_colamd.c tests/test_svd.c
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| COLAMD, fallback, rank-deficient, and refinement lanes have accepted or deferred dispositions. | Complete | Accepted core lanes and deferred Day 12 oracle/corpus lanes. |
| Accepted evidence remains behavior-specific. | Complete | Assertions stay in owner-local tests and no generic helper was added. |
| Focused validation matches touched surfaces. | Complete | `test_colamd` and companion `test_qr_solve` focused runs passed. |
| Full C-file quality gate passed. | Complete | `make format && make lint && make test`, `git diff --check`, and the focused whitespace scan passed. |
