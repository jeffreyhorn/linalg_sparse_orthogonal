# Sprint 126 Day 12 Underdetermined And QR-vs-SVD Gate

## Decision

Day 12 accepts one bounded Day 13 implementation candidate:
`qr_minnorm_5x10_exact_values`.

The accepted candidate strengthens the existing owner-local
`test_minnorm_5x10` fixture in `tests/test_colamd.c` by adding exact
closed-form solution-value and norm assertions. It does not add a new external
helper, SuiteSparse corpus fixture, or QR-vs-SVD corpus check.

Day 12 explicitly defers additional QR-vs-SVD minimum-norm cross-checks. The
existing Sprint 125 `test_minnorm_vs_pinv` lane remains the only accepted
QR-vs-SVD minimum-norm cross-check for now.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_126/PLAN.md` Day 12 | Defines the underdetermined and QR-vs-SVD gate deliverables. |
| `docs/planning/EPIC_11/SPRINT_125/artifacts/day10-minnorm-behavior-owner-map.md` | Defines behavior-specific owners and rejects generic minimum-norm helper movement. |
| `docs/planning/EPIC_11/SPRINT_125/artifacts/day11-core-minnorm-evidence.md` | Records accepted COLAMD, fallback, rank-deficient, refinement, and zero-row minimum-norm evidence. |
| `docs/planning/EPIC_11/SPRINT_125/artifacts/day12-oracle-corpus-minnorm-decision.md` | Records the bounded QR-vs-SVD cross-check and SuiteSparse submatrix smoke. |
| `docs/planning/EPIC_11/SPRINT_126/artifacts/day10-suitesparse-minnorm-corpus-gate.md` | Separates SuiteSparse minimum-norm corpus evidence from owner-local exact-value lanes. |
| `docs/planning/EPIC_11/SPRINT_126/artifacts/day11-suitesparse-minnorm-evidence-decision.md` | Defers additional SuiteSparse minimum-norm corpus evidence. |
| `tests/test_colamd.c` | Owns current owner-local minimum-norm exact, residual, COLAMD, fallback, rank-deficient, refinement, zero-row, QR-vs-SVD, and SuiteSparse smoke lanes. |
| `tests/test_qr_solve.c` | Owns exact external `qr_underdetermined_minnorm_2x4` QR solve evidence. |
| `tests/test_svd.c` | Owns SVD pseudoinverse behavior and bounded SVD-side validation. |

## Larger Underdetermined Candidate Table

| Candidate | Existing owner | Current evidence | Day 12 disposition |
| --- | --- | --- | --- |
| `qr_underdetermined_minnorm_2x4` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | Exact values, residual, and norm against standard-library reference. | Complete baseline; do not repeat. |
| `test_minnorm_3x6` | `tests/test_colamd.c` | Residual and diagnostic norm only. | Deferred; fixture has row-specific coefficient pairs and needs a separate exact-value table before promotion. |
| `test_minnorm_5x10` | `tests/test_colamd.c` | Residual and diagnostic norm only. | Accepted for Day 13 exact-value strengthening. |
| `test_minnorm_1xn` | `tests/test_colamd.c` | Exact all-ones values already asserted. | Complete owner-local edge case; do not repeat. |
| Larger synthetic underdetermined fixture | None | No checked-in fixture. | Deferred until fixture key, shape, exact values, and diagnostics are pinned. |
| SuiteSparse-derived underdetermined fixture | `tests/test_colamd.c` | Existing `west0067` submatrix smoke only. | Deferred by Day 11. |

## Accepted Day 13 Candidate Contract

| Field | Value |
| --- | --- |
| Candidate key | `qr_minnorm_5x10_exact_values` |
| Owner | `tests/test_colamd.c::test_minnorm_5x10` |
| Shape | 5 x 10 |
| Matrix pattern | For each row `i`, nonzeros `A[i, i] = 2` and `A[i, i + 5] = 1`. |
| RHS | `[1, 2, 3, 4, 5]` |
| Independent derivation | Each row is an independent one-constraint two-variable system `2a + b = rhs_i`; the minimum-norm solution is `[a, b] = [2*rhs_i/5, rhs_i/5]`. |
| Expected solution | `[0.4, 0.8, 1.2, 1.6, 2.0, 0.2, 0.4, 0.6, 0.8, 1.0]` |
| Expected norm | `sqrt(11)` |
| Residual tolerance | `1e-10` per row |
| Value tolerance | `1e-10` |
| Norm tolerance | `1e-10` |
| Diagnostics | Print solution norm and max residual; no SVD or SuiteSparse wording. |
| Validation if implemented | `make build/test_colamd && ./build/test_colamd`, then `make format && make lint && make test` because `.c` changes are required. |

## QR-vs-SVD Cross-Check Candidate Table

| Candidate | Current status | Day 12 disposition |
| --- | --- | --- |
| Existing 2x4 QR-vs-SVD cross-check | Accepted in Sprint 125 as `test_minnorm_vs_pinv`. | Complete baseline; do not duplicate. |
| 5x10 QR-vs-SVD cross-check | Possible but not necessary for exact-value strengthening. | Deferred to avoid turning SVD pseudoinverse into a broader oracle. |
| 3x6 QR-vs-SVD cross-check | Possible but lacks exact-value promotion metadata today. | Deferred. |
| SuiteSparse QR-vs-SVD corpus cross-check | Deferred by Day 11. | Deferred until corpus metadata, support tier, and non-oracle wording are pinned. |
| Generic QR-vs-SVD helper movement | No accepted owner. | Deferred; behavior-specific assertions must remain visible. |

## Exact-Value Ownership And Tolerance Policy

| Quantity | Policy |
| --- | --- |
| Exact values | Accept only when a fixture-local closed-form derivation is recorded before implementation. |
| Residual | Keep residual assertions at the owner call site; use per-row absolute tolerance for small exact fixtures. |
| Norm | Compare to exact norm when available; otherwise compare to a named feasible vector or defer. |
| SVD/pseudoinverse | Use only as a named bounded cross-check, not as a global QR minimum-norm oracle. |
| Helpers | Do not add generic minimum-norm assertion helpers during Day 13. |
| SuiteSparse | Keep corpus lanes governed by Days 10-11, not by the synthetic exact-value lane. |

## Non-Oracle Wording

Any QR-vs-SVD statement must use this style:

> For this named fixture and RHS, the QR minimum-norm solution and the product
> SVD pseudoinverse solution agree within the fixture-local tolerance.

It must not say or imply:

- SVD is the global oracle for QR minimum-norm behavior;
- QR and SVD have broad dense-library parity;
- the cross-check proves SuiteSparse, optional-large, performance, platform,
  backend, package, or public API behavior;
- the fixture proves all underdetermined or rank-deficient minimum-norm cases.

## Day 13 Implementation Checklist

1. Update `test_minnorm_5x10` only.
2. Add exact expected values and exact norm assertion.
3. Preserve the existing residual check.
4. Add max residual to the diagnostic print if useful.
5. Do not add a new helper or QR-vs-SVD comparison.
6. Run `make build/test_colamd && ./build/test_colamd`.
7. Because `.c` changes are expected, run `make format && make lint && make test`.
8. Record the result in the Day 13 artifact.

## Non-Claims Preserved

- No new QR-vs-SVD cross-check is accepted in Day 12.
- No SVD pseudoinverse as a global QR oracle.
- No broad QR minimum-norm optimality claim.
- No SuiteSparse, optional-large, rank-deficient corpus, platform, or
  performance claim.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, external package, COLAMD,
  fallback, refinement, nullspace, Q-basis, economy, sparse-mode, reorder,
  backend, package, ABI, public API, CI, CMake, CTest, scalability, memory, or
  state-of-the-art parity claim.

## Validation

Day 12 changed documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_126
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Items 6 and 7 have final implementation candidates or deferrals. | Complete | Day 13 candidate accepted for larger underdetermined exact values; additional QR-vs-SVD cross-checks deferred. |
| QR-vs-SVD checks remain bounded cross-checks, not SVD oracle claims. | Complete | Existing 2x4 cross-check remains baseline and non-oracle wording is pinned. |
| Exact-value contracts are assigned only to stable fixture shapes. | Complete | 5x10 exact values are accepted because each row is an independent two-variable closed-form system. |
