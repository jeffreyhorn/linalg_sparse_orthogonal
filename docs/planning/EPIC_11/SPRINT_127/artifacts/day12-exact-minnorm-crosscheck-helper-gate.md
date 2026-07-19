# Sprint 127 Day 12 Exact Minimum-Norm And QR-vs-SVD Gate

## Decision

Day 12 accepts one bounded Day 13 implementation candidate:
`qr_minnorm_3x6_exact_values`.

The accepted candidate strengthens the existing owner-local
`test_minnorm_3x6` fixture in `tests/test_colamd.c` by adding exact
closed-form solution-value and norm assertions. It does not add a new external
helper, SuiteSparse corpus fixture, optional-large fixture, or QR-vs-SVD
corpus check.

Day 12 explicitly defers additional QR-vs-SVD minimum-norm cross-checks. The
existing Sprint 125 `test_minnorm_vs_pinv` lane remains the only accepted
QR-vs-SVD minimum-norm cross-check for now.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| `docs/planning/EPIC_11/SPRINT_127/PLAN.md` Day 12 | Defines exact underdetermined, QR-vs-SVD, tolerance, helper, and non-oracle deliverables. |
| Sprint 125 Day 10-12 artifacts | Define behavior-specific minimum-norm owners, the bounded QR-vs-SVD cross-check, and SuiteSparse/corpus boundaries. |
| Sprint 126 Day 12-13 artifacts | Define and implement `qr_minnorm_5x10_exact_values`, and defer additional QR-vs-SVD checks. |
| Sprint 127 Day 10-11 artifacts | Explicitly defer SuiteSparse and optional-large minimum-norm corpus expansion. |
| `tests/test_colamd.c` | Owns current owner-local minimum-norm exact, residual, COLAMD, fallback, rank-deficient, refinement, zero-row, QR-vs-SVD, and SuiteSparse smoke lanes. |
| `tests/test_qr_solve.c` | Owns exact external `qr_underdetermined_minnorm_2x4` QR solve evidence. |
| `tests/qr_external_dense_reference.py` | Owns bounded external dense QR reference fixtures; not a generic minimum-norm helper owner. |
| `tests/test_svd.c` | Owns SVD pseudoinverse behavior and bounded SVD-side validation. |
| `docs/maintainer_guide.md` | Records bounded exact minimum-norm and QR-vs-SVD evidence without broad parity claims. |

## Larger Underdetermined Candidate Table

| Candidate | Existing owner | Current evidence | Day 12 disposition |
| --- | --- | --- | --- |
| `qr_underdetermined_minnorm_2x4` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | Exact values, residual, and norm against a bounded dense reference. | Complete baseline; do not repeat. |
| `test_minnorm_3x6` | `tests/test_colamd.c` | Residual and diagnostic norm only. | Accepted for Day 13 exact-value strengthening as `qr_minnorm_3x6_exact_values`. |
| `qr_minnorm_5x10_exact_values` | `tests/test_colamd.c`, `tests/qr_external_dense_reference.py` | Exact values, exact norm, residual, and diagnostics. | Complete Sprint 126 baseline; do not repeat. |
| `test_minnorm_1xn` | `tests/test_colamd.c` | Exact all-ones values already asserted. | Complete owner-local edge case; do not repeat. |
| `test_minnorm_with_colamd` | `tests/test_colamd.c` | Exact values and norm with COLAMD option behavior. | Complete reorder-option lane; do not relabel as generic exact-value corpus evidence. |
| `test_minnorm_rank_deficient` | `tests/test_colamd.c` | Exact values and norm for a rank-deficient small fixture. | Complete rank-deficient owner-local lane; do not repeat. |
| `test_minnorm_zero_row` | `tests/test_colamd.c` | Exact values and norm for a zero-row consistent fixture. | Complete zero-row owner-local lane; do not repeat. |
| Larger synthetic underdetermined fixture | None | No checked-in fixture. | Deferred until fixture key, shape, exact values, diagnostics, and owner are pinned. |
| SuiteSparse-derived underdetermined fixture | `tests/test_colamd.c` | Existing `west0067` submatrix smoke only. | Deferred by Day 11. |

## Accepted Day 13 Candidate Contract

| Field | Value |
| --- | --- |
| Candidate key | `qr_minnorm_3x6_exact_values` |
| Owner | `tests/test_colamd.c::test_minnorm_3x6` |
| Shape | 3 x 6 |
| Matrix pattern | Row 0 has `A[0,0] = 2`, `A[0,3] = 1`; row 1 has `A[1,1] = 3`, `A[1,4] = 1`; row 2 has `A[2,2] = 1`, `A[2,5] = 2`. |
| RHS | `[3, 4, 5]` |
| Independent derivation | Each row is an independent one-constraint two-variable system `ca + db = rhs`; the minimum-norm pair is `[a, b] = rhs * [c, d] / (c^2 + d^2)`. |
| Expected solution | `[1.2, 1.2, 1.0, 0.6, 0.4, 2.0]` |
| Expected norm | `sqrt(8.4)` |
| Residual tolerance | `1e-10` per row |
| Value tolerance | `1e-10` |
| Norm tolerance | `1e-10` |
| Diagnostics | Print solution norm and max residual; no SVD, SuiteSparse, optional-large, or external parity wording. |
| Validation if implemented | `make build/test_colamd && ./build/test_colamd`, then `make format && make lint && make test` because `.c` changes are required. |

## QR-vs-SVD Cross-Check Candidate Table

| Candidate | Current status | Day 12 disposition |
| --- | --- | --- |
| Existing 2 x 4 QR-vs-SVD cross-check | Accepted in Sprint 125 as `test_minnorm_vs_pinv`. | Complete baseline; do not duplicate. |
| 3 x 6 QR-vs-SVD cross-check | Possible after exact-value promotion, but exact values provide the Day 13 trust boundary. | Deferred to avoid turning SVD pseudoinverse into a broader oracle. |
| 5 x 10 QR-vs-SVD cross-check | Possible, but Sprint 126 exact-value evidence already covers this fixture. | Deferred. |
| SuiteSparse QR-vs-SVD corpus cross-check | Deferred by Day 11. | Deferred until corpus metadata, support tier, runtime, and non-oracle wording are pinned. |
| Generic QR-vs-SVD helper movement | No accepted owner. | Deferred; behavior-specific assertions must remain visible. |

## Exact-Value And Tolerance Policy

| Quantity | Policy |
| --- | --- |
| Exact values | Accept only when a fixture-local closed-form derivation is recorded before implementation. |
| Residual | Keep residual assertions at the owner call site; use per-row absolute tolerance for small exact fixtures. |
| Norm | Compare to exact norm when available; otherwise compare to a named feasible vector or defer. |
| SVD/pseudoinverse | Use only as a named bounded cross-check, not as a global QR minimum-norm oracle. |
| Helpers | Do not add generic minimum-norm assertion helpers during Day 13. |
| SuiteSparse and optional-large | Keep corpus lanes governed by Days 10-11, not by synthetic exact-value work. |

## Helper Movement Boundary

Day 12 does not move helpers. Future helper movement is allowed only when:

1. The helper name includes the behavior owner, such as `qr_minnorm_exact_*`
   or `qr_svd_minnorm_crosscheck_*`.
2. The owner call site still names the fixture key, residual tolerance, value
   tolerance, norm tolerance, and non-oracle role.
3. QR solve, COLAMD/reorder, SVD pseudoinverse, refinement, SuiteSparse, and
   optional-large behaviors remain separate owners.
4. Focused validation covers every executable whose behavior wording is
   touched.

Generic minimum-norm, pseudoinverse, or cross-solver helpers remain deferred.

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

1. Update `test_minnorm_3x6` only.
2. Add exact expected values and exact norm assertion.
3. Preserve the existing residual check.
4. Add max residual to the diagnostic print.
5. Do not add a new helper or QR-vs-SVD comparison.
6. Run `make build/test_colamd && ./build/test_colamd`.
7. Because `.c` changes are expected, run `make format && make lint &&
   make test`.
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
find docs/planning/EPIC_11/SPRINT_127 -type f -name '*.md' -print0 | \
  xargs -0 awk '(/[ \t]$/){print FILENAME ":" FNR ": trailing whitespace"; bad=1} END{exit bad}'
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| No exact fixture proceeds without closed-form expected values. | Complete | `qr_minnorm_3x6_exact_values` has a fixture-local derivation, expected solution, exact norm, and tolerances. |
| QR-vs-SVD checks remain bounded cross-checks, not oracle claims. | Complete | Additional QR-vs-SVD checks are deferred and non-oracle wording is pinned. |
| Helper movement has behavior-specific ownership and validation gates. | Complete | Generic helper movement is deferred; future helper gates require behavior-specific names and owner-local validation. |
