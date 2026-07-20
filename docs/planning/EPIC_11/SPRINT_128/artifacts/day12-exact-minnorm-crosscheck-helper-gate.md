# Sprint 128 Day 12 Exact Minimum-Norm, Cross-Check, And Helper Gate

## Decision

Day 12 does not accept a default Day 13 implementation candidate yet.

Sprint 125-127 already accepted the bounded exact and cross-check lanes that
are currently ready:

- `qr_underdetermined_minnorm_2x4`
- `qr_minnorm_5x10_exact_values`
- `qr_minnorm_3x6_exact_values`
- `qr_minnorm_vs_svd_pinv_crosscheck`

Sprint 128 must not repeat those lanes as new evidence. Day 13 may implement
new exact minimum-norm or QR-vs-SVD work only if it satisfies the metadata and
ownership gates below before code edits. Otherwise Day 13 must explicitly
defer Project Plan Item 7.

This artifact is policy-only. It does not change C source, headers, Python
helpers, Matrix Market data, public wording, maintainer claims, or helper
ownership.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 128 Plan Day 12 | Defines QR-vs-SVD cross-check, exact underdetermined, helper movement, tolerance, and non-oracle deliverables. |
| Sprint 125 Day 10-12 artifacts | Define behavior-specific minimum-norm owners, the bounded `2 x 4` QR-vs-SVD cross-check, and generic helper deferrals. |
| Sprint 126 Day 12-13 artifacts | Define and implement `qr_minnorm_5x10_exact_values` and defer additional QR-vs-SVD checks. |
| Sprint 127 Day 12-13 artifacts | Define and implement `qr_minnorm_3x6_exact_values` and keep helper movement deferred. |
| Sprint 128 Day 10-11 artifacts | Defer additional SuiteSparse, optional-large, and corpus QR-vs-SVD minimum-norm work. |
| `tests/test_colamd.c` | Owns current owner-local minimum-norm exact, COLAMD, fallback, rank-deficient, refinement, zero-row, QR-vs-SVD, and SuiteSparse smoke lanes. |
| `tests/test_qr_solve.c` | Owns exact external `qr_underdetermined_minnorm_2x4` QR solve evidence. |
| `tests/qr_external_dense_reference.py` | Owns bounded external dense QR reference fixtures; not a generic minimum-norm helper owner. |
| `tests/test_svd.c` | Owns SVD pseudoinverse behavior and bounded SVD-side validation. |
| `docs/maintainer_guide.md` | Records bounded exact minimum-norm and QR-vs-SVD evidence without broad parity claims. |

## Exact Underdetermined Candidate Table

| Candidate | Existing owner | Current evidence | Day 12 disposition |
| --- | --- | --- | --- |
| `qr_underdetermined_minnorm_2x4` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | Exact values, residual, and norm against a bounded dense reference. | Complete baseline; do not repeat. |
| `qr_minnorm_5x10_exact_values` | `tests/test_colamd.c`, `tests/qr_external_dense_reference.py` | Exact solution values, exact norm `sqrt(11)`, residual, and diagnostics. | Complete Sprint 126 baseline; do not repeat. |
| `qr_minnorm_3x6_exact_values` | `tests/test_colamd.c`, `tests/qr_external_dense_reference.py` | Exact solution values `[1.2, 1.2, 1.0, 0.6, 0.4, 2.0]`, exact norm `sqrt(8.4)`, residual, and diagnostics. | Complete Sprint 127 baseline; do not repeat. |
| `test_minnorm_1xn` | `tests/test_colamd.c` | Exact all-ones values already asserted. | Complete owner-local edge case; do not relabel as new Sprint 128 exact evidence. |
| `test_minnorm_with_colamd` | `tests/test_colamd.c` | Exact values and norm with COLAMD option behavior. | Complete reorder-option lane; do not relabel as generic exact-value evidence. |
| `test_minnorm_rank_deficient` | `tests/test_colamd.c` | Exact values and norm for a rank-deficient small fixture. | Complete rank-deficient owner-local lane; do not repeat. |
| `test_minnorm_zero_row` | `tests/test_colamd.c` | Exact values and norm for a zero-row consistent fixture. | Complete zero-row owner-local lane; do not repeat. |
| New larger synthetic underdetermined fixture | None | No fixture key, shape, exact values, or owner contract pinned. | Deferred until closed-form metadata is available. |
| SuiteSparse-derived exact underdetermined fixture | `tests/test_colamd.c` | Existing `west0067` submatrix smoke only. | Deferred by Day 11. |

## QR-vs-SVD Cross-Check Candidate Table

| Candidate | Current status | Day 12 disposition |
| --- | --- | --- |
| Existing `2 x 4` QR-vs-SVD cross-check | Accepted in Sprint 125 as `test_minnorm_vs_pinv`. | Complete baseline; do not duplicate. |
| `3 x 6` QR-vs-SVD cross-check | Exact values already provide fixture-local trust. | Deferred to avoid broadening SVD-pseudoinverse into an oracle. |
| `5 x 10` QR-vs-SVD cross-check | Exact values already provide fixture-local trust. | Deferred. |
| SuiteSparse QR-vs-SVD corpus cross-check | Deferred by Day 11. | Deferred until corpus metadata, support tier, runtime, and non-oracle wording are pinned. |
| Generic QR-vs-SVD helper movement | No accepted owner. | Deferred; behavior-specific assertions must remain visible. |

## Exact-Value And Tolerance Policy

| Quantity | Policy |
| --- | --- |
| Fixture key | Required before acceptance; must be non-duplicate and owner-local. |
| Shape | Required before acceptance; must add trust beyond `2 x 4`, `3 x 6`, `5 x 10`, `1 x n`, COLAMD, rank-deficient, and zero-row baselines. |
| Exact values | Accept only when a fixture-local closed-form derivation is recorded before implementation. |
| Residual | Keep residual assertions at the owner call site; use per-row absolute tolerance for small exact fixtures. |
| Norm | Compare to exact norm when available; otherwise compare to a named feasible vector or defer. |
| Value tolerance | Must be fixture-local and recorded before implementation. |
| Diagnostics | Print fixture key or named owner, max residual, solution norm, and expected norm when exact values are asserted. |
| SVD/pseudoinverse | Use only as a named bounded cross-check, not as a global QR minimum-norm oracle. |
| SuiteSparse and optional-large | Keep corpus lanes governed by Days 10-11, not by synthetic exact-value work. |

## Helper Movement Decision

Day 12 does not move helpers.

Future helper movement is allowed only when all of the following are true:

1. The helper name includes the behavior owner, such as `qr_minnorm_exact_*`,
   `qr_svd_minnorm_crosscheck_*`, or a similarly specific family name.
2. The owner call site still names the fixture key, residual tolerance, value
   tolerance, norm tolerance, and non-oracle role.
3. QR solve, COLAMD/reorder, SVD pseudoinverse, refinement, SuiteSparse, and
   optional-large behaviors remain separate owners.
4. The change does not move behavior into public headers or generic helper
   APIs without a separate API-design review.
5. Focused validation covers every executable whose behavior wording changes.
6. Full `make format && make lint && make test` is required if `.c` or `.h`
   files change.

Generic minimum-norm, pseudoinverse, cross-solver, or corpus helpers remain
deferred.

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

## Day 13 Candidate Decision

Day 13 has two valid paths:

| Path | Requirements | Outcome |
| --- | --- | --- |
| Accept one bounded exact or cross-check candidate | Provide non-duplicate fixture key, closed-form expected values or bounded QR-vs-SVD role, tolerances, diagnostics, owner-local placement, helper decision, and validation plan before code edits. | Add one narrowly named owner-local test update and run focused plus full validation if code changes. |
| Explicitly defer all candidates | Show no candidate satisfies the Day 12 gate. | Preserve completed Sprint 125-127 evidence and publish future-owner promotion gates. |

The preferred Day 13 default is explicit deferral unless a non-duplicate
closed-form fixture is available before implementation.

## Non-Claims Preserved

- No new exact minimum-norm fixture is accepted in Day 12.
- No new QR-vs-SVD cross-check is accepted in Day 12.
- No SVD pseudoinverse as a global QR oracle.
- No broad QR minimum-norm optimality claim.
- No SuiteSparse, optional-large, rank-deficient corpus, platform, or
  performance claim.
- No generic helper API or helper consolidation claim.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, external package, COLAMD,
  fallback, refinement, nullspace, Q-basis, economy, sparse-mode, reorder,
  backend, package, ABI, public API, CI, CMake, CTest, scalability, memory, or
  state-of-the-art parity claim.

## Validation

Day 12 changed documentation only. Required validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_128
```

No `.c`, `.h`, Python helper, build, public API, maintainer, Matrix Market,
optional-data, or public wording files changed for Day 12, so no new code
quality gate is required.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| No cross-check can become an SVD-pseudoinverse oracle claim. | Complete | Additional QR-vs-SVD checks are deferred and non-oracle wording is pinned. |
| Exact lanes have closed-form expected values before acceptance. | Complete | Completed lanes are preserved; new lanes must provide fixture-local derivation before Day 13 implementation. |
| Helper movement has behavior-specific ownership and validation gates. | Complete | Generic helper movement is deferred; future helper movement requires specific names, visible owner call-site tolerances, and focused plus full validation. |
