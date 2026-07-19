# Sprint 127 Day 11 SuiteSparse Optional-Large Minimum-Norm Evidence Decision

## Decision

Day 11 explicitly defers additional SuiteSparse and optional-large
minimum-norm evidence.

The existing Sprint 125 `west0067.mtx` first-30-row 30 x 67 submatrix smoke
remains the only accepted SuiteSparse minimum-norm corpus evidence. Day 11
does not add a new optional-large, rank-deficient, report-only, QR-vs-SVD
corpus, or alternate-submatrix fixture because no non-duplicate candidate
satisfies the Day 10 metadata protocol.

Day 11 does not change C tests, Matrix Market fixtures, optional-data gates,
external-reference helpers, public solver wording, or maintainer claims.

## Day 10 Gate Result

| Requirement | Result |
| --- | --- |
| Matrix path, extraction rule, shape, nnz, support tier | Available for the existing `west0067` smoke; missing or unaccepted for non-duplicate candidates. |
| RHS construction and feasible vector | Available for existing `west0067` smoke via `b = A * ones`; not pinned for new candidates. |
| Expected rank/nullity or threshold/rank metadata when claimed | Missing for optional-large, rank-deficient, and alternate-submatrix SuiteSparse minimum-norm candidates. |
| Residual metric and tolerance | Existing `west0067` smoke has max residual `< 1e-8`; new candidates lack fixture-local tolerances. |
| Solution norm metric and target | Existing `west0067` smoke compares `||x_min|| <= ||ones|| + 1e-8`; new candidates lack exact or independent norm targets. |
| Runtime and support tier | Existing smoke remains default checked-in; optional-large/report-only candidates lack promotion, runtime budget, and skip-path proof. |
| Diagnostics | Existing focused run prints shape, max residual, and norm bound; new candidates need matrix key, extraction, rank/nullity, residual, norm, gate, and failure diagnostics before registration. |
| Validation commands | Focused `test_colamd` run completed; no C or header changes were made, so the full C quality gate is not required for Day 11. |

## Candidate Review

| Candidate | Disposition | Blocker |
| --- | --- | --- |
| Repeat `west0067` first 30 rows | Rejected as duplicate | Already accepted with residual and feasible-vector norm-bound assertions. |
| New `west0067` row window or wider extraction | Deferred | No independently pinned rank/nullity, residual tolerance, or norm target proving distinct trust beyond the existing smoke. |
| `steam1.mtx` submatrix | Deferred | No extraction rule, expected rank/nullity or explicit non-rank owner, residual tolerance, norm target, and focused runtime contract for a minimum-norm claim. |
| `fs_541_1.mtx` submatrix | Deferred | Optional-large candidate lacks `SPARSE_TEST_LARGE=1` present/missing skip proof, runtime budget, extraction metadata, rank/nullity, residual tolerance, and norm target. |
| `orsirr_1.mtx` submatrix | Deferred | Optional-large candidate lacks `SPARSE_TEST_LARGE=1` present/missing skip proof, runtime budget, extraction metadata, rank/nullity, residual tolerance, and norm target. |
| Report-only matrices (`bcsstk14`, `s3rmt3m3`, `Kuu`, `bloweybq`, `Pres_Poisson`, `tuma1`) | Deferred | Not eligible for default evidence without support-tier promotion and pinned rank/nullity plus residual/norm metadata. |
| Rank-deficient SuiteSparse minimum-norm corpus | Deferred | No matrix/submatrix has independent rank/nullity metadata, threshold semantics, residual target, norm rule, and failure interpretation. |
| SuiteSparse QR-vs-SVD minimum-norm corpus cross-check | Deferred | No corpus fixture has bounded QR and SVD roles, tolerances, runtime expectations, and non-oracle wording pinned for Day 11. |

## Focused Diagnostics

Day 11 ran the owner-local COLAMD/minimum-norm executable:

```text
$ make build/test_colamd && ./build/test_colamd
minnorm west0067 submatrix 30x67: maxerr=1.78e-15, ||x||=4.30 <= ||1||=8.19
Tests run:    70
Tests failed: 0
Tests skipped: 0
Assertions:   310
Time:         4.723 s
ALL TESTS PASSED
```

This run preserves the existing default checked-in SuiteSparse smoke. It does
not create independent metadata for a new optional-large, alternate-submatrix,
QR-vs-SVD corpus, or rank-deficient SuiteSparse claim.

## Future Promotion Gate

A future sprint may promote additional SuiteSparse or optional-large
minimum-norm evidence only after all of the following are available:

1. Matrix path, extraction rule, shape, nnz, and support tier.
2. RHS construction and named feasible vector or independent expected solution
   metadata.
3. Expected rank/nullity and threshold semantics when rank deficiency is part
   of the claim.
4. Residual metric, tolerance, scale rule, and failure interpretation.
5. Solution-norm target: exact expected value, named feasible-vector bound, or
   bounded QR-vs-SVD cross-check target.
6. Optional-data present/missing behavior and runtime budget for optional-large
   candidates.
7. Diagnostics that print matrix key, extraction, gate state, rank/nullity,
   residual, norm, runtime-relevant context, and skip/failure reason.
8. Focused validation for the touched owner and full
   `make format && make lint && make test` if `.c` or `.h` files change.

## Evidence Preserved

- The existing `west0067` 30 x 67 minimum-norm submatrix smoke remains the
  default checked-in SuiteSparse baseline.
- Core owner-local minimum-norm lanes in `tests/test_colamd.c` remain separate
  from SuiteSparse corpus evidence.
- The exact external `qr_underdetermined_minnorm_2x4` lane remains the small
  QR solve minimum-norm anchor.
- The exact `qr_minnorm_5x10_exact_values` lane remains a bounded larger
  underdetermined fixture, not corpus evidence.
- QR-vs-SVD pseudoinverse behavior remains a bounded small-fixture cross-check,
  not a corpus oracle.

## Non-Claims Preserved

- No new SuiteSparse minimum-norm evidence is accepted in Day 11.
- No optional-large SuiteSparse minimum-norm behavior claim.
- No rank-deficient SuiteSparse minimum-norm behavior claim.
- No broad SuiteSparse corpus, platform, optional-data, or performance claim.
- No broad QR minimum-norm optimality claim.
- No SVD pseudoinverse as a global QR oracle.
- No COLAMD, fallback, refinement, nullspace, Q-basis, economy, sparse-mode,
  reorder, backend, package, ABI, public API, CI, CMake, CTest, scalability,
  memory, state-of-the-art, LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos,
  Eigen, ARPACK, vendor-backend, dense-library, ecosystem, or external package
  parity claim.

## Validation

Focused owner-local validation passed:

```text
make build/test_colamd && ./build/test_colamd
```

Day 11 changed documentation only. Required documentation validation:

```text
git diff --check
find docs/planning/EPIC_11/SPRINT_127 -type f -name '*.md' -print0 | \
  xargs -0 awk '(/[ \t]$/){print FILENAME ":" FNR ": trailing whitespace"; bad=1} END{exit bad}'
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 6 is complete or explicitly deferred. | Complete by deferral | Additional corpus expansion is deferred with candidate-specific blockers and future-owner requirements. |
| Accepted evidence reports residual and norm metrics under pinned semantics. | Complete | No new evidence accepted; existing `west0067` smoke reports residual and norm-bound diagnostics under prior pinned semantics. |
| Broad SuiteSparse, optional-data, platform, and minimum-norm claims remain absent. | Complete | See preserved evidence and non-claims. |
