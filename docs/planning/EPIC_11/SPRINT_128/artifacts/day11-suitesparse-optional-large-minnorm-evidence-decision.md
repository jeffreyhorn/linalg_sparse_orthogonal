# Sprint 128 Day 11 SuiteSparse Optional-Large Minimum-Norm Evidence Decision

## Decision

Day 11 explicitly defers additional SuiteSparse and optional-large
minimum-norm evidence.

The Day 10 gate requires a candidate to pin extraction, shape, nnz, RHS,
rank/nullity when claimed, residual metric and tolerance, norm comparison
target, support tier, skip behavior, runtime/platform expectations,
diagnostics, and validation before implementation. No non-duplicate candidate
currently satisfies that gate.

Day 11 therefore preserves the existing `west0067` first-30-row smoke as the
only accepted checked-in SuiteSparse minimum-norm baseline and does not add or
modify C tests, headers, Python helpers, Matrix Market fixtures,
optional-data gates, public wording, or maintainer claims.

## Day 10 Gate Result

| Requirement | Result |
| --- | --- |
| Matrix path, extraction rule, shape, nnz, and support tier | Available only for the existing `west0067` first-30-row smoke. Not pinned for any new candidate. |
| RHS construction and feasible-vector or expected-solution metadata | Available only for the existing `b = A * ones` `west0067` smoke. |
| Expected rank/nullity or threshold/rank metadata when claimed | Missing for rank-deficient SuiteSparse and optional-large candidates. |
| Residual metric and tolerance | Available for existing smoke; missing for non-duplicate corpus candidates. |
| Norm comparison target | Available for existing smoke through `||ones||`; missing for non-duplicate corpus candidates. |
| Support-tier, skip, runtime, and platform behavior | Defined by policy; not proven for optional-large or report-only candidates. |
| Diagnostics and validation | Focused owner validation passed for current evidence; no new code path accepted. |

## Candidate Review

| Candidate | Day 11 disposition | Reason |
| --- | --- | --- |
| Repeat `west0067` first 30 rows | Rejected as new evidence | Duplicate of the accepted Sprint 125 checked-in smoke. It adds no distinct extraction, residual, norm, or rank/nullity metric. |
| New `west0067` row window or wider extraction | Deferred | Needs extraction rule, shape, nnz, duplicate fence, residual target, norm comparison target, and rank/nullity policy if claimed. |
| `steam1.mtx` submatrix | Deferred | Default checked-in data exists, but no extraction, rank/nullity, residual tolerance, norm target, or focused runtime contract is pinned. |
| `fs_541_1.mtx` submatrix | Deferred | Optional-large lane lacks optional-data present/missing proof, runtime budget, extraction metadata, rank/nullity policy, residual/norm targets, and skip diagnostics. |
| `orsirr_1.mtx` submatrix | Deferred | Optional-large lane lacks optional-data present/missing proof, runtime budget, extraction metadata, rank/nullity policy, residual/norm targets, and skip diagnostics. |
| Report-only matrices (`bcsstk14`, `s3rmt3m3`, `Kuu`, `bloweybq`, `Pres_Poisson`, `tuma1`) | Deferred | Not eligible for default evidence without support-tier promotion and pinned extraction plus metric metadata. |
| Rank-deficient SuiteSparse minimum-norm corpus | Deferred | Requires independent rank/nullity metadata, threshold semantics, residual target, norm rule, and failure interpretation. |
| SuiteSparse QR-vs-SVD minimum-norm corpus cross-check | Deferred | Requires QR and SVD owner roles, tolerances, runtime, support tier, and non-oracle wording before implementation. |

## Focused Diagnostics

Day 11 ran the current minimum-norm owner executable:

```text
$ make build/test_colamd && ./build/test_colamd
minnorm west0067 submatrix 30x67: maxerr=1.78e-15, ||x||=4.30 <= ||1||=8.19
Tests run:    70
Tests failed: 0
Tests skipped: 0
Assertions:   317
ALL TESTS PASSED
```

These diagnostics preserve the current accepted smoke. They are not new
SuiteSparse, optional-large, rank-deficient, or QR-vs-SVD corpus evidence.

## Evidence Preserved

Existing minimum-norm evidence remains bounded:

- `qr_underdetermined_minnorm_2x4` remains the exact small QR solve anchor.
- `qr_minnorm_5x10_exact_values` remains exact-value underdetermined evidence.
- `qr_minnorm_3x6_exact_values` remains exact-value underdetermined evidence.
- `qr_minnorm_vs_svd_pinv_crosscheck` remains one bounded QR-vs-SVD
  pseudoinverse cross-check, not an oracle.
- `qr_minnorm_suitesparse_submatrix` remains one `west0067` checked-in
  submatrix smoke, not broad SuiteSparse corpus evidence.

## Optional-Data Behavior

No optional-large minimum-norm evidence is accepted on Day 11.

Future optional-large promotion must keep missing-data and numerical failure
separate:

1. The gate must be explicit, such as `SPARSE_TEST_LARGE=1` or a narrower
   QR minimum-norm opt-in.
2. A missing optional matrix may skip only before the numerical claim is active
   and only with a message naming the matrix, gate, and owner.
3. Once data is present and accepted metadata exists, load, solve, residual,
   norm, or rank disagreement must fail rather than skip.
4. Runtime expectations must be recorded before adding the path to default or
   optional CI.

## Future Promotion Gate

A future sprint may promote additional SuiteSparse or optional-large
minimum-norm evidence only after all of the following are available:

1. Matrix path, extraction rule, resulting shape, nnz, and support tier.
2. RHS construction and either a named feasible vector or independent expected
   solution metadata.
3. Expected rank/nullity or threshold/rank metadata when rank deficiency is
   part of the claim.
4. Residual metric, tolerance, and scaling rule.
5. Norm metric and comparison target.
6. Claim owner and non-claim text.
7. Diagnostics printed before assertions.
8. Skip behavior for missing optional data and platform limitations.
9. Runtime budget and default-suite eligibility.
10. Focused validation command and full `make format && make lint &&
    make test` validation if `.c` or `.h` files change.

## Non-Claims Preserved

- No new SuiteSparse minimum-norm evidence is accepted in Day 11.
- No optional-large SuiteSparse minimum-norm behavior.
- No rank-deficient SuiteSparse minimum-norm behavior.
- No broad QR minimum-norm optimality.
- No SVD pseudoinverse as a global QR oracle.
- No broad SuiteSparse corpus correctness, platform support, or performance.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, or external package parity.
- No COLAMD, fallback, refinement, nullspace, Q-basis, economy, sparse-mode,
  reorder, backend, package, ABI, public API, CI, CMake, CTest, scalability,
  memory, or state-of-the-art behavior.

## Validation Notes

Focused minimum-norm owner validation passed:

```text
make build/test_colamd && ./build/test_colamd
```

Day 11 changed documentation only. Required documentation validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_128`

No `.c`, `.h`, Python helper, build, public API, maintainer, Matrix Market,
optional-data, or public wording files changed for Day 11.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 6 is complete or explicitly deferred. | Complete by deferral | Additional SuiteSparse and optional-large minimum-norm candidates do not satisfy the Day 10 metadata gate. |
| Accepted evidence reports residual and norm behavior independently. | Complete | No new evidence is accepted; existing `west0067` smoke diagnostics remain recorded. |
| Optional-large and SuiteSparse support tiers remain visible. | Complete | Candidate review and optional-data behavior sections separate default, optional-large, and report-only paths. |
