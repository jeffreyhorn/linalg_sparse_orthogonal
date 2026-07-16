# Sprint 126 Day 9 SuiteSparse Rank-Deficient QR Evidence Decision

## Decision

Day 9 explicitly defers SuiteSparse rank-deficient QR corpus evidence.

No checked-in SuiteSparse matrix satisfies the Day 8 metadata protocol for a
rank-deficient QR fixture. The matrices already exercised by QR remain
full-rank controls, and the other checked-in matrices lack independent
expected-rank metadata, threshold semantics, support-tier promotion, and
focused QR diagnostics needed before registration.

Day 9 therefore does not add or modify C tests, Matrix Market fixtures,
external-reference helpers, optional-data gates, or public/maintainer claims.

## Day 8 Gate Result

| Requirement | Result |
| --- | --- |
| Matrix path, shape, nnz, and support tier | Available for checked-in corpus. |
| Claim type | No rank-deficient QR claim accepted. |
| Expected rank/nullity or threshold/rank pairs | Missing for all plausible rank-deficient candidates. |
| Independent expected-rank source | Missing. Product QR diagnostics alone are not accepted as expected metadata. |
| Threshold semantics and absolute-threshold diagnostics | Missing for any corpus rank-deficient claim. |
| Factorization expectations and failure interpretation | Defined by Day 8 policy, but no accepted candidate satisfies earlier metadata gates. |
| Optional-data skip behavior | Defined by Day 8 policy; no optional-large candidate is accepted. |
| Focused and full validation commands | Focused QR solve diagnostics run; no C changes, so full gate is not required for Day 9. |

## Candidate Review

| Candidate | Disposition | Reason |
| --- | --- | --- |
| `west0067.mtx` | Rejected as rank-deficient evidence | Focused QR solve diagnostic reports rank `67` for a 67 x 67 matrix. It remains a QR solve and sparse-mode full-rank control. |
| `nos4.mtx` | Rejected as rank-deficient evidence | Focused QR solve diagnostic reports rank `100` for a 100 x 100 matrix. It remains a QR solve, QR-vs-LU, refine, economy, reorder/fillin, and sparse-mode control. |
| `bcsstk04.mtx` | Rejected as rank-deficient evidence | Focused QR solve diagnostic reports rank `132` for a 132 x 132 matrix, and the product test asserts full rank. It remains a reconstruction, solve, and sparse-mode control. |
| `steam1.mtx` | Deferred | Default checked-in data exists, but there is no independent expected-rank/nullity or threshold/rank metadata for a QR rank-deficient claim. |
| `fs_541_1.mtx` | Deferred | Optional-large support tier remains behind `SPARSE_TEST_LARGE=1`; no independent expected-rank metadata or QR-specific runtime/skip proof is pinned. |
| `orsirr_1.mtx` | Deferred | Optional-large support tier remains behind `SPARSE_TEST_LARGE=1`; no independent expected-rank metadata or QR-specific runtime/skip proof is pinned. |
| `bcsstk14.mtx` | Deferred | Report-only for this claim; too large for default QR rank-deficient evidence without support-tier promotion and expected-rank metadata. |
| `s3rmt3m3.mtx` | Deferred | Report-only; no QR rank-deficient owner or independent expected-rank metadata. |
| `Kuu.mtx` | Deferred | Report-only; no QR rank-deficient owner or independent expected-rank metadata. |
| `bloweybq.mtx` | Deferred | Report-only; no QR rank-deficient owner or independent expected-rank metadata. |
| `Pres_Poisson.mtx` | Deferred | Report-only; no QR rank-deficient owner or independent expected-rank metadata. |
| `tuma1.mtx` | Deferred | Report-only; no QR rank-deficient owner or independent expected-rank metadata. |

## Focused Diagnostics

Day 9 ran the current QR solve executable to preserve the control-matrix
diagnostics:

```text
$ make build/test_qr_solve && ./build/test_qr_solve
nos4 QR solve: rank=100
bcsstk04: rank=132
west0067: rank=67
Tests run:    19
Tests failed: 0
Tests skipped: 0
Assertions:   1104
ALL TESTS PASSED
```

These diagnostics are product observations, not independent expected-rank
metadata for a new corpus claim.

## Future Promotion Gate

A future sprint may promote SuiteSparse rank-deficient QR evidence only after
all of the following are available:

1. A named matrix path and support tier.
2. Independent expected rank and nullity, or explicit threshold/rank pairs.
3. Threshold semantics with relative and absolute threshold diagnostics.
4. Factorization, rank, `R` diagonal, and residual or reconstruction
   diagnostics as appropriate to the claim.
5. Default, optional-large, or report-only skip behavior proven separately
   from numerical failure behavior.
6. A focused validation command for the exact executable path.
7. Full `make format && make lint && make test` validation if `.c` or `.h`
   files change.

## Evidence Preserved

The following existing evidence remains valid but is not relabeled:

- `west0067.mtx`, `nos4.mtx`, and `bcsstk04.mtx` remain full-rank QR controls.
- Sprint 125 and Sprint 126 dense/synthetic rank-deficient fixtures remain the
  source of bounded rank-deficient QR evidence.
- Day 8 remains the support-tier, metadata, diagnostics, and skip-behavior gate
  for any future SuiteSparse rank-deficient QR promotion.

## Non-Claims Preserved

- No SuiteSparse rank-deficient QR evidence is accepted in Day 9.
- No broad SuiteSparse corpus correctness claim.
- No broad QR rank-deficient behavior claim.
- No global QR rank-threshold policy.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, or external package parity claim.
- No minimum-norm, pseudoinverse, nullspace, Q-basis, economy, sparse-mode,
  reorder, backend, platform, performance, scalability, memory, package, ABI,
  public API, CI, CMake, or Makefile claim.

## Validation

Focused QR diagnostic validation passed:

```text
make build/test_qr_solve && ./build/test_qr_solve
```

Day 9 changed documentation only. Required documentation validation:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_126
```

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| SuiteSparse rank-deficient QR work is accepted only under Day 8 rules. | Complete by deferral | No candidate satisfies the expected-rank metadata and independent-source requirements. |
| Checked-in and optional-data behavior is explicit. | Complete | Candidate review separates default full-rank controls, default deferred candidates, optional-large matrices, and report-only matrices. |
| Broad corpus, platform, and performance claims remain absent. | Complete | See non-claims and preserved evidence boundaries. |
