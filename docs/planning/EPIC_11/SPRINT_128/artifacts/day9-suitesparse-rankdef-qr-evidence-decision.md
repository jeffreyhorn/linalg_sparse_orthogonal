# Sprint 128 Day 9 SuiteSparse Rank-Deficient QR Evidence Decision

## Decision

Day 9 explicitly defers SuiteSparse rank-deficient QR corpus evidence.

No checked-in SuiteSparse matrix satisfies the Day 8 metadata gate for a new
rank-deficient QR fixture. Existing SuiteSparse QR matrices remain controls,
and the other default, optional-large, and report-only candidates still lack
independent expected-rank metadata, threshold/rank pairs, support-tier
promotion, runtime budget, deterministic skip proof, and focused QR
diagnostics required before registration.

Day 9 therefore does not add or modify C tests, headers, Python helpers,
Matrix Market fixtures, optional-data gates, public wording, or maintainer
claims.

## Day 8 Gate Result

| Requirement | Result |
| --- | --- |
| Matrix path, shape, nnz, and support tier | Available for the checked-in corpus inventory. |
| Claim type | No rank-deficient QR claim accepted. |
| Expected rank/nullity or threshold/rank pairs | Missing for every plausible rank-deficient SuiteSparse candidate. |
| Independent expected-rank source | Missing. Product QR diagnostics remain observations only. |
| Threshold semantics and absolute-threshold diagnostics | Defined by Day 8, but no candidate has metadata for an accepted threshold claim. |
| Runtime budget | Defined by Day 8 support-tier policy, but no candidate satisfies the metadata gates. |
| Optional-data skip behavior | Defined by Day 8 policy; no optional-large candidate is accepted. |
| Focused and full validation commands | Focused QR solve diagnostics passed; no code files changed for Day 9. |

## Candidate Review

| Candidate | Day 9 disposition | Reason |
| --- | --- | --- |
| `west0067.mtx` | Rejected as rank-deficient evidence | Focused QR solve diagnostic reports rank `67` for a 67 x 67 matrix. It remains a QR solve and sparse-mode full-rank control. |
| `nos4.mtx` | Rejected as rank-deficient evidence | Focused QR solve diagnostic reports rank `100` for a 100 x 100 matrix. It remains a QR solve, QR-vs-LU, refine, economy, reorder/fillin, and sparse-mode control. |
| `bcsstk04.mtx` | Rejected as rank-deficient evidence | Focused QR solve diagnostic reports rank `132` for a 132 x 132 matrix, and product tests assert full-rank behavior. It remains a reconstruction, solve, and sparse-mode control. |
| `steam1.mtx` | Deferred | Default checked-in data exists, but independent expected-rank/nullity or threshold/rank metadata for a QR rank-deficient claim is unavailable. |
| `fs_541_1.mtx` | Deferred | Optional-large support tier remains behind `SPARSE_TEST_LARGE=1`; no independent expected-rank metadata, QR-specific runtime budget, or skip proof is pinned. |
| `orsirr_1.mtx` | Deferred | Optional-large support tier remains behind `SPARSE_TEST_LARGE=1`; no independent expected-rank metadata, QR-specific runtime budget, or skip proof is pinned. |
| `bcsstk14.mtx` | Deferred | Report-only for this lane; too large for default QR rank-deficient evidence without support-tier promotion and expected-rank metadata. |
| `s3rmt3m3.mtx` | Deferred | Report-only; no QR rank-deficient owner or independent expected-rank metadata. |
| `Kuu.mtx` | Deferred | Report-only; no QR rank-deficient owner or independent expected-rank metadata. |
| `bloweybq.mtx` | Deferred | Report-only; no QR rank-deficient owner or independent expected-rank metadata. |
| `Pres_Poisson.mtx` | Deferred | Report-only; no QR rank-deficient owner or independent expected-rank metadata. |
| `tuma1.mtx` | Deferred | Report-only; no QR rank-deficient owner or independent expected-rank metadata. |

## Focused Diagnostics

Day 9 ran the existing QR solve executable to preserve control-matrix
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

## Optional-Data Behavior

No optional-large SuiteSparse QR evidence is accepted on Day 9.

Future optional-large promotion must keep missing-data behavior deterministic:

1. The gate must be explicit, such as `SPARSE_TEST_LARGE=1` or a narrower
   QR-specific opt-in.
2. A missing matrix may skip only before the numerical claim is active and
   only with a message naming the matrix, gate, and owner.
3. Once data is present and accepted metadata exists, load, factorization, or
   rank disagreement must fail rather than skip.
4. Runtime expectations must be recorded before adding the path to default or
   optional CI.

## Future Promotion Gate

A future sprint may promote SuiteSparse rank-deficient QR evidence only after
all of the following are available:

1. A named matrix path and support tier.
2. Independent expected rank and nullity, or explicit threshold/rank pairs.
3. Threshold semantics with relative and absolute threshold diagnostics.
4. Load, factorization, rank, R diagonal, residual, and reconstruction
   diagnostics as appropriate to the claim.
5. Runtime budget for the selected support tier.
6. Default, optional-large, or report-only skip behavior proven separately
   from numerical failure behavior.
7. A focused validation command for the exact executable path.
8. Full `make format && make lint && make test` validation if `.c` or `.h`
   files change.

## Evidence Preserved

Existing SuiteSparse QR controls remain valid as controls, not as
rank-deficient evidence:

- `west0067.mtx` remains a full-rank QR solve and sparse-mode control.
- `nos4.mtx` remains a full-rank QR solve, QR-vs-LU, refine, economy,
  reorder/fillin, and sparse-mode control.
- `bcsstk04.mtx` remains a full-rank QR reconstruction, solve, and
  sparse-mode control.
- Sprint 125-128 dense and synthetic fixtures remain the source of bounded
  rank-deficient QR evidence.

## Non-Claims Preserved

- No SuiteSparse rank-deficient QR evidence is accepted in Day 9.
- No broad SuiteSparse corpus correctness claim.
- No broad QR rank-deficient behavior claim.
- No global QR rank-threshold, default-threshold, or numerical-rank policy.
- No LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, ecosystem, or external package parity claim.
- No residual, compatible-solve, minimum-norm, pseudoinverse, nullspace,
  subspace, Q-basis, economy, sparse-mode, reorder, backend, platform,
  performance, scalability, memory, package, ABI, public API, CI, CMake, or
  Makefile claim.

## Validation Notes

Focused QR diagnostic validation passed:

```text
make build/test_qr_solve && ./build/test_qr_solve
```

Day 9 changed documentation only. Required documentation validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_128`

No `.c`, `.h`, Python helper, build, public API, maintainer, Matrix Market,
optional-data, or public wording files changed for Day 9.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Accepted corpus evidence follows the Day 8 gate exactly. | Complete by deferral | No accepted candidate satisfies the Day 8 metadata gate. |
| Optional data never causes reviewed checks to fail unpredictably. | Complete | No optional-large evidence is accepted; future optional behavior is gated explicitly. |
| Corpus evidence does not imply broad external-library parity. | Complete | See candidate review, evidence preserved, and non-claims. |
